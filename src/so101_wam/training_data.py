"""Leakage-resistant episode pairing and window targets for CompactWAM."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from .constants import ACTION_DIM, PRIMARY_CAMERA_COUNT
from .dataset import DatasetError, EpisodeData, load_episode, physical_prompt_from_episode
from .vision import COMPACT_IMAGE_MAX_SIDE, compact_rgb_image

class ActionTiming(StrEnum):
    OBSERVATION_THEN_COMMAND = "observation_then_command"


ACTION_TIMING_KEY = "action_timing"
INITIAL_PREVIOUS_ACTION_KEY = "initial_previous_action"
OBSERVATION_THEN_COMMAND = ActionTiming.OBSERVATION_THEN_COMMAND.value
FLOAT32_MAX = float(np.finfo(np.float32).max)


class TrainingDataError(ValueError):
    """Raised when offline data cannot support an auditable ICL sample."""


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One checksum-validated immutable episode and its source path."""

    path: Path
    data: EpisodeData

    @property
    def fingerprint(self) -> str:
        return self.data.fingerprint


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    """Explicit task-disjoint train/validation episode sets."""

    train: tuple[EpisodeRecord, ...]
    validation: tuple[EpisodeRecord, ...]
    train_digest: str
    validation_digest: str


@dataclass(frozen=True, slots=True)
class EpisodePair:
    """Same-task, different-episode physical prompt and robot target."""

    prompt: EpisodeRecord
    target: EpisodeRecord
    prompt_policy_indices: tuple[int, ...]
    target_policy_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingWindowSpec:
    """A target-policy anchor with enough past and future supervision."""

    pair: EpisodePair
    anchor_policy_position: int
    action_history_steps: int
    future_steps: int
    action_horizon: int
    servo_hz: float
    ifp_steps: int
    ifp_stride: int


@dataclass(frozen=True, slots=True)
class CompactWAMTrainingBatch:
    """Single-sample teacher-forcing batch plus video/action targets."""

    prompt_images: Tensor
    prompt_proprio: Tensor
    prompt_actions: Tensor
    live_images: Tensor
    live_proprio: Tensor
    live_actions: Tensor
    prompt_mask: Tensor
    target_future_images: Tensor
    target_actions: Tensor
    target_ifp_images: Tensor | None

    def model_kwargs(self) -> Mapping[str, Tensor]:
        return {
            "prompt_images": self.prompt_images,
            "prompt_proprio": self.prompt_proprio,
            "prompt_actions": self.prompt_actions,
            "live_images": self.live_images,
            "live_proprio": self.live_proprio,
            "live_actions": self.live_actions,
            "prompt_mask": self.prompt_mask,
        }

    def to(self, device: torch.device | str) -> CompactWAMTrainingBatch:
        target = torch.device(device)
        return CompactWAMTrainingBatch(
            prompt_images=self.prompt_images.to(target),
            prompt_proprio=self.prompt_proprio.to(target),
            prompt_actions=self.prompt_actions.to(target),
            live_images=self.live_images.to(target),
            live_proprio=self.live_proprio.to(target),
            live_actions=self.live_actions.to(target),
            prompt_mask=self.prompt_mask.to(target),
            target_future_images=self.target_future_images.to(target),
            target_actions=self.target_actions.to(target),
            target_ifp_images=(
                None
                if self.target_ifp_images is None
                else self.target_ifp_images.to(target)
            ),
        )


def discover_episode_paths(inputs: Sequence[str | Path]) -> tuple[Path, ...]:
    """Resolve explicit NPZ files or directories without silent duplicates."""

    if not inputs:
        raise TrainingDataError("at least one episode file or directory is required")
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(candidate for candidate in path.glob("*.npz") if candidate.is_file()))
        elif path.is_file() and path.suffix == ".npz":
            paths.append(path)
        else:
            raise TrainingDataError(f"episode input is not an NPZ file or directory: {path}")
    if not paths:
        raise TrainingDataError("episode inputs contain no .npz files")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        absolute = path.resolve()
        if absolute in seen:
            raise TrainingDataError(f"duplicate episode path: {absolute}")
        seen.add(absolute)
        resolved.append(absolute)
    return tuple(resolved)


def load_episode_records(inputs: Sequence[str | Path]) -> tuple[EpisodeRecord, ...]:
    """Load every source through the dataset checksum/manifest validator."""

    records: list[EpisodeRecord] = []
    try:
        for path in discover_episode_paths(inputs):
            data = load_episode(path)
            _validate_timing(data)
            records.append(EpisodeRecord(path=path, data=data))
    except DatasetError as error:
        raise TrainingDataError(str(error)) from error
    return tuple(records)


def validate_task_disjoint_split(
    train: Sequence[EpisodeRecord],
    validation: Sequence[EpisodeRecord],
) -> EpisodeSplit:
    """Reject episode, task-id, task-label, and resolution leakage."""

    train_records = tuple(train)
    validation_records = tuple(validation)
    if not train_records or not validation_records:
        raise TrainingDataError("train and validation splits must both be non-empty")

    train_groups = _validate_episode_group(train_records, split_name="train")
    validation_groups = _validate_episode_group(
        validation_records, split_name="validation"
    )
    train_fingerprints = {record.fingerprint for record in train_records}
    validation_fingerprints = {record.fingerprint for record in validation_records}
    overlap = train_fingerprints & validation_fingerprints
    if overlap:
        raise TrainingDataError(
            f"train/validation episode fingerprint leakage: {sorted(overlap)}"
        )
    train_contents = {record.data.content_fingerprint for record in train_records}
    validation_contents = {
        record.data.content_fingerprint for record in validation_records
    }
    content_overlap = train_contents & validation_contents
    if content_overlap:
        raise TrainingDataError(
            f"train/validation episode content leakage: {sorted(content_overlap)}"
        )

    train_task_ids = {task_index for task_index, _ in train_groups}
    validation_task_ids = {task_index for task_index, _ in validation_groups}
    if train_task_ids & validation_task_ids:
        raise TrainingDataError(
            "train/validation task_index sets must be disjoint"
        )
    train_task_labels = {task for _, task in train_groups}
    validation_task_labels = {task for _, task in validation_groups}
    if train_task_labels & validation_task_labels:
        raise TrainingDataError("train/validation task labels must be disjoint")

    resolutions = {record.data.resolution for record in (*train_records, *validation_records)}
    if len(resolutions) != 1:
        raise TrainingDataError(
            f"all episodes must share one wrist resolution, got {sorted(resolutions)}"
        )

    return EpisodeSplit(
        train=train_records,
        validation=validation_records,
        train_digest=episode_split_digest(train_records),
        validation_digest=episode_split_digest(validation_records),
    )


def build_training_windows(
    records: Sequence[EpisodeRecord],
    *,
    policy_hz: float,
    servo_hz: float,
    action_history_steps: int,
    future_steps: int,
    action_horizon: int,
    ifp_steps: int,
    ifp_stride: int = 2,
    max_context_steps: int = 300,
) -> tuple[TrainingWindowSpec, ...]:
    """Build deterministic same-task/different-episode ICL windows."""

    _positive_rate(policy_hz, "policy_hz")
    _positive_rate(servo_hz, "servo_hz")
    for value, name in (
        (action_history_steps, "action_history_steps"),
        (future_steps, "future_steps"),
        (action_horizon, "action_horizon"),
        (ifp_stride, "ifp_stride"),
        (max_context_steps, "max_context_steps"),
    ):
        _positive_int(value, name)
    if not isinstance(ifp_steps, int) or isinstance(ifp_steps, bool) or ifp_steps < 0:
        raise TrainingDataError("ifp_steps must be a non-negative integer")

    groups = _validate_episode_group(tuple(records), split_name="window source")
    windows: list[TrainingWindowSpec] = []
    for group_key in sorted(groups):
        group_records = groups[group_key]
        ordered = tuple(
            sorted(
                group_records,
                key=lambda record: (record.data.episode_index, record.fingerprint),
            )
        )
        for target_index, target in enumerate(ordered):
            prompt = ordered[(target_index + 1) % len(ordered)]
            if prompt.fingerprint == target.fingerprint:
                raise TrainingDataError("prompt and target episode must be different")
            try:
                physical_prompt_from_episode(prompt.data, policy_hz=policy_hz)
            except DatasetError as error:
                raise TrainingDataError(
                    f"prompt episode {prompt.path} is invalid: {error}"
                ) from error

            prompt_indices = _nearest_rate_indices(prompt.data, rate_hz=policy_hz)
            target_indices = _nearest_rate_indices(target.data, rate_hz=policy_hz)
            if len(prompt_indices) + action_history_steps > max_context_steps:
                raise TrainingDataError(
                    "prompt plus live history exceeds max_context_steps for "
                    f"{prompt.path}"
                )
            pair = EpisodePair(
                prompt=prompt,
                target=target,
                prompt_policy_indices=prompt_indices,
                target_policy_indices=target_indices,
            )

            visual_offset = future_steps
            if ifp_steps:
                visual_offset = max(
                    visual_offset,
                    2 + (ifp_steps - 1) * ifp_stride,
                )
            before_count = len(windows)
            for anchor_position in range(
                action_history_steps - 1,
                len(target_indices) - visual_offset,
            ):
                anchor_index = target_indices[anchor_position]
                anchor_time_s = float(target.data.timestamps_s[anchor_index])
                final_action_time_s = anchor_time_s + (action_horizon - 1) / servo_hz
                if final_action_time_s > float(target.data.timestamps_s[-1]) + 1e-9:
                    continue
                windows.append(
                    TrainingWindowSpec(
                        pair=pair,
                        anchor_policy_position=anchor_position,
                        action_history_steps=action_history_steps,
                        future_steps=future_steps,
                        action_horizon=action_horizon,
                        servo_hz=float(servo_hz),
                        ifp_steps=ifp_steps,
                        ifp_stride=ifp_stride,
                    )
                )
            if len(windows) == before_count:
                raise TrainingDataError(
                    f"target episode has no complete training window: {target.path}"
                )
    return tuple(windows)


def draw_task_balanced_windows(
    windows: Sequence[TrainingWindowSpec],
    *,
    steps: int,
    rng: np.random.Generator,
    start_draw: int = 0,
) -> tuple[TrainingWindowSpec, ...]:
    """Draw windows by task first, then by within-task shuffled windows."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise TrainingDataError("steps must be a non-negative integer")
    if (
        not isinstance(start_draw, int)
        or isinstance(start_draw, bool)
        or start_draw < 0
    ):
        raise TrainingDataError("start_draw must be a non-negative integer")

    groups = _window_groups(windows)
    if steps == 0:
        return ()
    task_keys = sorted(groups)
    streams = {
        key: _shuffled_window_stream(groups[key], rng=rng)
        for key in task_keys
    }
    drawn: list[TrainingWindowSpec] = []
    draw_index = 0
    stop_draw = start_draw + steps
    while draw_index < stop_draw:
        order = list(task_keys)
        rng.shuffle(order)
        for key in order:
            if draw_index == stop_draw:
                break
            window = next(streams[key])
            if draw_index >= start_draw:
                drawn.append(window)
            draw_index += 1
    return tuple(drawn)


def task_draw_counts(
    windows: Sequence[TrainingWindowSpec],
) -> dict[str, int]:
    """Count training windows by stable task identity."""

    counts: dict[str, int] = {}
    for spec in windows:
        key = _task_key(spec)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def materialize_training_window(
    spec: TrainingWindowSpec,
    *,
    image_max_side: int = COMPACT_IMAGE_MAX_SIDE,
) -> CompactWAMTrainingBatch:
    """Create tensors and interpolate a 50 Hz action target deterministically."""

    pair = spec.pair
    anchor = spec.anchor_policy_position
    history_positions = range(anchor - spec.action_history_steps + 1, anchor + 1)
    live_indices = tuple(pair.target_policy_indices[position] for position in history_positions)
    future_indices = tuple(
        pair.target_policy_indices[anchor + offset]
        for offset in range(1, spec.future_steps + 1)
    )
    ifp_indices = tuple(
        pair.target_policy_indices[anchor + 2 + index * spec.ifp_stride]
        for index in range(spec.ifp_steps)
    )

    prompt_images = _image_tensor(
        pair.prompt.data,
        pair.prompt_policy_indices,
        image_max_side=image_max_side,
    )
    live_images = _image_tensor(
        pair.target.data,
        live_indices,
        image_max_side=image_max_side,
    )
    target_future_images = _image_tensor(
        pair.target.data,
        future_indices,
        image_max_side=image_max_side,
    )
    target_ifp_images = (
        None
        if not ifp_indices
        else _image_tensor(
            pair.target.data,
            ifp_indices,
            image_max_side=image_max_side,
        )
    )
    if prompt_images.shape[-2:] != live_images.shape[-2:]:
        raise TrainingDataError("prompt and target preprocessing resolutions differ")

    anchor_index = pair.target_policy_indices[anchor]
    anchor_time_s = float(pair.target.data.timestamps_s[anchor_index])
    # ServoExecutor consumes row zero at the policy deadline, followed by one
    # row per servo period. Label times mirror that exact runtime schedule.
    action_times_s = anchor_time_s + (
        np.arange(spec.action_horizon, dtype=np.float64) / spec.servo_hz
    )
    target_actions = np.stack(
        [
            np.interp(
                action_times_s,
                pair.target.data.timestamps_s,
                pair.target.data.action[:, axis],
            )
            for axis in range(ACTION_DIM)
        ],
        axis=-1,
    ).astype(np.float32)

    return CompactWAMTrainingBatch(
        prompt_images=prompt_images,
        prompt_proprio=_axis_tensor(pair.prompt.data.joint_state, pair.prompt_policy_indices),
        prompt_actions=_axis_tensor(pair.prompt.data.action, pair.prompt_policy_indices),
        live_images=live_images,
        live_proprio=_axis_tensor(pair.target.data.joint_state, live_indices),
        live_actions=torch.from_numpy(
            episode_action_history(pair.target.data, live_indices)
        ).unsqueeze(0),
        prompt_mask=torch.ones((1, len(pair.prompt_policy_indices)), dtype=torch.bool),
        target_future_images=target_future_images,
        target_actions=torch.from_numpy(target_actions).unsqueeze(0),
        target_ifp_images=target_ifp_images,
    )


def episode_action_history(
    data: EpisodeData,
    indices: Sequence[int],
) -> np.ndarray:
    """Return actions already available at each observation index."""

    timing = _action_timing(data)
    if timing is None:
        return _indexed_actions(data.action, indices)

    initial = _initial_previous_action(data)
    rows: list[np.ndarray] = []
    for index in indices:
        _valid_index(data, index)
        if index == 0:
            rows.append(initial)
            continue
        rows.append(data.action[index - 1])
    return np.stack(rows, axis=0).astype(np.float32)


def training_axis_statistics(
    records: Sequence[EpisodeRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only position/action normalization with a safe unit floor."""

    if not records:
        raise TrainingDataError("axis statistics require at least one episode")
    count = 0
    total = np.zeros(ACTION_DIM, dtype=np.float64)
    square_total = np.zeros(ACTION_DIM, dtype=np.float64)
    for record in sorted(records, key=lambda item: item.fingerprint):
        for values in (record.data.joint_state, record.data.action):
            array = np.asarray(values, dtype=np.float64)
            count += int(array.shape[0])
            total += array.sum(axis=0)
            square_total += np.square(array).sum(axis=0)
    mean = total / count
    variance = np.maximum(square_total / count - np.square(mean), 0.0)
    scale = np.maximum(np.sqrt(variance), 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def episode_split_digest(records: Sequence[EpisodeRecord]) -> str:
    """Hash content identities and task assignments, independent of file location."""

    payload = [
        {
            "episode_index": record.data.episode_index,
            "fingerprint": record.fingerprint,
            "task": record.data.task,
            "task_index": record.data.task_index,
        }
        for record in sorted(records, key=lambda item: item.fingerprint)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def action_source_summary(records: Iterable[EpisodeRecord]) -> str:
    """Return a scalar provenance label suitable for checkpoint metadata."""

    sources = {
        str(
            cast(Mapping[str, object], record.data.metadata).get(
                "action_source",
                "unspecified",
            )
        )
        for record in records
    }
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed:" + ",".join(sorted(sources))


def _validate_episode_group(
    records: tuple[EpisodeRecord, ...],
    *,
    split_name: str,
) -> dict[tuple[int, str], tuple[EpisodeRecord, ...]]:
    if not records:
        raise TrainingDataError(f"{split_name} episodes must be non-empty")
    fingerprints: set[str] = set()
    contents: set[str] = set()
    identities: set[tuple[int, int]] = set()
    id_to_label: dict[int, str] = {}
    label_to_id: dict[str, int] = {}
    groups: dict[tuple[int, str], list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        _validate_timing(record.data)
        if record.fingerprint in fingerprints:
            raise TrainingDataError(
                f"duplicate episode fingerprint in {split_name}: {record.fingerprint}"
            )
        fingerprints.add(record.fingerprint)
        content = record.data.content_fingerprint
        if content in contents:
            raise TrainingDataError(
                f"duplicate episode content in {split_name}: {content}"
            )
        contents.add(content)
        identity = (record.data.task_index, record.data.episode_index)
        if identity in identities:
            raise TrainingDataError(
                f"duplicate task/episode identity in {split_name}: {identity}"
            )
        identities.add(identity)
        previous_label = id_to_label.setdefault(record.data.task_index, record.data.task)
        previous_id = label_to_id.setdefault(record.data.task, record.data.task_index)
        if previous_label != record.data.task or previous_id != record.data.task_index:
            raise TrainingDataError(
                f"task labels and task_index values must be one-to-one in {split_name}"
            )
        groups[(record.data.task_index, record.data.task)].append(record)
    short = {key: len(value) for key, value in groups.items() if len(value) < 2}
    if short:
        raise TrainingDataError(
            "each task needs at least two episodes for different prompt/target pairing: "
            f"{short}"
        )
    return {key: tuple(value) for key, value in groups.items()}


def _task_key(spec: TrainingWindowSpec) -> str:
    task_index = spec.pair.target.data.task_index
    task = spec.pair.target.data.task
    return f"{task_index}:{task}"


def _validate_timing(data: EpisodeData) -> None:
    timing = _action_timing(data)
    if timing is None:
        return
    _initial_previous_action(data)


def _action_timing(data: EpisodeData) -> ActionTiming | None:
    metadata = cast(Mapping[str, object], data.metadata)
    if ACTION_TIMING_KEY not in metadata:
        return None

    raw = metadata[ACTION_TIMING_KEY]
    if not isinstance(raw, str):
        raise TrainingDataError(f"unknown action_timing: {raw}")
    try:
        return ActionTiming(raw)
    except ValueError as error:
        raise TrainingDataError(f"unknown action_timing: {raw}") from error


def _initial_previous_action(data: EpisodeData) -> np.ndarray:
    metadata = cast(Mapping[str, object], data.metadata)
    raw = metadata.get(INITIAL_PREVIOUS_ACTION_KEY)
    if not isinstance(raw, list) or len(raw) != ACTION_DIM:
        raise TrainingDataError(
            f"initial_previous_action must contain {ACTION_DIM} values"
        )

    values = []
    for item in raw:
        if type(item) not in (float, int):
            raise TrainingDataError("initial_previous_action values must be finite")
        value = float(item)
        if not isfinite(value) or abs(value) > FLOAT32_MAX:
            raise TrainingDataError("initial_previous_action values must be finite")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _indexed_actions(actions: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    rows = []
    for index in indices:
        _valid_index_array(actions, index)
        rows.append(actions[index])
    return np.stack(rows, axis=0).astype(np.float32)


def _valid_index(data: EpisodeData, index: int) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TrainingDataError("action history index must be an integer")
    if index < 0 or index >= data.frame_count:
        raise TrainingDataError("action history index out of range")


def _valid_index_array(actions: np.ndarray, index: int) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TrainingDataError("action history index must be an integer")
    if index < 0 or index >= int(actions.shape[0]):
        raise TrainingDataError("action history index out of range")


def _window_groups(
    windows: Sequence[TrainingWindowSpec],
) -> dict[str, tuple[TrainingWindowSpec, ...]]:
    if not windows:
        raise TrainingDataError("task-balanced sampling requires at least one window")

    groups: dict[str, list[TrainingWindowSpec]] = defaultdict(list)
    for spec in windows:
        groups[_task_key(spec)].append(spec)
    return {key: tuple(value) for key, value in groups.items()}


def _shuffled_window_stream(
    windows: Sequence[TrainingWindowSpec],
    *,
    rng: np.random.Generator,
) -> Iterator[TrainingWindowSpec]:
    if not windows:
        raise TrainingDataError("window stream requires at least one window")
    while True:
        for index in rng.permutation(len(windows)):
            yield windows[int(index)]


def _nearest_rate_indices(data: EpisodeData, *, rate_hz: float) -> tuple[int, ...]:
    if rate_hz > data.fps + 1e-9:
        raise TrainingDataError(
            f"policy_hz {rate_hz:g} cannot exceed episode fps {data.fps:g}"
        )
    start_s = float(data.timestamps_s[0])
    end_s = float(data.timestamps_s[-1])
    target_count = int(np.floor((end_s - start_s) * rate_hz + 1e-9)) + 1
    targets = start_s + np.arange(target_count, dtype=np.float64) / rate_hz
    indices: list[int] = []
    for target_s in targets:
        insertion = int(np.searchsorted(data.timestamps_s, target_s, side="left"))
        candidates = tuple(
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < data.frame_count
        )
        nearest = min(
            candidates,
            key=lambda index: abs(float(data.timestamps_s[index]) - float(target_s)),
        )
        if not indices or nearest != indices[-1]:
            indices.append(nearest)
    if indices[-1] != data.frame_count - 1:
        indices.append(data.frame_count - 1)
    return tuple(indices)


def _image_tensor(
    data: EpisodeData,
    indices: Sequence[int],
    *,
    image_max_side: int,
) -> Tensor:
    frames: list[np.ndarray] = []
    for index in indices:
        views = [
            np.transpose(
                compact_rgb_image(data.wrist_rgb[index, view], max_side=image_max_side),
                (2, 0, 1),
            )
            for view in range(PRIMARY_CAMERA_COUNT)
        ]
        frames.append(np.stack(views, axis=0))
    return torch.from_numpy(np.stack(frames, axis=0).copy()).unsqueeze(0)


def _axis_tensor(values: np.ndarray, indices: Sequence[int]) -> Tensor:
    array = np.stack([values[index] for index in indices], axis=0).astype(
        np.float32, copy=True
    )
    return torch.from_numpy(array).unsqueeze(0)


def _positive_rate(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise TrainingDataError(f"{name} must be finite and positive")


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingDataError(f"{name} must be a positive integer")


__all__ = [
    "ACTION_TIMING_KEY",
    "INITIAL_PREVIOUS_ACTION_KEY",
    "OBSERVATION_THEN_COMMAND",
    "ActionTiming",
    "CompactWAMTrainingBatch",
    "EpisodePair",
    "EpisodeRecord",
    "EpisodeSplit",
    "TrainingDataError",
    "TrainingWindowSpec",
    "action_source_summary",
    "build_training_windows",
    "discover_episode_paths",
    "draw_task_balanced_windows",
    "episode_action_history",
    "episode_split_digest",
    "load_episode_records",
    "materialize_training_window",
    "task_draw_counts",
    "training_axis_statistics",
    "validate_task_disjoint_split",
]
