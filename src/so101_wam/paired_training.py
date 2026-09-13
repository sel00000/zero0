"""Action-free human-video paired training for offline candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .checkpoint import compact_wam_architecture
from .constants import ACTION_DIM, PRIMARY_CAMERA_COUNT
from .dataset import EpisodeData
from .deployment import canonical_json_sha256
from .model import ActionRangeConstraint, CompactWAM, WAMLossWeights
from .paired_data import (
    PAIRED_DATA_KIND,
    PAIRED_DATA_SCHEMA,
    HumanRobotPairRecord,
    SemanticMatchStatus,
    paired_episode_records,
    validate_pair_disjoint_split,
)
from .paired_task_specs import PairedTaskSpecError, human_prompt_from_pair
from .tensorizer import TensorizerError, tensorize_task_spec
from .training import (
    TRAINING_REPORT_SCHEMA,
    CompactWAMTrainingConfig,
    SamplingStrategy,
    TrainingArtifactExistsError,
    TrainingError,
    _optimizer_step,
    _publish_candidate_pair,
    _seed_everything,
    _state_dict_sha256,
)
from .training_data import (
    CompactWAMTrainingBatch,
    action_source_summary,
    episode_action_history,
    training_axis_statistics,
)
from .vision import COMPACT_IMAGE_MAX_SIDE, compact_rgb_image


class PairedTrainingError(ValueError):
    """Raised when a human/robot pair cannot form honest training input."""


class PairedWindowMode(StrEnum):
    """Select which complete temporal windows each reviewed pair contributes."""

    EARLIEST = "earliest"
    ALL_COMPLETE = "all_complete"


_PROMPT_LEN_CONTROL = "nearest_uniform_index_resample_to_matched_prompt_length"


@dataclass(frozen=True, slots=True)
class PairedTrainingArtifacts:
    """Published paired candidate paths and held-out offline metrics."""

    checkpoint_path: str
    report_path: str
    checkpoint_id: str
    training_evidence_sha256: str
    train_pairs: int
    validation_pairs: int
    optimizer_steps: int
    validation_action_mse_normalized: float
    validation_action_mae_native: float
    validation_future_latent_mse: float
    trained: bool = False
    deployment_ready: bool = False


@dataclass(frozen=True, slots=True)
class _PairedBatch:
    pair: HumanRobotPairRecord
    batch: CompactWAMTrainingBatch
    anchor_policy_position: int = 0
    anchor_time_s: float = 0.0


def paired_training_batch(
    pair: HumanRobotPairRecord,
    *,
    neutral_axes: np.ndarray | list[float] | tuple[float, ...],
    policy_hz: float,
    servo_hz: float,
    action_history_steps: int,
    future_steps: int,
    action_horizon: int,
    max_context_steps: int = 300,
    anchor_policy_position: int | None = None,
) -> CompactWAMTrainingBatch:
    """Materialize one complete temporal batch from one reviewed pair."""

    _validate_inputs(
        pair,
        policy_hz=policy_hz,
        servo_hz=servo_hz,
        action_history_steps=action_history_steps,
        future_steps=future_steps,
        action_horizon=action_horizon,
        max_context_steps=max_context_steps,
    )
    try:
        prompt = human_prompt_from_pair(pair)
    except PairedTaskSpecError as error:
        raise PairedTrainingError(str(error)) from error
    if len(prompt.frames) + action_history_steps > max_context_steps:
        raise PairedTrainingError(
            "human prompt plus live history exceeds max_context_steps"
        )

    target = pair.robot_episode.data
    policy_indices = _nearest_rate_indices(target, rate_hz=policy_hz)
    anchor = _anchor_position(
        policy_indices,
        action_history_steps=action_history_steps,
        future_steps=future_steps,
        anchor_policy_position=anchor_policy_position,
    )

    history_positions = range(anchor - action_history_steps + 1, anchor + 1)
    live_indices = tuple(policy_indices[position] for position in history_positions)
    future_indices = tuple(
        policy_indices[anchor + offset]
        for offset in range(1, future_steps + 1)
    )
    anchor_index = policy_indices[anchor]
    anchor_time_s = float(target.timestamps_s[anchor_index])
    final_action_time_s = anchor_time_s + (action_horizon - 1) / servo_hz
    if final_action_time_s > float(target.timestamps_s[-1]) + 1e-9:
        raise PairedTrainingError(
            "robot episode has no complete action target horizon"
        )

    target_frames = tuple(target.frames())
    history_actions = episode_action_history(target, live_indices)
    live_frames = tuple(
        replace(target_frames[index], executed_action=history_actions[position])
        for position, index in enumerate(live_indices)
    )
    try:
        model_batch = tensorize_task_spec(
            prompt,
            live_frames,
            neutral_axes=neutral_axes,
        ).batch
    except TensorizerError as error:
        raise PairedTrainingError(str(error)) from error

    target_future_images = _image_tensor(target, future_indices)
    if model_batch.prompt_images.shape[-2:] != target_future_images.shape[-2:]:
        raise PairedTrainingError(
            "human prompt and robot preprocessing resolutions differ"
        )

    action_times_s = anchor_time_s + (
        np.arange(action_horizon, dtype=np.float64) / servo_hz
    )
    target_actions = np.stack(
        [
            np.interp(
                action_times_s,
                target.timestamps_s,
                target.action[:, axis],
            )
            for axis in range(ACTION_DIM)
        ],
        axis=-1,
    ).astype(np.float32)

    return CompactWAMTrainingBatch(
        prompt_images=model_batch.prompt_images,
        prompt_proprio=model_batch.prompt_proprio,
        prompt_actions=model_batch.prompt_actions,
        live_images=model_batch.live_images,
        live_proprio=model_batch.live_proprio,
        live_actions=model_batch.live_actions,
        prompt_mask=model_batch.prompt_mask,
        target_future_images=target_future_images,
        target_actions=torch.from_numpy(target_actions).unsqueeze(0),
        target_ifp_images=None,
    )


def train_paired_candidate(
    train_pairs: Sequence[HumanRobotPairRecord],
    validation_pairs: Sequence[HumanRobotPairRecord],
    *,
    checkpoint_path: str | Path,
    report_path: str | Path,
    checkpoint_id: str,
    config: CompactWAMTrainingConfig,
    device: str | torch.device = "cpu",
    window_mode: PairedWindowMode = PairedWindowMode.EARLIEST,
) -> PairedTrainingArtifacts:
    """Train and immutably publish one multi-pair offline candidate."""

    checkpoint_target = Path(checkpoint_path).resolve()
    report_target = Path(report_path).resolve()
    _require_new_artifacts(checkpoint_target, report_target)
    _require_training_scope(train_pairs, validation_pairs)
    _require_paired_config(config)
    _require_window_mode(window_mode)
    checkpoint_id = _checkpoint_id(checkpoint_id)

    split = validate_pair_disjoint_split(train_pairs, validation_pairs)
    train_records = paired_episode_records(split.train)
    validation_records = paired_episode_records(split.validation)
    axis_mean, axis_scale = training_axis_statistics(train_records)
    train_batches = _paired_batches(
        split.train,
        neutral_axes=axis_mean,
        config=config,
        window_mode=window_mode,
    )
    validation_batches = _paired_batches(
        split.validation,
        neutral_axes=axis_mean,
        config=config,
        window_mode=window_mode,
    )
    _require_shared_image_shape((*train_batches, *validation_batches))
    target_device = _training_device(device)

    _seed_everything(config.seed)
    model = CompactWAM(
        latent_dim=config.latent_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        action_history_steps=config.action_history_steps,
        ifp_steps=0,
        max_context_steps=config.max_context_steps,
        action_decoder=config.action_decoder,
    ).to(target_device)
    model.set_axis_normalization(
        torch.from_numpy(axis_mean),
        torch.from_numpy(axis_scale),
    )
    initial_model_state_sha256 = _state_dict_sha256(model)
    _seed_everything(config.seed)

    rng = np.random.default_rng(config.seed)
    stage1_schedule = _draw_pair_schedule(
        train_batches,
        steps=config.stage1_steps,
        rng=rng,
        window_mode=window_mode,
    )
    stage2_schedule = _draw_pair_schedule(
        train_batches,
        steps=config.stage2_steps,
        rng=rng,
        window_mode=window_mode,
    )
    stage1_loss = _train_paired_inverse_dynamics(
        model,
        stage1_schedule,
        config=config,
        device=target_device,
    )
    stage2_metrics = _train_paired_end_to_end(
        model,
        stage2_schedule,
        config=config,
        device=target_device,
    )
    validation_metrics, mismatch_pairs = _evaluate_paired(
        model,
        validation_batches,
        device=target_device,
    )

    full_schedule = (*stage1_schedule, *stage2_schedule)
    schedule_sha256 = canonical_json_sha256(
        {
            "stage1": [
                _schedule_identity(item, window_mode=window_mode)
                for item in stage1_schedule
            ],
            "stage2": [
                _schedule_identity(item, window_mode=window_mode)
                for item in stage2_schedule
            ],
        }
    )
    train_tasks = _task_inventory(split.train)
    validation_tasks = _task_inventory(split.validation)
    action_target_source = (
        "paired_robot_episode_action;"
        f"train={action_source_summary(train_records)};"
        f"validation={action_source_summary(validation_records)}"
    )
    report_core: dict[str, Any] = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "evidence_level": "offline",
        "artifact_kind": "compact_wam_candidate",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": checkpoint_id,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "protocol": {
            "prompt_pairing": "human_video_reviewed_pair",
            "prompt_modality": "human_video_task_spec",
            "prompt_robot_signals": "neutral_train_mean",
            "human_prompt_robot_signal_policy": (
                "forbidden_by_schema_and_checksum_rechecked"
            ),
            "paired_batch_scope": _window_scope(window_mode),
            "paired_window_mode": window_mode.value,
            "split": "task_disjoint",
            "sampling_strategy": config.sampling_strategy.value,
            "stage1": "ground_truth_future_latent_inverse_dynamics",
            "stage2": "human_prompt_future_latent_action_mse",
            "ifp_architecture": "disabled_for_paired_candidate_v1",
            "ifp_module_removed_from_inference": model.ifp_head is None,
            "action_target_source": action_target_source,
            "action_row_zero_timing": "policy_anchor_immediate",
            "real_output_authorized": False,
        },
        "data": {
            "pair_source_schema": PAIRED_DATA_SCHEMA,
            "pair_artifact_kind": PAIRED_DATA_KIND,
            "semantic_match_required": SemanticMatchStatus.HUMAN_REVIEWED.value,
            "human_task_spec_required": True,
            "train_episode_count": len(train_records),
            "validation_episode_count": len(validation_records),
            "train_pair_count": len(split.train),
            "validation_pair_count": len(split.validation),
            "train_task_count": len(train_tasks),
            "validation_task_count": len(validation_tasks),
            "train_tasks": train_tasks,
            "validation_tasks": validation_tasks,
            "train_window_count": len(train_batches),
            "validation_window_count": len(validation_batches),
            "train_window_inventory_sha256": _window_inventory_sha256(
                train_batches,
                window_mode=window_mode,
            ),
            "validation_window_inventory_sha256": _window_inventory_sha256(
                validation_batches,
                window_mode=window_mode,
            ),
            "train_pair_digest": split.train_digest,
            "validation_pair_digest": split.validation_digest,
            "train_pairs": _pair_inventory(split.train),
            "validation_pairs": _pair_inventory(split.validation),
            "train_action_source": action_source_summary(train_records),
            "validation_action_source": action_source_summary(validation_records),
            "sampling_audit": {
                "schema_version": "so101_wam.paired_task_sampling.v1",
                "strategy": config.sampling_strategy.value,
                "seed": config.seed,
                "optimizer_steps": len(full_schedule),
                "task_draw_counts": _pair_draw_counts(full_schedule),
                "window_draw_counts": _window_draw_counts(full_schedule),
                "schedule_sha256": schedule_sha256,
            },
        },
        "model": compact_wam_architecture(model),
        "initial_model_state_sha256": initial_model_state_sha256,
        "training_schedule_sha256": schedule_sha256,
        "normalization": {
            "axis_mean": [float(value) for value in axis_mean],
            "axis_scale": [float(value) for value in axis_scale],
        },
        "optimization": {
            **asdict(config),
            "device": str(target_device),
            "optimizer": "AdamW",
            "stage1_final_action_mse_normalized": stage1_loss,
            **stage2_metrics,
        },
        "validation": {
            **validation_metrics,
            "evaluation_scope": "task_disjoint_human_video_offline_proxy",
            "evaluation_window_weighting": "one_complete_window_one_vote",
            "mismatched_prompt_length_control": _PROMPT_LEN_CONTROL,
            "pair_digest": split.validation_digest,
            "mismatched_pairing": mismatch_pairs,
            "mismatched_pairing_sha256": canonical_json_sha256(
                {"pairs": mismatch_pairs}
            ),
            "prompt_causality_claimed": False,
            "semantic_task_success_evaluated": False,
        },
        "limitations": [
            "human semantic matches are imported reviewer declarations",
            "offline matched/null/mismatched deltas do not prove prompt causality",
            "offline losses do not certify semantic or real-world task success",
            "paired candidate is not authorized for real SO-101 output",
        ],
    }
    evidence_sha256 = canonical_json_sha256(report_core)
    metadata: dict[str, str | int | float | bool | None] = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "checkpoint_id": checkpoint_id,
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "training_evidence_sha256": evidence_sha256,
        "training_objective": "paired_human_future_latent_action_mse",
        "paired_human_video": True,
        "training_ifp_steps": 0,
        "inference_ifp_module_present": model.ifp_head is not None,
        "action_target_source": action_target_source,
        "train_task_count": len(train_tasks),
        "validation_task_count": len(validation_tasks),
        "train_pair_count": len(split.train),
        "validation_pair_count": len(split.validation),
        "optimizer_steps": len(full_schedule),
        "seed": config.seed,
        "sampling_strategy": config.sampling_strategy.value,
        "sampling_schedule_sha256": schedule_sha256,
        "paired_window_mode": window_mode.value,
        "validation_action_mse_normalized": validation_metrics["action_mse_normalized"],
        "action_decoder": model.action_decoder.value,
    }
    _publish_candidate_pair(
        model=model,
        metadata=metadata,
        report={
            **report_core,
            "training_evidence_sha256": evidence_sha256,
        },
        checkpoint_path=checkpoint_target,
        report_path=report_target,
    )

    return PairedTrainingArtifacts(
        checkpoint_path=str(checkpoint_target),
        report_path=str(report_target),
        checkpoint_id=checkpoint_id,
        training_evidence_sha256=evidence_sha256,
        train_pairs=len(split.train),
        validation_pairs=len(split.validation),
        optimizer_steps=len(full_schedule),
        validation_action_mse_normalized=validation_metrics["action_mse_normalized"],
        validation_action_mae_native=validation_metrics["action_mae_native"],
        validation_future_latent_mse=validation_metrics["future_latent_mse"],
    )


def paired_training_artifacts_as_json(
    artifacts: PairedTrainingArtifacts,
) -> Mapping[str, Any]:
    """Return a read-only JSON-compatible training result."""

    if not isinstance(artifacts, PairedTrainingArtifacts):
        raise PairedTrainingError("artifacts must use PairedTrainingArtifacts")
    return MappingProxyType(asdict(artifacts))


def _require_new_artifacts(checkpoint: Path, report: Path) -> None:
    if checkpoint == report:
        raise PairedTrainingError("checkpoint and report paths must be different")
    if checkpoint.exists() or report.exists():
        raise TrainingArtifactExistsError(
            "paired training artifacts are immutable; choose new "
            "checkpoint/report paths"
        )


def _require_training_scope(
    train_pairs: Sequence[HumanRobotPairRecord],
    validation_pairs: Sequence[HumanRobotPairRecord],
) -> None:
    if len(train_pairs) < 2:
        raise PairedTrainingError("paired training requires at least two train pairs")
    validation_tasks = {
        (pair.task_index, pair.task)
        for pair in validation_pairs
        if isinstance(pair, HumanRobotPairRecord)
    }
    if len(validation_tasks) < 2:
        raise PairedTrainingError(
            "paired training requires at least two validation tasks"
        )


def _require_paired_config(config: CompactWAMTrainingConfig) -> None:
    if not isinstance(config, CompactWAMTrainingConfig):
        raise PairedTrainingError("config must use CompactWAMTrainingConfig")
    if config.ifp_steps != 0:
        raise PairedTrainingError("paired human-video training requires ifp_steps=0")
    if config.sampling_strategy is not SamplingStrategy.TASK_BALANCED:
        raise PairedTrainingError(
            "paired human-video training requires task-balanced sampling"
        )


def _require_window_mode(window_mode: PairedWindowMode) -> None:
    if not isinstance(window_mode, PairedWindowMode):
        raise PairedTrainingError("window_mode must be a PairedWindowMode")


def _checkpoint_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PairedTrainingError("checkpoint_id must be a non-empty string")
    return value.strip()


def _paired_batches(
    pairs: Sequence[HumanRobotPairRecord],
    *,
    neutral_axes: np.ndarray,
    config: CompactWAMTrainingConfig,
    window_mode: PairedWindowMode = PairedWindowMode.EARLIEST,
) -> tuple[_PairedBatch, ...]:
    _require_window_mode(window_mode)
    items: list[_PairedBatch] = []
    for pair in pairs:
        anchors = _complete_anchors(pair, config=config, window_mode=window_mode)
        for anchor in anchors:
            items.append(
                _PairedBatch(
                    pair=pair,
                    batch=paired_training_batch(
                        pair,
                        neutral_axes=neutral_axes,
                        policy_hz=config.policy_hz,
                        servo_hz=config.servo_hz,
                        action_history_steps=config.action_history_steps,
                        future_steps=config.future_steps,
                        action_horizon=config.action_horizon,
                        max_context_steps=config.max_context_steps,
                        anchor_policy_position=anchor,
                    ),
                    anchor_policy_position=anchor,
                    anchor_time_s=_anchor_time_s(pair, config=config, anchor=anchor),
                )
            )
    return tuple(items)


def _require_shared_image_shape(batches: Sequence[_PairedBatch]) -> None:
    image_shapes = {tuple(item.batch.live_images.shape[-3:]) for item in batches}
    if len(image_shapes) != 1:
        raise PairedTrainingError(
            "all paired training images must share one resolution"
        )


def _training_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise PairedTrainingError(f"invalid training device: {value}") from error
    if device.type not in {"cpu", "cuda"}:
        raise PairedTrainingError("training device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairedTrainingError(f"requested CUDA device is unavailable: {device}")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise PairedTrainingError(f"requested CUDA device is unavailable: {device}")
    return device


def _shuffled_pair_stream(
    batches: Sequence[_PairedBatch],
    *,
    rng: np.random.Generator,
) -> Iterator[_PairedBatch]:
    if not batches:
        raise PairedTrainingError("paired training batches must be non-empty")
    while True:
        for index in rng.permutation(len(batches)):
            yield batches[int(index)]


def _pair_balanced_window_stream(
    batches: Sequence[_PairedBatch],
    *,
    rng: np.random.Generator,
) -> Iterator[_PairedBatch]:
    grouped: dict[str, list[_PairedBatch]] = defaultdict(list)
    for item in batches:
        grouped[item.pair.fingerprint].append(item)
    streams = {
        key: _shuffled_pair_stream(tuple(grouped[key]), rng=rng)
        for key in sorted(grouped)
    }
    while True:
        keys = tuple(streams)
        for index in rng.permutation(len(keys)):
            yield next(streams[keys[int(index)]])


def _draw_pair_schedule(
    batches: Sequence[_PairedBatch],
    *,
    steps: int,
    rng: np.random.Generator,
    window_mode: PairedWindowMode = PairedWindowMode.EARLIEST,
) -> tuple[_PairedBatch, ...]:
    if steps == 0:
        return ()

    _require_window_mode(window_mode)
    grouped: dict[tuple[int, str], list[_PairedBatch]] = defaultdict(list)
    for item in batches:
        grouped[(item.pair.task_index, item.pair.task)].append(item)
    streams = {
        key: _task_stream(tuple(grouped[key]), rng=rng, window_mode=window_mode)
        for key in sorted(grouped)
    }

    schedule: list[_PairedBatch] = []
    while len(schedule) < steps:
        keys = tuple(streams)
        for index in rng.permutation(len(keys)):
            schedule.append(next(streams[keys[int(index)]]))
            if len(schedule) == steps:
                break
    return tuple(schedule)


def _task_stream(
    batches: Sequence[_PairedBatch],
    *,
    rng: np.random.Generator,
    window_mode: PairedWindowMode,
) -> Iterator[_PairedBatch]:
    if window_mode is PairedWindowMode.EARLIEST:
        return _shuffled_pair_stream(batches, rng=rng)
    return _pair_balanced_window_stream(batches, rng=rng)


def _train_paired_inverse_dynamics(
    model: CompactWAM,
    schedule: Sequence[_PairedBatch],
    *,
    config: CompactWAMTrainingConfig,
    device: torch.device,
) -> float | None:
    if not schedule:
        return None

    optimizer = torch.optim.AdamW(
        model.action_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    final_loss = 0.0
    for item in schedule:
        batch = item.batch.to(device)
        with torch.no_grad():
            target_future = model.encoder(
                batch.target_future_images,
                segment_id=1,
            )
        current = model.normalize_axes(batch.live_proprio[:, -1, :])
        history = model.normalize_axes(
            batch.live_actions[:, -model.action_history_steps :, :]
        )
        target = model.normalize_axes(batch.target_actions)
        raw_predicted = model.action_head(target_future, current, history)
        predicted = raw_predicted
        if model.action_range_constraint is not ActionRangeConstraint.UNBOUNDED:
            predicted = model.normalize_axes(model.decode_actions(raw_predicted))
        loss = F.mse_loss(predicted, target)
        _optimizer_step(
            loss,
            optimizer=optimizer,
            parameters=model.action_head.parameters(),
            max_grad_norm=config.max_grad_norm,
        )
        final_loss = float(loss.detach().cpu())
    return final_loss


def _train_paired_end_to_end(
    model: CompactWAM,
    schedule: Sequence[_PairedBatch],
    *,
    config: CompactWAMTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    parameters = tuple(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weights = WAMLossWeights(
        future_latent=config.future_latent_weight,
        action=config.action_weight,
        ifp=config.ifp_weight,
    )
    model.train()
    final: dict[str, float] = {}
    for item in schedule:
        batch = item.batch.to(device)
        outputs = model(**batch.model_kwargs(), compute_ifp=False)
        with torch.no_grad():
            target_future = model.encoder(
                batch.target_future_images,
                segment_id=1,
            )
        losses = model.loss(
            outputs,
            target_future_latents=target_future,
            target_actions=batch.target_actions,
            target_ifp_latents=None,
            weights=weights,
            action_scale=cast(Tensor, model.axis_scale),
        )
        _optimizer_step(
            losses["total"],
            optimizer=optimizer,
            parameters=parameters,
            max_grad_norm=config.max_grad_norm,
        )
        final = {
            f"stage2_final_{name}_loss": float(value.detach().cpu())
            for name, value in losses.items()
        }
    return final


@torch.no_grad()
def _evaluate_paired(
    model: CompactWAM,
    batches: Sequence[_PairedBatch],
    *,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    ordered = tuple(
        sorted(batches, key=lambda item: (item.pair.task_index, item.pair.pair_id))
    )
    totals: defaultdict[str, float] = defaultdict(float)
    action_count = 0
    future_count = 0
    mismatch_pairs: list[dict[str, Any]] = []

    for item in ordered:
        mismatch = next(
            candidate
            for candidate in ordered
            if candidate.pair.task_index != item.pair.task_index
        )
        batch = item.batch.to(device)
        mismatch_batch = mismatch.batch.to(device)
        kwargs = dict(batch.model_kwargs())
        matched = model(**kwargs, compute_ifp=False)

        neutral = cast(Tensor, model.axis_mean).view(1, 1, -1)
        null_kwargs = {
            **kwargs,
            "prompt_images": torch.zeros_like(batch.prompt_images),
            "prompt_proprio": neutral.expand_as(batch.prompt_proprio),
            "prompt_actions": neutral.expand_as(batch.prompt_actions),
        }
        mismatch_kwargs = {
            **kwargs,
            "prompt_images": mismatch_batch.prompt_images,
            "prompt_proprio": mismatch_batch.prompt_proprio,
            "prompt_actions": mismatch_batch.prompt_actions,
            "prompt_mask": mismatch_batch.prompt_mask,
        }
        controlled_kwargs = {
            **kwargs,
            "prompt_images": _resample_prompt(
                mismatch_batch.prompt_images,
                target_steps=int(batch.prompt_images.shape[1]),
            ),
            "prompt_proprio": _resample_prompt(
                mismatch_batch.prompt_proprio,
                target_steps=int(batch.prompt_proprio.shape[1]),
            ),
            "prompt_actions": _resample_prompt(
                mismatch_batch.prompt_actions,
                target_steps=int(batch.prompt_actions.shape[1]),
            ),
            "prompt_mask": batch.prompt_mask,
        }
        null = model(**null_kwargs, compute_ifp=False)
        mismatched = model(**mismatch_kwargs, compute_ifp=False)
        controlled = model(**controlled_kwargs, compute_ifp=False)
        matched_future = _output_tensor(matched, "future_latents")
        matched_actions = _output_tensor(matched, "actions")
        null_future = _output_tensor(null, "future_latents")
        null_actions = _output_tensor(null, "actions")
        mismatch_future = _output_tensor(mismatched, "future_latents")
        mismatch_actions = _output_tensor(mismatched, "actions")
        controlled_future = _output_tensor(controlled, "future_latents")
        controlled_actions = _output_tensor(controlled, "actions")
        target_future = model.encoder(
            batch.target_future_images,
            segment_id=1,
        )

        normalized_error = (matched_actions - batch.target_actions) / cast(
            Tensor,
            model.axis_scale,
        )
        totals["action_squared"] += float(torch.square(normalized_error).sum().cpu())
        totals["action_absolute"] += float(
            torch.abs(matched_actions - batch.target_actions).sum().cpu()
        )
        totals["future_squared"] += float(
            torch.square(matched_future - target_future).sum().cpu()
        )
        totals["null_action_delta"] += float(
            torch.abs(matched_actions - null_actions).sum().cpu()
        )
        totals["null_future_delta"] += float(
            torch.abs(matched_future - null_future).sum().cpu()
        )
        totals["mismatch_action_delta"] += float(
            torch.abs(matched_actions - mismatch_actions).sum().cpu()
        )
        totals["mismatch_future_delta"] += float(
            torch.abs(matched_future - mismatch_future).sum().cpu()
        )
        totals["controlled_action_delta"] += float(
            torch.abs(matched_actions - controlled_actions).sum().cpu()
        )
        totals["controlled_future_delta"] += float(
            torch.abs(matched_future - controlled_future).sum().cpu()
        )
        action_count += int(matched_actions.numel())
        future_count += int(matched_future.numel())
        mismatch_pairs.append(
            {
                "validation_pair_id": item.pair.pair_id,
                "validation_pair_fingerprint": item.pair.fingerprint,
                "validation_anchor_policy_position": (
                    item.anchor_policy_position
                ),
                "validation_anchor_time_s": item.anchor_time_s,
                "validation_task": item.pair.task,
                "validation_task_index": item.pair.task_index,
                "mismatched_pair_id": mismatch.pair.pair_id,
                "mismatched_pair_fingerprint": mismatch.pair.fingerprint,
                "mismatched_anchor_policy_position": (
                    mismatch.anchor_policy_position
                ),
                "mismatched_anchor_time_s": mismatch.anchor_time_s,
                "mismatched_task": mismatch.pair.task,
                "mismatched_task_index": mismatch.pair.task_index,
                "prompt_length_control": _PROMPT_LEN_CONTROL,
                "matched_prompt_steps": int(batch.prompt_images.shape[1]),
                "mismatched_prompt_steps": int(
                    mismatch_batch.prompt_images.shape[1]
                ),
                "controlled_mismatched_prompt_steps": int(
                    controlled_kwargs["prompt_images"].shape[1]
                ),
            }
        )

    metrics = {
        "action_mse_normalized": totals["action_squared"] / action_count,
        "action_mae_native": totals["action_absolute"] / action_count,
        "future_latent_mse": totals["future_squared"] / future_count,
        "null_prompt_action_mean_abs_delta": (
            totals["null_action_delta"] / action_count
        ),
        "null_prompt_future_mean_abs_delta": (
            totals["null_future_delta"] / future_count
        ),
        "mismatched_prompt_action_mean_abs_delta": (
            totals["mismatch_action_delta"] / action_count
        ),
        "mismatched_prompt_future_mean_abs_delta": (
            totals["mismatch_future_delta"] / future_count
        ),
        "length_controlled_mismatched_prompt_action_mean_abs_delta": (
            totals["controlled_action_delta"] / action_count
        ),
        "length_controlled_mismatched_prompt_future_mean_abs_delta": (
            totals["controlled_future_delta"] / future_count
        ),
    }
    if not all(isfinite(value) for value in metrics.values()):
        raise TrainingError(f"paired validation produced non-finite metrics: {metrics}")
    return metrics, mismatch_pairs


def _resample_prompt(value: Tensor, *, target_steps: int) -> Tensor:
    source_steps = int(value.shape[1])
    if source_steps == target_steps:
        return value

    positions = torch.linspace(
        0,
        source_steps - 1,
        target_steps,
        device=value.device,
    )
    indices = torch.floor(positions + 0.5).long()
    return value.index_select(1, indices)


def _output_tensor(
    outputs: Mapping[str, Tensor | None],
    name: str,
) -> Tensor:
    value = outputs.get(name)
    if value is None:
        raise TrainingError(f"model output {name} is missing")
    return value


def _task_inventory(
    pairs: Sequence[HumanRobotPairRecord],
) -> list[dict[str, str | int]]:
    identities = sorted({(pair.task_index, pair.task) for pair in pairs})
    return [{"task_index": task_index, "task": task} for task_index, task in identities]


def _pair_inventory(
    pairs: Sequence[HumanRobotPairRecord],
) -> list[dict[str, str | int]]:
    inventory: list[dict[str, str | int]] = []
    for pair in sorted(pairs, key=lambda item: item.pair_id):
        if pair.human_task_spec is None:
            raise PairedTrainingError(
                "paired training requires a checksum-bound human task-spec"
            )
        inventory.append(
            {
                "pair_id": pair.pair_id,
                "pair_fingerprint": pair.fingerprint,
                "task": pair.task,
                "task_index": pair.task_index,
                "semantic_match": pair.semantic_match.value,
                "human_video_sha256": pair.human_video_sha256,
                "human_task_spec_sha256": pair.human_task_spec.sha256,
                "robot_episode_sha256": pair.robot_episode_sha256,
                "robot_episode_fingerprint": pair.robot_episode.fingerprint,
            }
        )
    return inventory


def _pair_draw_counts(
    schedule: Sequence[_PairedBatch],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in schedule:
        counts[f"{item.pair.task_index}:{item.pair.task}"] += 1
    return dict(sorted(counts.items()))


def _window_draw_counts(
    schedule: Sequence[_PairedBatch],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in schedule:
        counts[_window_key(item)] += 1
    return dict(sorted(counts.items()))


def _window_inventory_sha256(
    batches: Sequence[_PairedBatch],
    *,
    window_mode: PairedWindowMode,
) -> str:
    return canonical_json_sha256(
        {
            "paired_window_mode": window_mode.value,
            "windows": [_window_identity(item) for item in batches],
        }
    )


def _schedule_identity(
    item: _PairedBatch,
    *,
    window_mode: PairedWindowMode,
) -> str | dict[str, str | int | float]:
    if window_mode is PairedWindowMode.EARLIEST:
        return item.pair.fingerprint
    return {
        "paired_window_mode": window_mode.value,
        **_window_identity(item),
    }


def _window_identity(item: _PairedBatch) -> dict[str, str | int | float]:
    return {
        "pair_fingerprint": item.pair.fingerprint,
        "pair_id": item.pair.pair_id,
        "task": item.pair.task,
        "task_index": item.pair.task_index,
        "anchor_policy_position": item.anchor_policy_position,
        "anchor_time_s": item.anchor_time_s,
    }


def _window_key(item: _PairedBatch) -> str:
    return f"{item.pair.fingerprint}:{item.anchor_policy_position}"


def _window_scope(window_mode: PairedWindowMode) -> str:
    if window_mode is PairedWindowMode.EARLIEST:
        return "earliest_complete_anchor_per_pair"
    return "all_complete_anchors_per_pair"


def _complete_anchors(
    pair: HumanRobotPairRecord,
    *,
    config: CompactWAMTrainingConfig,
    window_mode: PairedWindowMode,
) -> tuple[int, ...]:
    data = pair.robot_episode.data
    policy_indices = _nearest_rate_indices(data, rate_hz=config.policy_hz)
    first = config.action_history_steps - 1
    last = len(policy_indices) - config.future_steps - 1
    if last < first:
        raise PairedTrainingError(
            "robot episode has no complete paired-training batch"
        )
    if window_mode is PairedWindowMode.EARLIEST:
        return (first,)

    anchors = tuple(
        anchor
        for anchor in range(first, last + 1)
        if _has_action_horizon(
            data,
            policy_indices=policy_indices,
            anchor_policy_position=anchor,
            servo_hz=config.servo_hz,
            action_horizon=config.action_horizon,
        )
    )
    if not anchors:
        raise PairedTrainingError(
            "robot episode has no complete action target horizon"
        )
    return anchors


def _anchor_time_s(
    pair: HumanRobotPairRecord,
    *,
    config: CompactWAMTrainingConfig,
    anchor: int,
) -> float:
    data = pair.robot_episode.data
    policy_indices = _nearest_rate_indices(data, rate_hz=config.policy_hz)
    return float(data.timestamps_s[policy_indices[anchor]])


def _anchor_position(
    policy_indices: Sequence[int],
    *,
    action_history_steps: int,
    future_steps: int,
    anchor_policy_position: int | None,
) -> int:
    anchor = action_history_steps - 1
    if anchor_policy_position is not None:
        if (
            not isinstance(anchor_policy_position, int)
            or isinstance(anchor_policy_position, bool)
        ):
            raise PairedTrainingError(
                "anchor_policy_position must be an integer"
            )
        anchor = anchor_policy_position
    if anchor < action_history_steps - 1:
        raise PairedTrainingError(
            "anchor_policy_position lacks complete action history"
        )
    if anchor + future_steps >= len(policy_indices):
        raise PairedTrainingError(
            "robot episode has no complete paired-training batch"
        )
    return anchor


def _has_action_horizon(
    data: EpisodeData,
    *,
    policy_indices: Sequence[int],
    anchor_policy_position: int,
    servo_hz: float,
    action_horizon: int,
) -> bool:
    anchor_index = policy_indices[anchor_policy_position]
    anchor_time_s = float(data.timestamps_s[anchor_index])
    final_action_time_s = anchor_time_s + (action_horizon - 1) / servo_hz
    return final_action_time_s <= float(data.timestamps_s[-1]) + 1e-9


def _validate_inputs(
    pair: HumanRobotPairRecord,
    *,
    policy_hz: float,
    servo_hz: float,
    action_history_steps: int,
    future_steps: int,
    action_horizon: int,
    max_context_steps: int,
) -> None:
    if not isinstance(pair, HumanRobotPairRecord):
        raise PairedTrainingError("pair must use HumanRobotPairRecord")
    for value, name in (
        (policy_hz, "policy_hz"),
        (servo_hz, "servo_hz"),
    ):
        if not isfinite(value) or value <= 0:
            raise PairedTrainingError(f"{name} must be finite and positive")
    for value, name in (
        (action_history_steps, "action_history_steps"),
        (future_steps, "future_steps"),
        (action_horizon, "action_horizon"),
        (max_context_steps, "max_context_steps"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PairedTrainingError(f"{name} must be a positive integer")
    if pair.semantic_match is not SemanticMatchStatus.HUMAN_REVIEWED:
        raise PairedTrainingError(
            "paired training requires semantic_match=human_reviewed"
        )
    if pair.human_task_spec is None:
        raise PairedTrainingError(
            "paired training requires a checksum-bound human task-spec"
        )
    if (
        pair.task != pair.robot_episode.data.task
        or pair.task_index != pair.robot_episode.data.task_index
    ):
        raise PairedTrainingError(
            "pair task identity does not match robot supervision"
        )
    if _file_sha256(pair.human_video_path) != pair.human_video_sha256:
        raise PairedTrainingError("human video checksum changed after import")
    if _file_sha256(pair.robot_episode.path) != pair.robot_episode_sha256:
        raise PairedTrainingError("robot episode checksum changed after import")


def _nearest_rate_indices(
    data: EpisodeData,
    *,
    rate_hz: float,
) -> tuple[int, ...]:
    if rate_hz > data.fps + 1e-9:
        raise PairedTrainingError(
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
            key=lambda index: abs(
                float(data.timestamps_s[index]) - float(target_s)
            ),
        )
        if not indices or nearest != indices[-1]:
            indices.append(nearest)
    if indices[-1] != data.frame_count - 1:
        indices.append(data.frame_count - 1)
    return tuple(indices)


def _image_tensor(data: EpisodeData, indices: Sequence[int]) -> Tensor:
    frames: list[np.ndarray] = []
    for index in indices:
        views = [
            np.transpose(
                compact_rgb_image(
                    data.wrist_rgb[index, view],
                    max_side=COMPACT_IMAGE_MAX_SIDE,
                ),
                (2, 0, 1),
            )
            for view in range(PRIMARY_CAMERA_COUNT)
        ]
        frames.append(np.stack(views, axis=0))
    return torch.from_numpy(np.stack(frames, axis=0).copy()).unsqueeze(0)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PairedTrainingError(f"failed to read paired artifact: {error}") from error
    return digest.hexdigest()


__all__ = [
    "PairedTrainingArtifacts",
    "PairedTrainingError",
    "PairedWindowMode",
    "paired_training_batch",
    "paired_training_artifacts_as_json",
    "train_paired_candidate",
]
