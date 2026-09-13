"""Offline prompt perturbation diagnostics for one CompactWAM checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoint import CheckpointError, load_compact_wam_bundle
from .config import ConfigError, ProjectConfig, SafetyConfig
from .constants import ACTION_DIM
from .context import Gen15Context
from .contracts import ContractError, PhysicalPrompt, SensorimotorFrame
from .dataset import (
    DatasetError,
    EpisodeData,
    load_episode,
    physical_prompt_from_episode,
)
from .deployment import file_sha256, project_config_sha256
from .model import ModelContractError
from .policy import CompactWAMPolicy, PolicyError
from .tensorizer import TensorizerError


SCHEMA_VERSION = 1
PROMPT_CONTROL_SCOPE = "diagnostic_prompt_control_action_divergence"
_SUITE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_MANIFEST_FIELDS = {
    "schema_version",
    "suite_id",
    "seed",
    "config",
    "checkpoint",
    "live_episode",
    "matched_prompt",
    "same_task_alternate_prompt",
    "wrong_task_prompt",
}


class PromptControlError(ValueError):
    """Raised when prompt-control inputs or evidence are invalid."""


class PromptCondition(StrEnum):
    MATCHED = "matched"
    SAME_TASK_ALTERNATE = "same_task_alternate"
    WRONG_TASK = "wrong_task"
    TEMPORAL_SHUFFLE = "temporal_shuffle"
    IMAGE_FRAME_SHUFFLE = "image_frame_shuffle"
    NULL = "null"
    COUNTERFACTUAL = "counterfactual"


@dataclass(frozen=True, slots=True)
class PromptControlManifest:
    suite_id: str
    seed: int
    config_path: Path
    checkpoint_path: Path
    live_episode_path: Path
    matched_prompt_path: Path
    same_task_alternate_prompt_path: Path
    wrong_task_prompt_path: Path


def _parse_manifest(source: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromptControlError(f"failed to read prompt-control manifest: {error}") from error
    if not isinstance(payload, dict):
        raise PromptControlError("prompt-control manifest must be a JSON object")
    if set(payload) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - set(payload))
        extra = sorted(set(payload) - _MANIFEST_FIELDS)
        raise PromptControlError(
            f"prompt-control manifest fields mismatch: missing={missing}, extra={extra}"
        )
    return payload


def _manifest_path(value: object, *, name: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PromptControlError(f"manifest {name} must be a non-empty path")
    raw_path = value.strip()
    relative_path = Path(raw_path)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != raw_path
    ):
        raise PromptControlError(
            f"manifest {name} must be a normalized bundle-relative path"
        )

    bundle_root = root.resolve()
    resolved_path = (bundle_root / relative_path).resolve()
    if not resolved_path.is_relative_to(bundle_root):
        raise PromptControlError(
            f"manifest {name} must remain within its bundle-relative root"
        )
    return resolved_path


def load_prompt_control_manifest_bytes(
    source: bytes,
    *,
    root: str | Path,
) -> PromptControlManifest:
    payload = _parse_manifest(source)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise PromptControlError("prompt-control manifest requires schema_version=1")
    suite_id = payload["suite_id"]
    if (
        not isinstance(suite_id, str)
        or _SUITE_ID_PATTERN.fullmatch(suite_id.strip()) is None
    ):
        raise PromptControlError("suite_id must be an artifact-safe identifier")
    seed = payload["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PromptControlError("seed must be a non-negative integer")
    bundle_root = Path(root)
    return PromptControlManifest(
        suite_id=suite_id.strip(),
        seed=seed,
        config_path=_manifest_path(
            payload["config"],
            name="config",
            root=bundle_root,
        ),
        checkpoint_path=_manifest_path(
            payload["checkpoint"],
            name="checkpoint",
            root=bundle_root,
        ),
        live_episode_path=_manifest_path(
            payload["live_episode"],
            name="live_episode",
            root=bundle_root,
        ),
        matched_prompt_path=_manifest_path(
            payload["matched_prompt"],
            name="matched_prompt",
            root=bundle_root,
        ),
        same_task_alternate_prompt_path=_manifest_path(
            payload["same_task_alternate_prompt"],
            name="same_task_alternate_prompt",
            root=bundle_root,
        ),
        wrong_task_prompt_path=_manifest_path(
            payload["wrong_task_prompt"],
            name="wrong_task_prompt",
            root=bundle_root,
        ),
    )


def load_prompt_control_manifest(path: str | Path) -> PromptControlManifest:
    target = Path(path)
    try:
        source = target.read_bytes()
    except OSError as error:
        raise PromptControlError(
            f"failed to read prompt-control manifest: {error}"
        ) from error
    return load_prompt_control_manifest_bytes(source, root=target.parent)


def validate_prompt_control_sources(
    live: EpisodeData,
    matched: EpisodeData,
    same_task_alternate: EpisodeData,
    wrong_task: EpisodeData,
) -> None:
    same_task_sources = (live, matched, same_task_alternate)
    task_identity = (live.task_index, live.task)
    if any(
        (episode.task_index, episode.task) != task_identity
        for episode in same_task_sources[1:]
    ):
        raise PromptControlError(
            "live, matched, and same-task alternate episodes must share one task"
        )
    if wrong_task.task == live.task or wrong_task.task_index == live.task_index:
        raise PromptControlError(
            "wrong-task episode must differ in both task label and task_index"
        )
    episodes = (*same_task_sources, wrong_task)
    fingerprints = {episode.fingerprint for episode in episodes}
    if len(fingerprints) != 4:
        raise PromptControlError("prompt-control episodes must be distinct")
    content_hashes = {_episode_content_sha256(episode) for episode in episodes}
    if len(content_hashes) != 4:
        raise PromptControlError(
            "prompt-control recordings must be content-distinct"
        )
    resolutions = {
        episode.resolution for episode in (*same_task_sources, wrong_task)
    }
    if len(resolutions) != 1:
        raise PromptControlError("prompt-control episodes must share one resolution")
    frame_rates = {episode.fps for episode in (*same_task_sources, wrong_task)}
    if len(frame_rates) != 1:
        raise PromptControlError("prompt-control episodes must share one FPS")


def _episode_content_sha256(episode: EpisodeData) -> str:
    digest = sha256()
    for name, value in (
        ("timestamps_s", episode.timestamps_s),
        ("wrist_rgb", episode.wrist_rgb),
        ("joint_state", episode.joint_state),
        ("action", episode.action),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _permutation(length: int, *, seed: int, salt: int) -> tuple[int, ...]:
    rng = np.random.default_rng(np.random.SeedSequence((seed, salt)))
    order = rng.permutation(length)
    if np.array_equal(order, np.arange(length)):
        order = np.roll(order, 1)
    return tuple(int(index) for index in order)


def _copy_frame(
    source: SensorimotorFrame,
    *,
    timestamp_s: float,
    images: Mapping[str, np.ndarray] | None = None,
    joint_position: np.ndarray | None = None,
    executed_action: np.ndarray | None = None,
) -> SensorimotorFrame:
    if source.executed_action is None:
        raise PromptControlError("prompt frame is missing executed_action")
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images=source.images if images is None else images,
        joint_position=(
            source.joint_position if joint_position is None else joint_position
        ),
        executed_action=(
            source.executed_action if executed_action is None else executed_action
        ),
    )


def _executed_action(frame: SensorimotorFrame) -> np.ndarray:
    if frame.executed_action is None:
        raise PromptControlError("prompt frame is missing executed_action")
    return frame.executed_action


def _temporal_shuffle(prompt: PhysicalPrompt, *, seed: int) -> PhysicalPrompt:
    order = _permutation(len(prompt.frames), seed=seed, salt=1)
    frames = tuple(
        _copy_frame(
            prompt.frames[source_index],
            timestamp_s=destination.timestamp_s,
        )
        for destination, source_index in zip(prompt.frames, order, strict=True)
    )
    return PhysicalPrompt(
        frames,
        min_duration_s=prompt.min_duration_s,
        max_duration_s=prompt.max_duration_s,
    )


def _image_frame_shuffle(prompt: PhysicalPrompt, *, seed: int) -> PhysicalPrompt:
    order = _permutation(len(prompt.frames), seed=seed, salt=2)
    frames = tuple(
        _copy_frame(
            destination,
            timestamp_s=destination.timestamp_s,
            images=prompt.frames[source_index].images,
        )
        for destination, source_index in zip(prompt.frames, order, strict=True)
    )
    return PhysicalPrompt(
        frames,
        min_duration_s=prompt.min_duration_s,
        max_duration_s=prompt.max_duration_s,
    )


def _null_prompt(prompt: PhysicalPrompt, *, safety: SafetyConfig) -> PhysicalPrompt:
    lower = np.asarray(safety.joint_lower, dtype=np.float32)
    upper = np.asarray(safety.joint_upper, dtype=np.float32)
    neutral = np.clip(np.zeros(ACTION_DIM, dtype=np.float32), lower, upper)
    frames = tuple(
        _copy_frame(
            frame,
            timestamp_s=frame.timestamp_s,
            images={key: np.zeros_like(image) for key, image in frame.images.items()},
            joint_position=neutral,
            executed_action=neutral,
        )
        for frame in prompt.frames
    )
    return PhysicalPrompt(
        frames,
        min_duration_s=prompt.min_duration_s,
        max_duration_s=prompt.max_duration_s,
    )


def _counterfactual_prompt(
    prompt: PhysicalPrompt,
    *,
    safety: SafetyConfig,
) -> PhysicalPrompt:
    lower = np.asarray(safety.joint_lower, dtype=np.float32)
    upper = np.asarray(safety.joint_upper, dtype=np.float32)

    def mirror(value: np.ndarray) -> np.ndarray:
        return lower + upper - np.asarray(value, dtype=np.float32)

    frames = tuple(
        _copy_frame(
            frame,
            timestamp_s=frame.timestamp_s,
            joint_position=mirror(frame.joint_position),
            executed_action=mirror(_executed_action(frame)),
        )
        for frame in prompt.frames
    )
    return PhysicalPrompt(
        frames,
        min_duration_s=prompt.min_duration_s,
        max_duration_s=prompt.max_duration_s,
    )


def build_prompt_conditions(
    matched: PhysicalPrompt,
    same_task_alternate: PhysicalPrompt,
    wrong_task: PhysicalPrompt,
    *,
    safety: SafetyConfig,
    seed: int,
) -> dict[PromptCondition, PhysicalPrompt]:
    return {
        PromptCondition.MATCHED: matched,
        PromptCondition.SAME_TASK_ALTERNATE: same_task_alternate,
        PromptCondition.WRONG_TASK: wrong_task,
        PromptCondition.TEMPORAL_SHUFFLE: _temporal_shuffle(matched, seed=seed),
        PromptCondition.IMAGE_FRAME_SHUFFLE: _image_frame_shuffle(
            matched, seed=seed
        ),
        PromptCondition.NULL: _null_prompt(matched, safety=safety),
        PromptCondition.COUNTERFACTUAL: _counterfactual_prompt(
            matched, safety=safety
        ),
    }


def summarize_action_divergence(
    actions: Mapping[PromptCondition, np.ndarray],
) -> dict[str, dict[str, Any]]:
    if PromptCondition.MATCHED not in actions:
        raise PromptControlError("action comparison requires a matched baseline")
    matched = np.asarray(actions[PromptCondition.MATCHED], dtype=np.float64)
    if matched.ndim != 2 or matched.shape[1] != ACTION_DIM:
        raise PromptControlError(
            f"matched action must have shape [H, {ACTION_DIM}]"
        )
    result: dict[str, dict[str, Any]] = {}
    for condition, value in actions.items():
        action = np.asarray(value, dtype=np.float64)
        if action.shape != matched.shape:
            raise PromptControlError("all prompt-control actions must share one shape")
        finite = bool(np.isfinite(action).all())
        if not finite:
            raise PromptControlError("prompt-control action contains NaN or infinity")
        delta = action - matched
        result[condition.value] = {
            "l2_vs_matched": float(np.linalg.norm(delta)),
            "mean_abs_vs_matched": float(np.mean(np.abs(delta))),
            "max_abs_vs_matched": float(np.max(np.abs(delta))),
            "per_axis_max_abs_vs_matched": [
                float(item) for item in np.max(np.abs(delta), axis=0)
            ],
            "action_shape": list(action.shape),
            "all_actions_finite": finite,
        }
    return result


def _validate_episode_limits(
    episode: EpisodeData,
    *,
    safety: SafetyConfig,
    name: str,
) -> None:
    lower = np.asarray(safety.joint_lower, dtype=np.float64)
    upper = np.asarray(safety.joint_upper, dtype=np.float64)
    for field, values in (
        ("joint_state", episode.joint_state),
        ("action", episode.action),
    ):
        outside = np.argwhere((values < lower) | (values > upper))
        if outside.size:
            frame, axis = (int(item) for item in outside[0])
            raise PromptControlError(
                f"{name} {field} is outside safety limits at frame={frame}, axis={axis}"
            )


def _frames_fingerprint(frames: Sequence[SensorimotorFrame]) -> str:
    digest = sha256()
    for frame in frames:
        digest.update(np.float64(frame.timestamp_s).tobytes())
        digest.update(frame.joint_position.tobytes())
        if frame.executed_action is None:
            raise PromptControlError("live frame is missing executed_action")
        digest.update(frame.executed_action.tobytes())
        for image in frame.primary_images:
            digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
            digest.update(image.tobytes())
    return digest.hexdigest()


def _action_sha256(action: np.ndarray) -> str:
    value = np.ascontiguousarray(action, dtype=np.float32)
    digest = sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _source_episode_record(
    *,
    bundle_root: Path,
    path: Path,
    episode: EpisodeData,
) -> dict[str, Any]:
    return {
        "manifest_path": path.relative_to(bundle_root).as_posix(),
        "file_sha256": file_sha256(path),
        "episode_fingerprint": episode.fingerprint,
        "content_sha256": _episode_content_sha256(episode),
        "task": episode.task,
        "task_index": episode.task_index,
        "episode_index": episode.episode_index,
    }


def _predict_action(
    policy: CompactWAMPolicy,
    config: ProjectConfig,
    prompt: PhysicalPrompt,
    live_frames: tuple[SensorimotorFrame, ...],
) -> np.ndarray:
    context = Gen15Context(
        prompt,
        policy_hz=config.runtime.policy_hz,
        total_duration_s=config.runtime.context_seconds,
    )
    context.extend_live(live_frames)
    now_s = live_frames[-1].timestamp_s + 1.0 / config.runtime.policy_hz
    return np.asarray(
        policy.predict(context.snapshot(), now_s=now_s).target_joint_position,
        dtype=np.float32,
    )


def _write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    if target.exists():
        raise PromptControlError(f"prompt-control evidence already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, target)
        except FileExistsError as error:
            raise PromptControlError(
                f"prompt-control evidence already exists: {target}"
            ) from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return target


def _artifact_paths(artifact_dir: Path) -> dict[PromptCondition, Path]:
    return {
        condition: artifact_dir / f"{condition.value}.json"
        for condition in PromptCondition
    }


def _preflight_output_paths(report_path: Path, artifacts: Mapping[Any, Path]) -> None:
    output_paths = (report_path, *artifacts.values())
    resolved = tuple(path.resolve() for path in output_paths)
    if len(set(resolved)) != len(resolved):
        raise PromptControlError("prompt-control output paths must be distinct")
    existing = next(
        (path for path in output_paths if path.exists()),
        None,
    )
    if existing is not None:
        raise PromptControlError(f"prompt-control evidence already exists: {existing}")


def run_prompt_control_suite(
    manifest_path: str | Path,
    *,
    artifact_dir: str | Path,
    report_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    manifest_target = Path(manifest_path)
    manifest = load_prompt_control_manifest(manifest_target)
    bundle_root = manifest_target.parent.resolve()
    report_target = Path(report_path)
    artifact_targets = _artifact_paths(Path(artifact_dir))
    _preflight_output_paths(report_target, artifact_targets)

    config = ProjectConfig.load(manifest.config_path)
    bundle = load_compact_wam_bundle(manifest.checkpoint_path, device=device)
    if bundle.model.action_horizon != config.runtime.action_horizon:
        raise PromptControlError(
            "checkpoint action_horizon does not match runtime.action_horizon"
        )
    checkpoint_id = bundle.metadata.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise PromptControlError("checkpoint metadata requires checkpoint_id")

    live = load_episode(manifest.live_episode_path)
    matched = load_episode(manifest.matched_prompt_path)
    alternate = load_episode(manifest.same_task_alternate_prompt_path)
    wrong = load_episode(manifest.wrong_task_prompt_path)
    validate_prompt_control_sources(live, matched, alternate, wrong)
    for name, episode in (
        ("live", live),
        ("matched", matched),
        ("same_task_alternate", alternate),
        ("wrong_task", wrong),
    ):
        _validate_episode_limits(episode, safety=config.safety, name=name)

    matched_prompt = physical_prompt_from_episode(
        matched, policy_hz=config.runtime.policy_hz
    )
    alternate_prompt = physical_prompt_from_episode(
        alternate, policy_hz=config.runtime.policy_hz
    )
    wrong_prompt = physical_prompt_from_episode(
        wrong, policy_hz=config.runtime.policy_hz
    )
    conditions = build_prompt_conditions(
        matched_prompt,
        alternate_prompt,
        wrong_prompt,
        safety=config.safety,
        seed=manifest.seed,
    )
    live_prompt = physical_prompt_from_episode(
        live, policy_hz=config.runtime.policy_hz
    )
    history_steps = bundle.model.action_history_steps
    live_frames = live_prompt.frames[-history_steps:]
    if len(live_frames) != history_steps:
        raise PromptControlError("live episode cannot provide the required history")

    bundle.model.eval()
    policy = CompactWAMPolicy(
        bundle.model,
        servo_hz=config.runtime.servo_hz,
        device=device,
    )
    actions = {
        condition: _predict_action(policy, config, prompt, live_frames)
        for condition, prompt in conditions.items()
    }
    matched_repeat = _predict_action(
        policy,
        config,
        conditions[PromptCondition.MATCHED],
        live_frames,
    )
    matched_repeat_max_abs = float(
        np.max(np.abs(matched_repeat - actions[PromptCondition.MATCHED]))
    )
    metrics = summarize_action_divergence(actions)
    live_fingerprint = _frames_fingerprint(live_frames)
    source_fingerprints = {
        PromptCondition.MATCHED: matched.fingerprint,
        PromptCondition.SAME_TASK_ALTERNATE: alternate.fingerprint,
        PromptCondition.WRONG_TASK: wrong.fingerprint,
        PromptCondition.TEMPORAL_SHUFFLE: matched.fingerprint,
        PromptCondition.IMAGE_FRAME_SHUFFLE: matched.fingerprint,
        PromptCondition.NULL: matched.fingerprint,
        PromptCondition.COUNTERFACTUAL: matched.fingerprint,
    }

    condition_reports: list[dict[str, Any]] = []
    for condition in PromptCondition:
        action = actions[condition]
        artifact_payload = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": manifest.suite_id,
            "condition": condition.value,
            "prompt_fingerprint": conditions[condition].fingerprint,
            "source_episode_fingerprint": source_fingerprints[condition],
            "live_context_fingerprint": live_fingerprint,
            "action_sha256": _action_sha256(action),
            "action": action.tolist(),
            "metrics": metrics[condition.value],
        }
        artifact_path = _write_json_no_overwrite(
            artifact_targets[condition], artifact_payload
        )
        condition_reports.append(
            {
                "condition": condition.value,
                "prompt_fingerprint": conditions[condition].fingerprint,
                "source_episode_fingerprint": source_fingerprints[condition],
                "action_sha256": artifact_payload["action_sha256"],
                "metrics": metrics[condition.value],
                "artifact": artifact_path.name,
                "artifact_sha256": file_sha256(artifact_path),
            }
        )

    nonzero_divergence_count = sum(
        int(metrics[condition.value]["max_abs_vs_matched"] > 0.0)
        for condition in PromptCondition
        if condition is not PromptCondition.MATCHED
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": manifest.suite_id,
        "result": "complete",
        "evidence_level": "offline",
        "scope": PROMPT_CONTROL_SCOPE,
        "result_semantics": "prompt perturbation diagnostic only",
        "robot_used": False,
        "mujoco_terminal_success_evaluated": False,
        "real_world_success_claimed": False,
        "causality_claimed": False,
        "manifest_sha256": file_sha256(manifest_target),
        "source_path_policy": "manifest_bundle_relative",
        "source_episodes": {
            "live": _source_episode_record(
                bundle_root=bundle_root,
                path=manifest.live_episode_path,
                episode=live,
            ),
            "matched": _source_episode_record(
                bundle_root=bundle_root,
                path=manifest.matched_prompt_path,
                episode=matched,
            ),
            "same_task_alternate": _source_episode_record(
                bundle_root=bundle_root,
                path=manifest.same_task_alternate_prompt_path,
                episode=alternate,
            ),
            "wrong_task": _source_episode_record(
                bundle_root=bundle_root,
                path=manifest.wrong_task_prompt_path,
                episode=wrong,
            ),
        },
        "config_sha256": project_config_sha256(config),
        "checkpoint_id": checkpoint_id.strip(),
        "checkpoint_sha256": file_sha256(manifest.checkpoint_path),
        "seed": manifest.seed,
        "live_episode": {
            "task": live.task,
            "task_index": live.task_index,
            "episode_fingerprint": live.fingerprint,
            "file_sha256": file_sha256(manifest.live_episode_path),
            "live_context_fingerprint": live_fingerprint,
            "history_steps": history_steps,
        },
        "summary": {
            "condition_count": len(PromptCondition),
            "nonzero_divergence_count": nonzero_divergence_count,
            "matched_repeat_max_abs": matched_repeat_max_abs,
            "scientific_pass_fail_evaluated": False,
        },
        "conditions": condition_reports,
        "limitations": [
            "action divergence is not terminal task success",
            "prompt perturbation is not proof of causal task understanding",
            "counterfactual sensorimotor values are synthetic",
            "offline evidence does not authorize real robot output",
        ],
    }
    _write_json_no_overwrite(report_target, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure offline CompactWAM action divergence under prompt controls."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        report = run_prompt_control_suite(
            args.manifest,
            artifact_dir=args.artifact_dir,
            report_path=args.report,
            device=args.device,
        )
    except (
        CheckpointError,
        ConfigError,
        ContractError,
        DatasetError,
        ModelContractError,
        OSError,
        PolicyError,
        PromptControlError,
        TensorizerError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "PROMPT_CONTROL_SCOPE",
    "PromptCondition",
    "PromptControlError",
    "PromptControlManifest",
    "build_prompt_conditions",
    "load_prompt_control_manifest",
    "load_prompt_control_manifest_bytes",
    "main",
    "run_prompt_control_suite",
    "summarize_action_divergence",
    "validate_prompt_control_sources",
]


if __name__ == "__main__":
    raise SystemExit(main())
