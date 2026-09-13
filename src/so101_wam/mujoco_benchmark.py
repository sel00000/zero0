"""Closed-loop MuJoCo held-out benchmark reporting."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import CheckpointError, load_compact_wam_bundle
from .config import ConfigError, DEFAULT_MUJOCO_CONFIG_PATH, ProjectConfig
from .constants import ACTION_DIM, JOINT_KEYS, PRIMARY_CAMERA_KEYS
from .context import ContextSnapshot
from .contracts import ActionChunk, SensorimotorFrame
from .dataset import DatasetError, load_episode
from .deployment import file_sha256, project_config_sha256
from .mujoco_cli import (
    MujocoCLIError,
    _VirtualClock,
    _adapter,
    _checkpoint_identity,
    _capture_home_prompt,
    _require_mujoco_backend,
    run_mujoco_checkpoint_session,
    write_mujoco_report,
)
from .policy import PolicyError
from .rollout import RolloutError, run_managed_rollout
from .runtime import RuntimeErrorState, SO101WAMRuntime


MIN_SEED_COUNT = 3
SCHEMA_VERSION = 1
BENCHMARK_GATE = "G8-heldout-benchmark"
BENCHMARK_POLICY = "joint_target_reference"
BENCHMARK_SCOPE = "diagnostic_joint_reach_not_learned_policy"
CHECKPOINT_BENCHMARK_GATE = "G8-candidate-joint-proxy"
CHECKPOINT_BENCHMARK_POLICY = "compact_wam"
CHECKPOINT_BENCHMARK_SCOPE = "candidate_joint_proxy_not_heldout_verified"
CHECKPOINT_TASK_SPLIT = "not_verified"
PROMPT_TASK_MATCH = "not_verified"
INITIAL_BODY_JITTER = 0.05
INITIAL_GRIPPER_MARGIN = 1.0
WILSON_95_Z = 1.959963984540054
_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_GRIPPER_INDICES = tuple(
    index for index, key in enumerate(JOINT_KEYS) if key.endswith("gripper.pos")
)
_BODY_INDICES = tuple(
    index for index in range(ACTION_DIM) if index not in _GRIPPER_INDICES
)


class MujocoBenchmarkError(ValueError):
    """Raised when a held-out benchmark contract is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    label: str
    policy_steps: int
    seeds: tuple[int, ...]
    target_joint_position: tuple[float, ...]
    tolerance: float


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    benchmark_id: str
    train_task_ids: tuple[str, ...]
    heldout_tasks: tuple[BenchmarkTask, ...]


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    success: bool
    final_error: float
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkTrialResult:
    success: bool
    final_error: float
    failure_reason: str | None
    rollout: Mapping[str, Any]
    initial_joint_position: tuple[float, ...]
    final_joint_position: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _CheckpointBenchmarkEvidence:
    checkpoint_sha256: str
    checkpoint_id: str
    prompt_npz_sha256: str
    prompt_manifest_sha256: str
    prompt_fingerprint: str


class TrialRunner(Protocol):
    def __call__(
        self,
        config: ProjectConfig,
        task: BenchmarkTask,
        *,
        seed: int,
    ) -> BenchmarkTrialResult: ...


class _JointTargetPolicy:
    required_history_steps = 1

    def __init__(
        self,
        *,
        target_joint_position: tuple[float, ...],
        horizon: int,
        servo_hz: float,
    ) -> None:
        self.target = np.asarray(target_joint_position, dtype=np.float32)
        self.horizon = horizon
        self.dt_s = 1.0 / servo_hz

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        del snapshot
        targets = np.repeat(self.target[None, :], self.horizon, axis=0)
        return ActionChunk(
            target_joint_position=targets,
            dt_s=self.dt_s,
            created_at_s=now_s,
        )


def _read_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise MujocoBenchmarkError("benchmark manifest must be a JSON object")
    return payload


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MujocoBenchmarkError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise MujocoBenchmarkError(f"{name} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _seed_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise MujocoBenchmarkError("held-out task seeds must be a list")
    if len(value) < MIN_SEED_COUNT:
        raise MujocoBenchmarkError("held-out task requires at least 3 seeds")
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    ):
        raise MujocoBenchmarkError("held-out task seeds must be non-negative integers")
    seeds = tuple(int(item) for item in value)
    if len(set(seeds)) != len(seeds):
        raise MujocoBenchmarkError("held-out task seeds must be unique")
    return seeds


def _joint_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != ACTION_DIM:
        raise MujocoBenchmarkError(
            f"target_joint_position must contain {ACTION_DIM} values"
        )
    try:
        joints = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise MujocoBenchmarkError("target_joint_position must be numeric") from error
    if not all(isfinite(item) for item in joints):
        raise MujocoBenchmarkError("target_joint_position must be finite")
    return joints


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MujocoBenchmarkError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MujocoBenchmarkError(f"{name} must be positive")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise MujocoBenchmarkError(f"{name} must be positive")
    return result


def _task(value: object) -> BenchmarkTask:
    if not isinstance(value, dict):
        raise MujocoBenchmarkError("heldout_tasks entries must be objects")
    task_id = value.get("task_id")
    label = value.get("label")
    if not isinstance(task_id, str) or not task_id.strip():
        raise MujocoBenchmarkError("held-out task requires task_id")
    if not isinstance(label, str) or not label.strip():
        raise MujocoBenchmarkError("held-out task requires label")
    task_id = task_id.strip()
    if _TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise MujocoBenchmarkError(
            "held-out task_id must contain only letters, numbers, '_' or '-'"
        )
    return BenchmarkTask(
        task_id=task_id,
        label=label.strip(),
        policy_steps=_positive_int(value.get("policy_steps"), name="policy_steps"),
        seeds=_seed_tuple(value.get("seeds")),
        target_joint_position=_joint_tuple(value.get("target_joint_position")),
        tolerance=_positive_float(value.get("tolerance"), name="tolerance"),
    )


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    payload = _read_json_object(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MujocoBenchmarkError("benchmark manifest requires schema_version=1")
    benchmark_id = payload.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise MujocoBenchmarkError("benchmark manifest requires benchmark_id")

    train_task_ids = _string_tuple(payload.get("train_task_ids"), name="train_task_ids")
    tasks_value = payload.get("heldout_tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise MujocoBenchmarkError("heldout_tasks must be a non-empty list")
    tasks = tuple(_task(item) for item in tasks_value)
    train_set = set(train_task_ids)
    heldout_ids = tuple(task.task_id for task in tasks)
    if len(set(heldout_ids)) != len(heldout_ids):
        raise MujocoBenchmarkError("held-out task ids must be unique")
    if train_set.intersection(heldout_ids):
        raise MujocoBenchmarkError("train and held-out task ids must be disjoint")
    return BenchmarkManifest(
        benchmark_id=benchmark_id.strip(),
        train_task_ids=train_task_ids,
        heldout_tasks=tasks,
    )


def score_terminal_joint_success(
    task: BenchmarkTask,
    final_joint_position: np.ndarray | Sequence[float],
) -> TerminalOutcome:
    final = np.asarray(final_joint_position, dtype=np.float64)
    if final.shape != (ACTION_DIM,):
        raise MujocoBenchmarkError(
            f"final joint position must have shape ({ACTION_DIM},)"
        )
    if not np.isfinite(final).all():
        raise MujocoBenchmarkError("final joint position must be finite")
    target = np.asarray(task.target_joint_position, dtype=np.float64)
    final_error = float(np.max(np.abs(final - target)))
    success = final_error <= task.tolerance
    return TerminalOutcome(
        success=success,
        final_error=final_error,
        failure_reason=None if success else "terminal_tolerance",
    )


def _seeded_home_joint_position(
    config: ProjectConfig,
    *,
    seed: int,
) -> tuple[float, ...]:
    """Apply a small deterministic body-joint perturbation for one trial."""

    rng = np.random.default_rng(seed)
    home = np.asarray(config.mujoco.home_joint_position, dtype=np.float64)
    jitter = np.zeros(ACTION_DIM, dtype=np.float64)
    jitter[list(_BODY_INDICES)] = rng.uniform(
        -INITIAL_BODY_JITTER,
        INITIAL_BODY_JITTER,
        size=len(_BODY_INDICES),
    )
    lower = np.asarray(config.safety.joint_lower, dtype=np.float64)
    upper = np.asarray(config.safety.joint_upper, dtype=np.float64)
    seeded = home + jitter
    for index in _GRIPPER_INDICES:
        margin = min(INITIAL_GRIPPER_MARGIN, (upper[index] - lower[index]) / 2.0)
        seeded[index] = max(seeded[index], lower[index] + margin)
    seeded = np.clip(seeded, lower, upper)
    return tuple(float(item) for item in seeded)


def run_task_trial(
    config: ProjectConfig,
    task: BenchmarkTask,
    *,
    seed: int,
) -> BenchmarkTrialResult:
    _require_mujoco_backend(config)
    initial_joint_position = _seeded_home_joint_position(config, seed=seed)
    trial_config = replace(
        config,
        mujoco=replace(
            config.mujoco,
            home_joint_position=initial_joint_position,
        ),
    )
    adapter = _adapter(trial_config)
    prompt = _capture_home_prompt(adapter)
    policy = _JointTargetPolicy(
        target_joint_position=task.target_joint_position,
        horizon=trial_config.runtime.action_horizon,
        servo_hz=trial_config.runtime.servo_hz,
    )
    runtime = SO101WAMRuntime(
        config=trial_config,
        prompt=prompt,
        robot=adapter,
        policy=policy,
    )
    terminal_frame: SensorimotorFrame | None = None

    def observe_terminal(timestamp_s: float) -> None:
        nonlocal terminal_frame
        terminal_frame = adapter.get_observation(timestamp_s=timestamp_s)

    virtual_clock = _VirtualClock()
    summary = run_managed_rollout(
        runtime,
        adapter,
        policy_steps=task.policy_steps,
        calibrate_on_connect=False,
        clock=virtual_clock,
        sleeper=virtual_clock.sleep,
        terminal_observer=observe_terminal,
    )
    if terminal_frame is None:
        raise MujocoBenchmarkError("terminal observation was not captured")
    outcome = score_terminal_joint_success(task, terminal_frame.joint_position)
    return BenchmarkTrialResult(
        success=outcome.success,
        final_error=outcome.final_error,
        failure_reason=outcome.failure_reason,
        rollout=asdict(summary),
        initial_joint_position=initial_joint_position,
        final_joint_position=tuple(
            float(item) for item in terminal_frame.joint_position
        ),
    )


def run_checkpoint_task_trial(
    config: ProjectConfig,
    task: BenchmarkTask,
    *,
    seed: int,
    checkpoint_path: str | Path,
    prompt_path: str | Path,
    prompt_manifest_path: str | Path | None = None,
    device: str = "cpu",
) -> BenchmarkTrialResult:
    _require_mujoco_backend(config)
    initial_joint_position = _seeded_home_joint_position(config, seed=seed)
    trial_config = replace(
        config,
        mujoco=replace(
            config.mujoco,
            home_joint_position=initial_joint_position,
        ),
    )
    report = run_mujoco_checkpoint_session(
        trial_config,
        checkpoint_path=checkpoint_path,
        prompt_path=prompt_path,
        manifest_path=prompt_manifest_path,
        policy_steps=task.policy_steps,
        device=device,
    )
    terminal = _terminal_joint_position(report)
    rollout = report.get("rollout")
    if not isinstance(rollout, Mapping):
        raise MujocoBenchmarkError("checkpoint session report requires rollout object")

    outcome = score_terminal_joint_success(task, terminal)
    return BenchmarkTrialResult(
        success=outcome.success,
        final_error=outcome.final_error,
        failure_reason=outcome.failure_reason,
        rollout=dict(rollout),
        initial_joint_position=initial_joint_position,
        final_joint_position=terminal,
    )


def _terminal_joint_position(report: Mapping[str, Any]) -> tuple[float, ...]:
    terminal = report.get("terminal_joint_position")
    if (
        not isinstance(terminal, Sequence)
        or isinstance(terminal, (bytes, str))
        or len(terminal) != ACTION_DIM
    ):
        raise MujocoBenchmarkError(
            f"checkpoint session report requires {ACTION_DIM} terminal joints"
        )
    try:
        return tuple(float(value) for value in terminal)
    except (TypeError, ValueError) as error:
        raise MujocoBenchmarkError(
            "checkpoint session terminal joints must be numeric"
        ) from error


def _artifact_name(task_id: str, seed: int) -> str:
    return f"{task_id}-seed-{seed}.json"


def _trial_id(task_id: str, seed: int) -> str:
    return f"{task_id}:seed:{seed}"


def _validate_task_targets(
    config: ProjectConfig,
    manifest: BenchmarkManifest,
) -> None:
    lower = np.asarray(config.safety.joint_lower, dtype=np.float64)
    upper = np.asarray(config.safety.joint_upper, dtype=np.float64)
    for task in manifest.heldout_tasks:
        target = np.asarray(task.target_joint_position, dtype=np.float64)
        outside = np.flatnonzero((target < lower) | (target > upper))
        if outside.size:
            names = tuple(JOINT_KEYS[int(index)] for index in outside)
            raise MujocoBenchmarkError(
                f"task {task.task_id!r} target is outside safety limits at {names}"
            )


def _wilson_interval(success_count: int, trial_count: int) -> tuple[float, float]:
    if trial_count < 1:
        raise MujocoBenchmarkError("Wilson interval requires at least one trial")
    proportion = success_count / trial_count
    z_squared = WILSON_95_Z * WILSON_95_Z
    denominator = 1.0 + z_squared / trial_count
    center = (proportion + z_squared / (2.0 * trial_count)) / denominator
    margin = (
        WILSON_95_Z
        * (
            proportion * (1.0 - proportion) / trial_count
            + z_squared / (4.0 * trial_count * trial_count)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _write_evidence(path: Path, payload: Mapping[str, Any]) -> Path:
    try:
        return write_mujoco_report(path, dict(payload))
    except MujocoCLIError as error:
        raise MujocoBenchmarkError(str(error)) from error


def _failure_counts(trials: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(trial["failure_reason"])
        for trial in trials
        if trial["failure_reason"] is not None
    )
    return dict(sorted(counts.items()))


def _task_summaries(
    manifest: BenchmarkManifest,
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for task in manifest.heldout_tasks:
        task_trials = [trial for trial in trials if trial["task_id"] == task.task_id]
        success_count = sum(int(trial["success"]) for trial in task_trials)
        trial_count = len(task_trials)
        interval = _wilson_interval(success_count, trial_count)
        errors = [float(trial["final_error"]) for trial in task_trials]
        summaries.append(
            {
                "task_id": task.task_id,
                "label": task.label,
                "trial_count": trial_count,
                "success_count": success_count,
                "success_rate": success_count / trial_count,
                "success_rate_95ci": list(interval),
                "final_error_mean": fmean(errors),
                "final_error_max": max(errors),
                "failure_counts": _failure_counts(task_trials),
            }
        )
    return summaries


def _validate_trial_result(result: BenchmarkTrialResult) -> None:
    if not isinstance(result.success, bool):
        raise MujocoBenchmarkError("benchmark trial success must be boolean")
    if (
        not isinstance(result.final_error, (int, float))
        or isinstance(result.final_error, bool)
        or not isfinite(float(result.final_error))
        or result.final_error < 0
    ):
        raise MujocoBenchmarkError(
            "benchmark trial final_error must be finite and non-negative"
        )
    reason = result.failure_reason
    if result.success:
        valid_reason = reason is None
    else:
        valid_reason = isinstance(reason, str) and bool(reason.strip())
    if not valid_reason:
        raise MujocoBenchmarkError(
            "benchmark trial failure_reason must be absent for success "
            "and non-empty for failure"
        )


def _run_benchmark(
    config: ProjectConfig,
    *,
    manifest_path: str | Path,
    artifact_dir: str | Path,
    trial_runner: TrialRunner = run_task_trial,
    checkpoint_evidence: _CheckpointBenchmarkEvidence | None = None,
) -> dict[str, Any]:
    _require_mujoco_backend(config)
    gate = BENCHMARK_GATE
    policy = BENCHMARK_POLICY
    benchmark_scope = BENCHMARK_SCOPE
    evidence_fields: dict[str, Any] = {}
    if checkpoint_evidence is not None:
        gate = CHECKPOINT_BENCHMARK_GATE
        policy = CHECKPOINT_BENCHMARK_POLICY
        benchmark_scope = CHECKPOINT_BENCHMARK_SCOPE
        evidence_fields = {
            **asdict(checkpoint_evidence),
            "task_disjoint_basis": "manifest_ids_only",
            "checkpoint_task_split": CHECKPOINT_TASK_SPLIT,
            "prompt_task_match": PROMPT_TASK_MATCH,
        }
    manifest = load_benchmark_manifest(manifest_path)
    _validate_task_targets(config, manifest)
    artifacts = Path(artifact_dir)
    artifact_paths = {
        (task.task_id, seed): artifacts / _artifact_name(task.task_id, seed)
        for task in manifest.heldout_tasks
        for seed in task.seeds
    }
    existing = next((path for path in artifact_paths.values() if path.exists()), None)
    if existing is not None:
        raise MujocoBenchmarkError(
            f"MuJoCo benchmark artifact already exists: {existing}"
        )
    trials: list[dict[str, Any]] = []

    for task in manifest.heldout_tasks:
        for seed in task.seeds:
            result = trial_runner(config, task, seed=seed)
            _validate_trial_result(result)
            artifact_path = artifact_paths[(task.task_id, seed)]
            trial_id = _trial_id(task.task_id, seed)
            artifact_payload = {
                "schema_version": SCHEMA_VERSION,
                "trial_id": trial_id,
                "task_id": task.task_id,
                "seed": seed,
                "success": result.success,
                "final_error": result.final_error,
                "failure_reason": result.failure_reason,
                "policy": policy,
                "benchmark_scope": benchmark_scope,
                "initial_joint_position": list(result.initial_joint_position),
                "target_joint_position": list(task.target_joint_position),
                "final_joint_position": list(result.final_joint_position),
                "rollout": dict(result.rollout),
                **evidence_fields,
            }
            _write_evidence(artifact_path, artifact_payload)
            trials.append(
                {
                    "trial_id": trial_id,
                    "task_id": task.task_id,
                    "seed": seed,
                    "success": result.success,
                    "final_error": result.final_error,
                    "failure_reason": result.failure_reason,
                    "artifact": artifact_path.name,
                    "artifact_sha256": file_sha256(artifact_path),
                }
            )

    success_count = sum(int(trial["success"]) for trial in trials)
    trial_count = len(trials)
    interval = _wilson_interval(success_count, trial_count)
    all_trials_successful = success_count == trial_count
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "result": "pass" if all_trials_successful else "fail",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "benchmark_id": manifest.benchmark_id,
        "config_sha256": project_config_sha256(config),
        "manifest_sha256": file_sha256(manifest_path),
        "policy": policy,
        "benchmark_scope": benchmark_scope,
        "task_disjoint": True,
        "train_task_ids": list(manifest.train_task_ids),
        "heldout_task_ids": [task.task_id for task in manifest.heldout_tasks],
        "primary_cameras": list(PRIMARY_CAMERA_KEYS),
        **evidence_fields,
        "summary": {
            "task_count": len(manifest.heldout_tasks),
            "trial_count": trial_count,
            "success_count": success_count,
            "success_rate": success_count / trial_count if trial_count else 0.0,
            "success_rate_95ci": list(interval),
            "all_trials_successful": all_trials_successful,
            "failure_counts": _failure_counts(trials),
        },
        "tasks": _task_summaries(manifest, trials),
        "trials": trials,
    }


def run_benchmark(
    config: ProjectConfig,
    *,
    manifest_path: str | Path,
    artifact_dir: str | Path,
    trial_runner: TrialRunner = run_task_trial,
) -> dict[str, Any]:
    return _run_benchmark(
        config,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        trial_runner=trial_runner,
    )


def run_checkpoint_benchmark(
    config: ProjectConfig,
    *,
    manifest_path: str | Path,
    artifact_dir: str | Path,
    checkpoint_path: str | Path,
    prompt_path: str | Path,
    prompt_manifest_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    _require_mujoco_backend(config)
    bundle = load_compact_wam_bundle(checkpoint_path, device=device)
    checkpoint_sha256, checkpoint_id = _checkpoint_identity(
        checkpoint_path,
        bundle.metadata,
    )
    prompt_manifest = (
        Path(prompt_manifest_path)
        if prompt_manifest_path is not None
        else Path(prompt_path).with_suffix(".json")
    )
    prompt = load_episode(prompt_path, prompt_manifest)
    evidence = _CheckpointBenchmarkEvidence(
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
        prompt_npz_sha256=file_sha256(prompt_path),
        prompt_manifest_sha256=file_sha256(prompt_manifest),
        prompt_fingerprint=prompt.fingerprint,
    )

    def checkpoint_runner(
        config: ProjectConfig,
        task: BenchmarkTask,
        *,
        seed: int,
    ) -> BenchmarkTrialResult:
        return run_checkpoint_task_trial(
            config,
            task,
            seed=seed,
            checkpoint_path=checkpoint_path,
            prompt_path=prompt_path,
            prompt_manifest_path=prompt_manifest,
            device=device,
        )

    return _run_benchmark(
        config,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        trial_runner=checkpoint_runner,
        checkpoint_evidence=evidence,
    )


def write_benchmark_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    return _write_evidence(Path(path), report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run closed-loop held-out MuJoCo benchmark trials."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MUJOCO_CONFIG_PATH)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error(f"MuJoCo benchmark report already exists: {args.report}")
    if (args.checkpoint is None) != (args.prompt is None):
        parser.error("--checkpoint and --prompt must be supplied together")
    if args.prompt_manifest is not None and args.prompt is None:
        parser.error("--prompt-manifest requires --prompt")

    try:
        config = ProjectConfig.load(args.config)
        if args.checkpoint is not None and args.prompt is not None:
            report = run_checkpoint_benchmark(
                config,
                manifest_path=args.manifest,
                artifact_dir=args.artifact_dir,
                checkpoint_path=args.checkpoint,
                prompt_path=args.prompt,
                prompt_manifest_path=args.prompt_manifest,
                device=args.device,
            )
        else:
            report = run_benchmark(
                config,
                manifest_path=args.manifest,
                artifact_dir=args.artifact_dir,
            )
        write_benchmark_report(args.report, report)
    except (
        CheckpointError,
        ConfigError,
        DatasetError,
        MujocoAdapterError,
        MujocoBenchmarkError,
        MujocoCLIError,
        PolicyError,
        RolloutError,
        RuntimeErrorState,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["result"] == "pass" else 1


__all__ = [
    "BENCHMARK_POLICY",
    "BENCHMARK_SCOPE",
    "CHECKPOINT_BENCHMARK_GATE",
    "CHECKPOINT_BENCHMARK_POLICY",
    "CHECKPOINT_BENCHMARK_SCOPE",
    "BenchmarkManifest",
    "BenchmarkTask",
    "BenchmarkTrialResult",
    "MujocoBenchmarkError",
    "TerminalOutcome",
    "load_benchmark_manifest",
    "main",
    "run_benchmark",
    "run_checkpoint_benchmark",
    "run_checkpoint_task_trial",
    "run_task_trial",
    "score_terminal_joint_success",
    "write_benchmark_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
