"""Closed-loop MuJoCo prompt-condition diagnostics for one checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import CheckpointError
from .config import ConfigError, DEFAULT_MUJOCO_CONFIG_PATH, ProjectConfig
from .contracts import ContractError, PhysicalPrompt
from .dataset import (
    DatasetError,
    EpisodeBuffer,
    EpisodeData,
    load_episode,
    save_episode,
)
from .deployment import file_sha256, project_config_sha256
from .model import ModelContractError
from .mujoco_benchmark import (
    CHECKPOINT_BENCHMARK_GATE,
    CHECKPOINT_BENCHMARK_POLICY,
    CHECKPOINT_BENCHMARK_SCOPE,
    CHECKPOINT_TASK_SPLIT,
    PROMPT_TASK_MATCH,
    MujocoBenchmarkError,
    load_benchmark_manifest,
    run_checkpoint_benchmark,
    write_benchmark_report,
)
from .mujoco_cli import MujocoCLIError
from .policy import PolicyError
from .prompt_directionality import (
    DirectionPlan,
    PromptDirectionError,
    evaluate_directionality,
    load_direction_plan_bytes,
)
from .prompt_controls import (
    PromptCondition,
    PromptControlError,
    PromptControlManifest,
    build_prompt_conditions,
    load_prompt_control_manifest,
    validate_prompt_control_sources,
)
from .rollout import RolloutError
from .tensorizer import TensorizerError


SCHEMA_VERSION = 1
MUJOCO_PROMPT_CONTROL_GATE = "G8-candidate-prompt-control-joint-proxy"
MUJOCO_PROMPT_CONTROL_SCOPE = "diagnostic_prompt_control_mujoco_joint_proxy"
TASK_DISJOINT_BASIS = "manifest_ids_only"
CONDITION_GENERATOR = "prompt_controls_v1"
_SOURCE_CONFIG_FIELDS = (
    "camera_hz",
    "policy_hz",
    "servo_hz",
    "context_seconds",
    "action_horizon",
    "primary_cameras",
)
_CONDITION_SOURCES = {
    PromptCondition.MATCHED: "matched",
    PromptCondition.SAME_TASK_ALTERNATE: "same_task_alternate",
    PromptCondition.WRONG_TASK: "wrong_task",
    PromptCondition.TEMPORAL_SHUFFLE: "matched",
    PromptCondition.IMAGE_FRAME_SHUFFLE: "matched",
    PromptCondition.NULL: "matched",
    PromptCondition.COUNTERFACTUAL: "matched",
}


def _load_sources(
    manifest: PromptControlManifest,
) -> dict[str, tuple[Path, EpisodeData]]:
    sources = {
        "live": (
            manifest.live_episode_path,
            load_episode(manifest.live_episode_path),
        ),
        "matched": (
            manifest.matched_prompt_path,
            load_episode(manifest.matched_prompt_path),
        ),
        "same_task_alternate": (
            manifest.same_task_alternate_prompt_path,
            load_episode(manifest.same_task_alternate_prompt_path),
        ),
        "wrong_task": (
            manifest.wrong_task_prompt_path,
            load_episode(manifest.wrong_task_prompt_path),
        ),
    }
    validate_prompt_control_sources(
        sources["live"][1],
        sources["matched"][1],
        sources["same_task_alternate"][1],
        sources["wrong_task"][1],
    )
    return sources


def _validate_source_config(
    source: ProjectConfig,
    mujoco: ProjectConfig,
) -> None:
    mismatches = [
        name
        for name in _SOURCE_CONFIG_FIELDS
        if getattr(source.runtime, name) != getattr(mujoco.runtime, name)
    ]
    if mismatches:
        raise PromptControlError(
            f"source and MuJoCo runtime fields mismatch: {mismatches}"
        )


def _validate_source_limits(
    sources: Mapping[str, tuple[Path, EpisodeData]],
    config: ProjectConfig,
) -> None:
    lower = np.asarray(config.safety.joint_lower, dtype=np.float64)
    upper = np.asarray(config.safety.joint_upper, dtype=np.float64)
    for name, (_, episode) in sources.items():
        for field, values in (
            ("joint_state", episode.joint_state),
            ("action", episode.action),
        ):
            outside = np.argwhere((values < lower) | (values > upper))
            if outside.size == 0:
                continue
            frame, axis = (int(item) for item in outside[0])
            raise PromptControlError(
                f"{name} {field} is outside MuJoCo safety limits at "
                f"frame={frame}, axis={axis}"
            )


def _raw_prompt(episode: EpisodeData, *, name: str) -> PhysicalPrompt:
    try:
        return PhysicalPrompt(tuple(episode.frames()))
    except ValueError as error:
        raise PromptControlError(
            f"{name} episode is not a valid prompt: {error}"
        ) from error


def _condition_prompts(
    sources: Mapping[str, tuple[Path, EpisodeData]],
    *,
    config: ProjectConfig,
    seed: int,
) -> dict[PromptCondition, PhysicalPrompt]:
    return build_prompt_conditions(
        _raw_prompt(sources["matched"][1], name="matched"),
        _raw_prompt(
            sources["same_task_alternate"][1],
            name="same_task_alternate",
        ),
        _raw_prompt(sources["wrong_task"][1], name="wrong_task"),
        safety=config.safety,
        seed=seed,
    )


def _episode_from_prompt(
    prompt: PhysicalPrompt,
    source: EpisodeData,
) -> EpisodeData:
    buffer = EpisodeBuffer(
        fps=source.fps,
        task=source.task,
        episode_index=source.episode_index,
        task_index=source.task_index,
        metadata=source.metadata,
    )
    buffer.extend(iter(prompt.frames))
    return buffer.to_episode_data()


def _source_record(
    *,
    bundle_root: Path,
    path: Path,
    episode: EpisodeData,
) -> dict[str, Any]:
    manifest_path = path.with_suffix(".json")
    return {
        "npz_path": path.relative_to(bundle_root).as_posix(),
        "npz_sha256": file_sha256(path),
        "manifest_sha256": file_sha256(manifest_path),
        "episode_fingerprint": episode.fingerprint,
        "task": episode.task,
        "task_index": episode.task_index,
        "episode_index": episode.episode_index,
    }


def _preflight_outputs(
    artifact_dir: Path,
    report_path: Path,
) -> None:
    if artifact_dir.exists():
        raise PromptControlError(
            f"MuJoCo prompt-control artifact directory already exists: {artifact_dir}"
        )
    if report_path.exists():
        raise PromptControlError(
            f"MuJoCo prompt-control report already exists: {report_path}"
        )


def _load_directionality_plan(
    path: str | Path,
    *,
    suite_id: str,
    prompt_control_manifest_sha256: str,
    benchmark_manifest_sha256: str,
    checkpoint_sha256: str,
    mujoco_config_sha256: str,
) -> tuple[DirectionPlan, bytes, str]:
    target = Path(path)
    source_bytes = target.read_bytes()
    try:
        plan = load_direction_plan_bytes(
            source_bytes,
            expected_suite_id=suite_id,
            expected_prompt_control_manifest_sha256=prompt_control_manifest_sha256,
            expected_benchmark_manifest_sha256=benchmark_manifest_sha256,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_mujoco_config_sha256=mujoco_config_sha256,
        )
    except PromptDirectionError as error:
        raise PromptControlError(str(error)) from error
    return plan, source_bytes, sha256(source_bytes).hexdigest()


def _copy_direction_preregistration(source: bytes, artifact_root: Path) -> Path:
    target = artifact_root / "directionality_preregistration.json"
    if target.exists():
        raise PromptControlError(
            f"directionality preregistration artifact already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    return target


def _evaluate_directionality(
    plan: DirectionPlan,
    condition_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    try:
        return evaluate_directionality(plan, condition_reports)
    except PromptDirectionError as error:
        raise PromptControlError(str(error)) from error


def _expect_report_field(
    report: Mapping[str, Any],
    *,
    field: str,
    expected: object,
    condition: PromptCondition,
) -> None:
    if report.get(field) != expected:
        raise PromptControlError(
            f"{condition.value} benchmark {field} mismatch: "
            f"expected={expected!r}, actual={report.get(field)!r}"
        )


def _validate_trial_artifacts(
    report: Mapping[str, Any],
    *,
    condition: PromptCondition,
    artifact_dir: Path,
    expected_trials: tuple[tuple[str, int], ...],
) -> tuple[list[Mapping[str, Any]], list[float]]:
    trials = report.get("trials")
    if not isinstance(trials, list):
        raise PromptControlError(f"{condition.value} benchmark trials must be a list")
    actual_trials: list[tuple[str, int]] = []
    final_errors: list[float] = []
    validated: list[Mapping[str, Any]] = []
    artifact_root = artifact_dir.resolve()
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise PromptControlError(
                f"{condition.value} benchmark trial must be an object"
            )
        task_id = trial.get("task_id")
        seed = trial.get("seed")
        if not isinstance(task_id, str) or not isinstance(seed, int):
            raise PromptControlError(
                f"{condition.value} benchmark trial identity is invalid"
            )
        actual_trials.append((task_id, seed))
        success = trial.get("success")
        final_error = trial.get("final_error")
        if not isinstance(success, bool):
            raise PromptControlError(
                f"{condition.value} benchmark trial success must be boolean"
            )
        if (
            not isinstance(final_error, (int, float))
            or isinstance(final_error, bool)
            or not isfinite(float(final_error))
            or float(final_error) < 0.0
        ):
            raise PromptControlError(
                f"{condition.value} benchmark trial final_error is invalid"
            )
        final_errors.append(float(final_error))
        artifact = trial.get("artifact")
        artifact_sha256 = trial.get("artifact_sha256")
        if not isinstance(artifact, str) or not artifact:
            raise PromptControlError(
                f"{condition.value} benchmark trial artifact is invalid"
            )
        artifact_path = (artifact_dir / artifact).resolve()
        if not artifact_path.is_relative_to(artifact_root):
            raise PromptControlError(
                f"{condition.value} benchmark trial artifact escapes its directory"
            )
        if not artifact_path.is_file() or file_sha256(artifact_path) != artifact_sha256:
            raise PromptControlError(
                f"{condition.value} benchmark trial artifact hash mismatch"
            )
        validated.append(trial)
    if tuple(actual_trials) != expected_trials:
        raise PromptControlError(
            f"{condition.value} benchmark task/seed schedule mismatch"
        )
    return validated, final_errors


def _validate_benchmark_report(
    report: Mapping[str, Any],
    *,
    condition: PromptCondition,
    config: ProjectConfig,
    benchmark_manifest_path: Path,
    checkpoint_path: Path,
    prompt_path: Path,
    prompt_manifest_path: Path,
    prompt: EpisodeData,
    artifact_dir: Path,
    expected_trials: tuple[tuple[str, int], ...],
    expected_checkpoint_id: str | None,
) -> tuple[dict[str, Any], str]:
    expected_fields = {
        "gate": CHECKPOINT_BENCHMARK_GATE,
        "policy": CHECKPOINT_BENCHMARK_POLICY,
        "benchmark_scope": CHECKPOINT_BENCHMARK_SCOPE,
        "config_sha256": project_config_sha256(config),
        "manifest_sha256": file_sha256(benchmark_manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "prompt_npz_sha256": file_sha256(prompt_path),
        "prompt_manifest_sha256": file_sha256(prompt_manifest_path),
        "prompt_fingerprint": prompt.fingerprint,
        "task_disjoint_basis": TASK_DISJOINT_BASIS,
        "checkpoint_task_split": CHECKPOINT_TASK_SPLIT,
        "prompt_task_match": PROMPT_TASK_MATCH,
    }
    for field, expected_value in expected_fields.items():
        _expect_report_field(
            report,
            field=field,
            expected=expected_value,
            condition=condition,
        )
    checkpoint_id = report.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise PromptControlError(
            f"{condition.value} benchmark checkpoint_id is invalid"
        )
    checkpoint_id = checkpoint_id.strip()
    if expected_checkpoint_id is not None and checkpoint_id != expected_checkpoint_id:
        raise PromptControlError(f"{condition.value} benchmark checkpoint_id mismatch")
    trials, final_errors = _validate_trial_artifacts(
        report,
        condition=condition,
        artifact_dir=artifact_dir,
        expected_trials=expected_trials,
    )
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise PromptControlError(
            f"{condition.value} benchmark summary must be an object"
        )
    success_count = sum(int(bool(trial["success"])) for trial in trials)
    trial_count = len(trials)
    expected_rate = success_count / trial_count
    for field, summary_value in (
        ("trial_count", trial_count),
        ("success_count", success_count),
        ("success_rate", expected_rate),
    ):
        if summary.get(field) != summary_value:
            raise PromptControlError(
                f"{condition.value} benchmark summary {field} mismatch"
            )
    result = report.get("result")
    if result not in {"pass", "fail"}:
        raise PromptControlError(
            f"{condition.value} benchmark result must be pass or fail"
        )
    failure_counts = summary.get("failure_counts")
    interval = summary.get("success_rate_95ci")
    if not isinstance(failure_counts, Mapping) or not isinstance(interval, list):
        raise PromptControlError(
            f"{condition.value} benchmark summary evidence is invalid"
        )
    return (
        {
            "benchmark_result": result,
            "trial_count": trial_count,
            "success_count": success_count,
            "success_rate": expected_rate,
            "success_rate_95ci": list(interval),
            "final_error_mean": fmean(final_errors),
            "final_error_max": max(final_errors),
            "failure_counts": dict(failure_counts),
        },
        checkpoint_id,
    )


def run_mujoco_prompt_control_suite(
    config: ProjectConfig,
    *,
    prompt_control_manifest_path: str | Path,
    benchmark_manifest_path: str | Path,
    direction_preregistration_path: str | Path | None = None,
    artifact_dir: str | Path,
    report_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run every prompt condition through the same checkpoint joint proxy."""

    prompt_control_target = Path(prompt_control_manifest_path)
    benchmark_target = Path(benchmark_manifest_path)
    artifact_root = Path(artifact_dir)
    report_target = Path(report_path)
    _preflight_outputs(artifact_root, report_target)

    prompt_manifest = load_prompt_control_manifest(prompt_control_target)
    prompt_control_manifest_sha256 = file_sha256(prompt_control_target)
    benchmark_manifest_sha256 = file_sha256(benchmark_target)
    checkpoint_sha256 = file_sha256(prompt_manifest.checkpoint_path)
    mujoco_config_sha256 = project_config_sha256(config)
    direction_plan = None
    direction_preregistration_bytes = None
    direction_preregistration_sha256 = None
    direction_preregistration_target = (
        None
        if direction_preregistration_path is None
        else Path(direction_preregistration_path)
    )
    if direction_preregistration_target is not None:
        (
            direction_plan,
            direction_preregistration_bytes,
            direction_preregistration_sha256,
        ) = _load_directionality_plan(
            direction_preregistration_target,
            suite_id=prompt_manifest.suite_id,
            prompt_control_manifest_sha256=prompt_control_manifest_sha256,
            benchmark_manifest_sha256=benchmark_manifest_sha256,
            checkpoint_sha256=checkpoint_sha256,
            mujoco_config_sha256=mujoco_config_sha256,
        )
    source_config = ProjectConfig.load(prompt_manifest.config_path)
    _validate_source_config(source_config, config)
    sources = _load_sources(prompt_manifest)
    _validate_source_limits(sources, config)
    benchmark_manifest = load_benchmark_manifest(benchmark_target)
    expected_trials = tuple(
        (task.task_id, seed)
        for task in benchmark_manifest.heldout_tasks
        for seed in task.seeds
    )
    prompts = _condition_prompts(
        sources,
        config=config,
        seed=prompt_manifest.seed,
    )
    prompt_root = artifact_root / "prompts"
    benchmark_root = artifact_root / "benchmark_artifacts"
    benchmark_report_root = artifact_root / "benchmark_reports"
    condition_reports: list[dict[str, Any]] = []
    checkpoint_id: str | None = None

    for condition in PromptCondition:
        source_name = _CONDITION_SOURCES[condition]
        source_episode = sources[source_name][1]
        condition_episode = _episode_from_prompt(
            prompts[condition],
            source_episode,
        )
        prompt_path, prompt_manifest_path = save_episode(
            condition_episode,
            prompt_root,
            stem=condition.value,
        )
        trial_artifact_dir = benchmark_root / condition.value
        benchmark_report = run_checkpoint_benchmark(
            config,
            manifest_path=benchmark_target,
            artifact_dir=trial_artifact_dir,
            checkpoint_path=prompt_manifest.checkpoint_path,
            prompt_path=prompt_path,
            prompt_manifest_path=prompt_manifest_path,
            device=device,
        )
        metrics, observed_checkpoint_id = _validate_benchmark_report(
            benchmark_report,
            condition=condition,
            config=config,
            benchmark_manifest_path=benchmark_target,
            checkpoint_path=prompt_manifest.checkpoint_path,
            prompt_path=prompt_path,
            prompt_manifest_path=prompt_manifest_path,
            prompt=condition_episode,
            artifact_dir=trial_artifact_dir,
            expected_trials=expected_trials,
            expected_checkpoint_id=checkpoint_id,
        )
        if checkpoint_id is None:
            checkpoint_id = observed_checkpoint_id
        benchmark_report_path = benchmark_report_root / f"{condition.value}.json"
        write_benchmark_report(benchmark_report_path, benchmark_report)
        condition_reports.append(
            {
                "condition": condition.value,
                "source_episode": source_name,
                "condition_prompt_fingerprint": prompts[condition].fingerprint,
                "prompt_episode_fingerprint": condition_episode.fingerprint,
                "prompt_npz_sha256": file_sha256(prompt_path),
                "prompt_manifest_sha256": file_sha256(prompt_manifest_path),
                "prompt_npz_artifact": prompt_path.relative_to(
                    artifact_root
                ).as_posix(),
                "prompt_manifest_artifact": prompt_manifest_path.relative_to(
                    artifact_root
                ).as_posix(),
                "benchmark_report": benchmark_report_path.relative_to(
                    artifact_root
                ).as_posix(),
                "benchmark_report_sha256": file_sha256(benchmark_report_path),
                "checkpoint_sha256": file_sha256(prompt_manifest.checkpoint_path),
                "checkpoint_id": observed_checkpoint_id,
                **metrics,
            }
        )

    matched = condition_reports[0]
    matched_success_rate = float(matched["success_rate"])
    matched_final_error_mean = float(matched["final_error_mean"])
    for condition_report in condition_reports:
        condition_report["success_rate_delta_vs_matched"] = (
            float(condition_report["success_rate"]) - matched_success_rate
        )
        condition_report["final_error_mean_delta_vs_matched"] = (
            float(condition_report["final_error_mean"]) - matched_final_error_mean
        )

    directionality_evaluation = None
    direction_preregistration_artifact = None
    if direction_plan is not None and direction_preregistration_bytes is not None:
        directionality_evaluation = _evaluate_directionality(
            direction_plan,
            condition_reports,
        )
        direction_preregistration_artifact = _copy_direction_preregistration(
            direction_preregistration_bytes,
            artifact_root,
        )

    bundle_root = prompt_control_target.parent.resolve()
    directionality_preregistered = directionality_evaluation is not None
    limitations = [
        "joint-target tolerance is not semantic task success",
        "checkpoint training split and prompt-task match are not verified",
    ]
    if directionality_preregistered:
        limitations.extend(
            (
                "directionality is a local input-hash-bound joint-proxy diagnostic",
                "external preregistration timing is not verified",
            )
        )
    else:
        limitations.append(
            "condition deltas are descriptive because directionality is not "
            "preregistered"
        )
    limitations.extend(
        (
            "MuJoCo prompt sensitivity is not proof of prompt causality",
            "simulation evidence does not authorize real robot output",
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": MUJOCO_PROMPT_CONTROL_GATE,
        "result": "complete",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "scope": MUJOCO_PROMPT_CONTROL_SCOPE,
        "result_semantics": "all prompt conditions completed against a joint proxy",
        "suite_id": prompt_manifest.suite_id,
        "robot_used": False,
        "recorded_live_context_used": False,
        "mujoco_live_context_used": True,
        "mujoco_terminal_success_evaluated": True,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "scientific_pass_fail_evaluated": directionality_preregistered,
        "directionality_preregistered": directionality_preregistered,
        "prompt_control_manifest_sha256": prompt_control_manifest_sha256,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "source_config_sha256": project_config_sha256(source_config),
        "mujoco_config_sha256": mujoco_config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "benchmark_id": benchmark_manifest.benchmark_id,
        "seed": prompt_manifest.seed,
        "condition_generator": CONDITION_GENERATOR,
        "condition_safety_basis": "mujoco_config",
        "task_disjoint_basis": TASK_DISJOINT_BASIS,
        "checkpoint_task_split": CHECKPOINT_TASK_SPLIT,
        "prompt_task_match": PROMPT_TASK_MATCH,
        "source_episodes": {
            name: _source_record(
                bundle_root=bundle_root,
                path=path,
                episode=episode,
            )
            for name, (path, episode) in sources.items()
        },
        "summary": {
            "condition_count": len(condition_reports),
            "trials_per_condition": len(expected_trials),
            "total_trial_count": len(condition_reports) * len(expected_trials),
            "matched_success_rate": matched_success_rate,
            "matched_final_error_mean": matched_final_error_mean,
        },
        "conditions": condition_reports,
        "limitations": limitations,
    }
    if (
        directionality_evaluation is not None
        and direction_preregistration_artifact is not None
        and direction_preregistration_target is not None
        and direction_preregistration_sha256 is not None
    ):
        report["preregistration_level"] = "local_input_hash_bound_before_rollout"
        report["external_preregistration_timestamp_verified"] = False
        report["directionality_evaluation"] = directionality_evaluation
        report["directionality_preregistration_sha256"] = (
            direction_preregistration_sha256
        )
        report["directionality_preregistration_artifact"] = (
            direction_preregistration_artifact.relative_to(artifact_root).as_posix()
        )
        report["directionality_preregistration_artifact_sha256"] = file_sha256(
            direction_preregistration_artifact
        )
    write_benchmark_report(report_target, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run CompactWAM prompt controls through the MuJoCo joint proxy."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MUJOCO_CONFIG_PATH)
    parser.add_argument("--prompt-control-manifest", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--direction-preregistration", type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        report = run_mujoco_prompt_control_suite(
            ProjectConfig.load(args.config),
            prompt_control_manifest_path=args.prompt_control_manifest,
            benchmark_manifest_path=args.benchmark_manifest,
            direction_preregistration_path=args.direction_preregistration,
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
        MujocoAdapterError,
        MujocoBenchmarkError,
        MujocoCLIError,
        OSError,
        PolicyError,
        PromptControlError,
        RolloutError,
        TensorizerError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "MUJOCO_PROMPT_CONTROL_GATE",
    "MUJOCO_PROMPT_CONTROL_SCOPE",
    "main",
    "run_mujoco_prompt_control_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
