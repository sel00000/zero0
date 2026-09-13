"""MuJoCo object-state diagnostics across fixed prompt conditions."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import CheckpointError
from .config import ConfigError, DEFAULT_MUJOCO_CONFIG_PATH, ProjectConfig
from .contracts import ContractError
from .dataset import DatasetError, EpisodeData, save_episode
from .deployment import (
    canonical_json_sha256,
    file_sha256,
    project_config_sha256,
)
from .model import ModelContractError
from .mujoco_benchmark import MujocoBenchmarkError
from .mujoco_cli import MujocoCLIError
from .mujoco_prompt_controls import (
    CONDITION_GENERATOR,
    _CONDITION_SOURCES,
    _condition_prompts,
    _episode_from_prompt,
    _load_sources,
    _preflight_outputs,
    _source_record,
    _validate_source_config,
    _validate_source_limits,
)
from .mujoco_semantic_benchmark import (
    CHECKPOINT_SEMANTIC_GATE,
    CHECKPOINT_SEMANTIC_POLICY,
    CHECKPOINT_SEMANTIC_SCOPE,
    SCHEMA_VERSION as SEMANTIC_REPORT_SCHEMA,
    TERMINAL_CRITERION,
    ModelIdentitySource,
    PromptTaskExpectation,
    SemanticManifest,
    SemanticTrialStatus,
    load_semantic_manifest_bytes,
    run_checkpoint_benchmark,
    validate_failure_evidence,
    write_benchmark_report,
)
from .mujoco_identity import MujocoIdentityError, MujocoModelIdentity
from .policy import PolicyError
from .prompt_controls import (
    PromptCondition,
    PromptControlError,
    load_prompt_control_manifest_bytes,
)
from .rollout import RolloutError
from .runtime import RuntimeErrorState
from .semantic_prompt_directionality import (
    SemanticDirectionPlan,
    SemanticPromptDirectionError,
    evaluate_semantic_directionality,
    load_semantic_direction_plan_bytes,
)
from .tensorizer import TensorizerError


SCHEMA_VERSION = 4
MUJOCO_SEMANTIC_PROMPT_CONTROL_GATE = (
    "G8-candidate-prompt-control-semantic-object-state"
)
MUJOCO_SEMANTIC_PROMPT_CONTROL_SCOPE = "diagnostic_prompt_control_mujoco_object_state"
TASK_DISJOINT_BASIS = "checkpoint_bound_task_inventory"
CHECKPOINT_TASK_SPLIT = "training_report_verified"
SEMANTIC_TASK_MAPPING = "manifest_declared"
PROMPT_TASK_MATCH = "condition_expectation_verified"


def _read_source(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PromptControlError(f"failed to read {name}: {error}") from error


def _json_object(source: bytes, *, condition: PromptCondition) -> dict[str, Any]:
    try:
        value = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromptControlError(
            f"{condition.value} benchmark trial artifact is invalid"
        ) from error
    if not isinstance(value, dict):
        raise PromptControlError(
            f"{condition.value} benchmark trial artifact is invalid"
        )
    return value


def _load_directionality_plan(
    source: bytes,
    *,
    suite_id: str,
    prompt_control_manifest_sha256: str,
    semantic_manifest_sha256: str,
    training_report_sha256: str,
    checkpoint_sha256: str,
    mujoco_config_sha256: str,
) -> SemanticDirectionPlan:
    try:
        return load_semantic_direction_plan_bytes(
            source,
            expected_suite_id=suite_id,
            expected_prompt_control_manifest_sha256=(
                prompt_control_manifest_sha256
            ),
            expected_semantic_manifest_sha256=semantic_manifest_sha256,
            expected_training_report_sha256=training_report_sha256,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_mujoco_config_sha256=mujoco_config_sha256,
        )
    except SemanticPromptDirectionError as error:
        raise PromptControlError(str(error)) from error


def _evaluate_directionality(
    plan: SemanticDirectionPlan,
    condition_reports: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    try:
        return evaluate_semantic_directionality(plan, condition_reports)
    except SemanticPromptDirectionError as error:
        raise PromptControlError(str(error)) from error


def _copy_directionality_plan(source: bytes, artifact_root: Path) -> Path:
    target = artifact_root / "semantic_directionality_preregistration.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(source)
    except FileExistsError as error:
        raise PromptControlError(
            f"semantic directionality artifact already exists: {target}"
        ) from error
    return target


def _validate_dataset_task(
    sources: Mapping[str, tuple[Path, EpisodeData]],
    manifest: SemanticManifest,
) -> None:
    for name in ("live", "matched", "same_task_alternate"):
        episode = sources[name][1]
        if (
            episode.task != manifest.dataset_task
            or episode.task_index != manifest.dataset_task_index
        ):
            raise PromptControlError(
                f"{name} task identity does not match semantic manifest dataset_task"
            )


def _task_expectation(
    condition: PromptCondition,
) -> tuple[PromptTaskExpectation, bool, str]:
    if condition is PromptCondition.WRONG_TASK:
        return (
            PromptTaskExpectation.EXPECTED_MISMATCH,
            False,
            "control_mismatch_verified",
        )
    return (
        PromptTaskExpectation.MATCH_MANIFEST,
        True,
        "dataset_identity_verified",
    )


def _expect_field(
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


def _validate_digest(
    value: object,
    *,
    field: str,
    condition: PromptCondition,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PromptControlError(
            f"{condition.value} benchmark {field} is not lowercase SHA-256"
        )
    return value


def _model_identity(value: object, *, condition: PromptCondition) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PromptControlError(
            f"{condition.value} benchmark model identity is invalid"
        )
    try:
        return MujocoModelIdentity.from_payload(value).to_payload()
    except MujocoIdentityError as error:
        raise PromptControlError(
            f"{condition.value} benchmark model identity is invalid: {error}"
        ) from error


def _model_identity_sources(
    value: object,
    *,
    condition: PromptCondition,
) -> list[str]:
    if not isinstance(value, list):
        raise PromptControlError(
            f"{condition.value} benchmark model identity sources are invalid"
        )
    sources: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise PromptControlError(
                f"{condition.value} benchmark model identity source is invalid"
            )
        try:
            sources.add(ModelIdentitySource(item).value)
        except ValueError as error:
            raise PromptControlError(
                f"{condition.value} benchmark model identity source is invalid"
            ) from error
    if not sources:
        raise PromptControlError(
            f"{condition.value} benchmark model identity sources are invalid"
        )
    return sorted(sources)


def _model_identity_source(
    value: object,
    *,
    condition: PromptCondition,
) -> ModelIdentitySource:
    if not isinstance(value, str):
        raise PromptControlError(
            f"{condition.value} benchmark model identity source is invalid"
        )
    try:
        return ModelIdentitySource(value)
    except ValueError as error:
        raise PromptControlError(
            f"{condition.value} benchmark model identity source is invalid"
        ) from error


def _validate_trials(
    report: Mapping[str, Any],
    *,
    condition: PromptCondition,
    artifact_dir: Path,
    expected_trials: tuple[tuple[str, int], ...],
    object_body: str,
    object_profile_sha256: str,
    model_identity: Mapping[str, object],
) -> tuple[list[Mapping[str, Any]], list[float], list[str]]:
    trials = report.get("trials")
    if not isinstance(trials, list):
        raise PromptControlError(f"{condition.value} benchmark trials must be a list")

    actual_trials: list[tuple[str, int]] = []
    errors: list[float] = []
    identity_sources: set[str] = set()
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
        if trial.get("object_body") != object_body:
            raise PromptControlError(
                f"{condition.value} benchmark trial object_body mismatch"
            )
        if trial.get("object_physical_profile_sha256") != object_profile_sha256:
            raise PromptControlError(
                f"{condition.value} benchmark trial physical profile mismatch"
            )
        if _model_identity(trial.get("mujoco_model_identity"), condition=condition) != (
            model_identity
        ):
            raise PromptControlError(
                f"{condition.value} benchmark trial model identity mismatch"
            )

        success = trial.get("success")
        if not isinstance(success, bool):
            raise PromptControlError(
                f"{condition.value} benchmark trial success must be boolean"
            )
        status = trial.get("status")
        if status not in {
            SemanticTrialStatus.SCORED.value,
            SemanticTrialStatus.EXECUTION_FAILURE.value,
        }:
            raise PromptControlError(
                f"{condition.value} benchmark trial status is invalid"
            )
        failure_reason = trial.get("failure_reason")
        failure_evidence = trial.get("failure_evidence")
        object_error = trial.get("object_position_error_m")
        source = _model_identity_source(
            trial.get("model_identity_source"),
            condition=condition,
        )
        if status == SemanticTrialStatus.EXECUTION_FAILURE.value:
            if success or object_error is not None:
                raise PromptControlError(
                    f"{condition.value} execution failure has terminal outcome"
                )
            if not isinstance(failure_reason, str) or not failure_reason:
                raise PromptControlError(
                    f"{condition.value} execution failure reason is invalid"
                )
            try:
                validate_failure_evidence(failure_reason, failure_evidence)
            except MujocoBenchmarkError as failure_error:
                raise PromptControlError(
                    f"{condition.value} benchmark {failure_error}"
                ) from failure_error
        else:
            if source is not ModelIdentitySource.ROLLOUT_SESSION:
                raise PromptControlError(
                    f"{condition.value} scored trial model identity source is invalid"
                )
            if (
                not isinstance(object_error, (int, float))
                or isinstance(object_error, bool)
                or not isfinite(float(object_error))
                or float(object_error) < 0.0
            ):
                raise PromptControlError(
                    f"{condition.value} benchmark object_position_error_m is invalid"
                )
            if failure_evidence is not None:
                raise PromptControlError(
                    f"{condition.value} scored trial has failure evidence"
                )
            errors.append(float(object_error))
        if success and failure_reason is not None:
            raise PromptControlError(
                f"{condition.value} successful trial has failure reason"
            )
        if not success and (
            not isinstance(failure_reason, str) or not failure_reason
        ):
            raise PromptControlError(
                f"{condition.value} failed trial reason is invalid"
            )

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
        try:
            artifact_source = artifact_path.read_bytes()
        except OSError as error:
            raise PromptControlError(
                f"{condition.value} benchmark trial artifact is invalid"
            ) from error
        if sha256(artifact_source).hexdigest() != artifact_sha256:
            raise PromptControlError(
                f"{condition.value} benchmark trial artifact hash mismatch"
            )
        artifact_payload = _json_object(artifact_source, condition=condition)
        if (
            _model_identity(
                artifact_payload.get("mujoco_model_identity"),
                condition=condition,
            )
            != model_identity
            or artifact_payload.get("model_identity_source") != source.value
        ):
            raise PromptControlError(
                f"{condition.value} benchmark trial artifact model identity mismatch"
            )
        identity_sources.add(source.value)
        validated.append(trial)

    if tuple(actual_trials) != expected_trials:
        raise PromptControlError(
            f"{condition.value} benchmark task/seed schedule mismatch"
        )
    return validated, errors, sorted(identity_sources)


def _validate_summary(
    report: Mapping[str, Any],
    *,
    condition: PromptCondition,
    trials: Sequence[Mapping[str, Any]],
    errors: Sequence[float],
) -> dict[str, Any]:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise PromptControlError(
            f"{condition.value} benchmark summary must be an object"
        )

    success_count = sum(int(bool(trial["success"])) for trial in trials)
    trial_count = len(trials)
    scored_count = len(errors)
    success_rate = success_count / trial_count
    error_mean = None if not errors else fmean(errors)
    error_max = None if not errors else max(errors)
    failure_counts = dict(
        sorted(
            Counter(
                str(trial["failure_reason"])
                for trial in trials
                if trial.get("failure_reason") is not None
            ).items()
        )
    )
    expected_values = {
        "task_count": 1,
        "trial_count": trial_count,
        "scored_trial_count": scored_count,
        "execution_failure_count": trial_count - scored_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "all_trials_successful": success_count == trial_count,
        "object_position_error_mean_m": error_mean,
        "object_position_error_max_m": error_max,
        "failure_counts": failure_counts,
    }
    for field, expected in expected_values.items():
        if summary.get(field) != expected:
            raise PromptControlError(
                f"{condition.value} benchmark summary {field} mismatch"
            )

    interval = summary.get("success_rate_95ci")
    if not isinstance(interval, list):
        raise PromptControlError(
            f"{condition.value} benchmark summary evidence is invalid"
        )
    expected_result = "pass" if success_count == trial_count else "fail"
    if report.get("result") != expected_result:
        raise PromptControlError(
            f"{condition.value} benchmark result disagrees with trials"
        )
    return {
        "benchmark_result": expected_result,
        "trial_count": trial_count,
        "scored_trial_count": scored_count,
        "execution_failure_count": trial_count - scored_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "success_rate_95ci": list(interval),
        "object_position_error_mean_m": error_mean,
        "object_position_error_max_m": error_max,
        "failure_counts": failure_counts,
    }


def _validate_benchmark_report(
    report: Mapping[str, Any],
    *,
    condition: PromptCondition,
    config: ProjectConfig,
    manifest: SemanticManifest,
    semantic_manifest_sha256: str,
    checkpoint_sha256: str,
    training_report_sha256: str,
    prompt_path: Path,
    prompt_manifest_path: Path,
    prompt: EpisodeData,
    artifact_dir: Path,
    expected_trials: tuple[tuple[str, int], ...],
    expected_checkpoint_id: str | None,
    expected_training_evidence: str | None,
    expected_object_profile_sha256: str | None,
    expected_model_identity: Mapping[str, object] | None,
) -> tuple[dict[str, Any], str, str, dict[str, Any], str, dict[str, object], list[str]]:
    expectation, identity_matches, task_match = _task_expectation(condition)
    task = manifest.heldout_tasks[0]
    expected_fields = {
        "schema_version": SEMANTIC_REPORT_SCHEMA,
        "gate": CHECKPOINT_SEMANTIC_GATE,
        "policy": CHECKPOINT_SEMANTIC_POLICY,
        "benchmark_scope": CHECKPOINT_SEMANTIC_SCOPE,
        "benchmark_id": manifest.benchmark_id,
        "config_sha256": project_config_sha256(config),
        "manifest_sha256": semantic_manifest_sha256,
        "dataset_task": manifest.dataset_task,
        "dataset_task_index": manifest.dataset_task_index,
        "criterion": TERMINAL_CRITERION,
        "object_body": task.object_body,
        "task_disjoint": True,
        "train_task_ids": list(manifest.train_task_ids),
        "heldout_task_ids": [task.task_id for task in manifest.heldout_tasks],
        "checkpoint_sha256": checkpoint_sha256,
        "training_report_sha256": training_report_sha256,
        "prompt_npz_sha256": file_sha256(prompt_path),
        "prompt_manifest_sha256": file_sha256(prompt_manifest_path),
        "prompt_fingerprint": prompt.fingerprint,
        "prompt_task": prompt.task,
        "prompt_task_index": prompt.task_index,
        "prompt_task_expectation": expectation.value,
        "prompt_task_identity_matches_manifest": identity_matches,
        "prompt_task_match": task_match,
        "task_disjoint_basis": TASK_DISJOINT_BASIS,
        "checkpoint_task_split": CHECKPOINT_TASK_SPLIT,
        "semantic_task_mapping": SEMANTIC_TASK_MAPPING,
        "robot_used": False,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
    }
    for field, expected in expected_fields.items():
        _expect_field(
            report,
            field=field,
            expected=expected,
            condition=condition,
        )

    raw_profile = report.get("object_physical_profile")
    if not isinstance(raw_profile, Mapping):
        raise PromptControlError(
            f"{condition.value} benchmark object_physical_profile is invalid"
        )
    object_profile = dict(raw_profile)
    object_profile_sha256 = _validate_digest(
        report.get("object_physical_profile_sha256"),
        field="object_physical_profile_sha256",
        condition=condition,
    )
    try:
        actual_profile_sha256 = canonical_json_sha256(object_profile)
    except ValueError as error:
        raise PromptControlError(
            f"{condition.value} benchmark object_physical_profile is invalid"
        ) from error
    if actual_profile_sha256 != object_profile_sha256:
        raise PromptControlError(
            f"{condition.value} benchmark object physical profile hash mismatch"
        )
    if (
        expected_object_profile_sha256 is not None
        and object_profile_sha256 != expected_object_profile_sha256
    ):
        raise PromptControlError(
            f"{condition.value} benchmark object physical profile changed"
        )
    model_identity = _model_identity(
        report.get("mujoco_model_identity"),
        condition=condition,
    )
    if (
        expected_model_identity is not None
        and model_identity != expected_model_identity
    ):
        raise PromptControlError(
            f"{condition.value} benchmark model identity changed"
        )
    report_identity_sources = _model_identity_sources(
        report.get("model_identity_sources"),
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

    training_evidence = _validate_digest(
        report.get("training_evidence_sha256"),
        field="training_evidence_sha256",
        condition=condition,
    )
    if (
        expected_training_evidence is not None
        and training_evidence != expected_training_evidence
    ):
        raise PromptControlError(
            f"{condition.value} benchmark training_evidence_sha256 mismatch"
        )

    trials, errors, trial_identity_sources = _validate_trials(
        report,
        condition=condition,
        artifact_dir=artifact_dir,
        expected_trials=expected_trials,
        object_body=task.object_body,
        object_profile_sha256=object_profile_sha256,
        model_identity=model_identity,
    )
    if report_identity_sources != trial_identity_sources:
        raise PromptControlError(
            f"{condition.value} benchmark model identity sources mismatch"
        )
    if report.get("semantic_mujoco_object_state_evaluated") is not bool(errors):
        raise PromptControlError(
            f"{condition.value} benchmark terminal criterion marker mismatch"
        )
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise PromptControlError(
            f"{condition.value} benchmark summary must be an object"
        )
    if (
        _model_identity(summary.get("mujoco_model_identity"), condition=condition)
        != model_identity
        or _model_identity_sources(
            summary.get("model_identity_sources"),
            condition=condition,
        )
        != trial_identity_sources
    ):
        raise PromptControlError(
            f"{condition.value} benchmark summary model identity mismatch"
        )
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise PromptControlError(f"{condition.value} benchmark tasks must be a list")
    if len(tasks) != len(manifest.heldout_tasks):
        raise PromptControlError(
            f"{condition.value} benchmark task summaries mismatch"
        )
    for task_summary in tasks:
        if not isinstance(task_summary, Mapping):
            raise PromptControlError(
                f"{condition.value} benchmark task summary is invalid"
            )
        if (
            _model_identity(
                task_summary.get("mujoco_model_identity"),
                condition=condition,
            )
            != model_identity
            or _model_identity_sources(
                task_summary.get("model_identity_sources"),
                condition=condition,
            )
            != trial_identity_sources
        ):
            raise PromptControlError(
                f"{condition.value} benchmark task model identity mismatch"
            )
    metrics = _validate_summary(
        report,
        condition=condition,
        trials=trials,
        errors=errors,
    )
    return (
        metrics,
        checkpoint_id,
        training_evidence,
        object_profile,
        object_profile_sha256,
        model_identity,
        trial_identity_sources,
    )


def run_mujoco_semantic_prompt_control_suite(
    config: ProjectConfig,
    *,
    prompt_control_manifest_path: str | Path,
    semantic_manifest_path: str | Path,
    training_report_path: str | Path,
    direction_preregistration_path: str | Path | None = None,
    artifact_dir: str | Path,
    report_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run every prompt condition against one semantic object-state task."""

    prompt_control_target = Path(prompt_control_manifest_path)
    semantic_target = Path(semantic_manifest_path)
    training_report_target = Path(training_report_path)
    artifact_root = Path(artifact_dir)
    report_target = Path(report_path)
    _preflight_outputs(artifact_root, report_target)

    # Freeze required evidence before any condition artifact is created.
    prompt_control_source = _read_source(
        prompt_control_target,
        name="prompt-control manifest",
    )
    semantic_source = _read_source(semantic_target, name="semantic manifest")
    training_report_source = _read_source(
        training_report_target,
        name="training report",
    )
    direction_target = (
        None
        if direction_preregistration_path is None
        else Path(direction_preregistration_path)
    )
    direction_source = (
        None
        if direction_target is None
        else _read_source(
            direction_target,
            name="semantic direction preregistration",
        )
    )
    prompt_manifest = load_prompt_control_manifest_bytes(
        prompt_control_source,
        root=prompt_control_target.parent,
    )
    checkpoint_source = _read_source(
        prompt_manifest.checkpoint_path,
        name="checkpoint",
    )
    semantic_manifest = load_semantic_manifest_bytes(semantic_source)
    if len(semantic_manifest.heldout_tasks) != 1:
        raise PromptControlError(
            "semantic prompt controls require exactly one held-out task"
        )

    source_config = ProjectConfig.load(prompt_manifest.config_path)
    _validate_source_config(source_config, config)
    sources = _load_sources(prompt_manifest)
    _validate_source_limits(sources, config)
    _validate_dataset_task(sources, semantic_manifest)

    prompt_control_sha256 = sha256(prompt_control_source).hexdigest()
    semantic_manifest_sha256 = sha256(semantic_source).hexdigest()
    training_report_sha256 = sha256(training_report_source).hexdigest()
    checkpoint_sha256 = sha256(checkpoint_source).hexdigest()
    mujoco_config_sha256 = project_config_sha256(config)
    direction_plan = None
    direction_source_sha256 = None
    if direction_source is not None:
        direction_source_sha256 = sha256(direction_source).hexdigest()
        direction_plan = _load_directionality_plan(
            direction_source,
            suite_id=prompt_manifest.suite_id,
            prompt_control_manifest_sha256=prompt_control_sha256,
            semantic_manifest_sha256=semantic_manifest_sha256,
            training_report_sha256=training_report_sha256,
            checkpoint_sha256=checkpoint_sha256,
            mujoco_config_sha256=mujoco_config_sha256,
        )
    expected_trials = tuple(
        (task.task_id, seed)
        for task in semantic_manifest.heldout_tasks
        for seed in task.seeds
    )
    prompts = _condition_prompts(
        sources,
        config=config,
        seed=prompt_manifest.seed,
    )
    bundle_root = prompt_control_target.parent.resolve()
    source_records = {
        name: _source_record(
            bundle_root=bundle_root,
            path=path,
            episode=episode,
        )
        for name, (path, episode) in sources.items()
    }

    prompt_root = artifact_root / "prompts"
    benchmark_root = artifact_root / "semantic_benchmark_artifacts"
    benchmark_report_root = artifact_root / "semantic_benchmark_reports"
    condition_reports: list[dict[str, Any]] = []
    checkpoint_id: str | None = None
    training_evidence: str | None = None
    object_profile: dict[str, Any] | None = None
    object_profile_sha256: str | None = None
    model_identity: dict[str, object] | None = None
    model_identity_sources: set[str] = set()

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
        expectation, identity_matches, task_match = _task_expectation(condition)
        trial_artifact_dir = benchmark_root / condition.value
        benchmark_report = run_checkpoint_benchmark(
            config,
            manifest_path=semantic_target,
            artifact_dir=trial_artifact_dir,
            checkpoint_path=prompt_manifest.checkpoint_path,
            training_report_path=training_report_target,
            prompt_path=prompt_path,
            prompt_manifest_path=prompt_manifest_path,
            device=device,
            prompt_task_expectation=expectation,
        )
        (
            metrics,
            observed_checkpoint_id,
            observed_training_evidence,
            observed_object_profile,
            observed_object_profile_sha256,
            observed_model_identity,
            observed_identity_sources,
        ) = (
            _validate_benchmark_report(
                benchmark_report,
                condition=condition,
                config=config,
                manifest=semantic_manifest,
                semantic_manifest_sha256=semantic_manifest_sha256,
                checkpoint_sha256=checkpoint_sha256,
                training_report_sha256=training_report_sha256,
                prompt_path=prompt_path,
                prompt_manifest_path=prompt_manifest_path,
                prompt=condition_episode,
                artifact_dir=trial_artifact_dir,
                expected_trials=expected_trials,
                expected_checkpoint_id=checkpoint_id,
                expected_training_evidence=training_evidence,
                expected_object_profile_sha256=object_profile_sha256,
                expected_model_identity=model_identity,
            )
        )
        if checkpoint_id is None:
            checkpoint_id = observed_checkpoint_id
        if training_evidence is None:
            training_evidence = observed_training_evidence
        if object_profile is None:
            object_profile = observed_object_profile
            object_profile_sha256 = observed_object_profile_sha256
        if model_identity is None:
            model_identity = observed_model_identity
        model_identity_sources.update(observed_identity_sources)

        benchmark_report_path = benchmark_report_root / f"{condition.value}.json"
        write_benchmark_report(benchmark_report_path, benchmark_report)
        condition_reports.append(
            {
                "condition": condition.value,
                "source_episode": source_name,
                "condition_prompt_fingerprint": prompts[condition].fingerprint,
                "prompt_episode_fingerprint": condition_episode.fingerprint,
                "prompt_task": condition_episode.task,
                "prompt_task_index": condition_episode.task_index,
                "prompt_task_expectation": expectation.value,
                "prompt_task_identity_matches_manifest": identity_matches,
                "prompt_task_match": task_match,
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
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_id": observed_checkpoint_id,
                "training_report_sha256": training_report_sha256,
                "training_evidence_sha256": observed_training_evidence,
                "task_disjoint_basis": TASK_DISJOINT_BASIS,
                "checkpoint_task_split": CHECKPOINT_TASK_SPLIT,
                "semantic_task_mapping": SEMANTIC_TASK_MAPPING,
                "object_body": semantic_manifest.heldout_tasks[0].object_body,
                "object_physical_profile_sha256": (
                    observed_object_profile_sha256
                ),
                "mujoco_model_identity": observed_model_identity,
                "model_identity_sources": observed_identity_sources,
                **metrics,
            }
        )

    matched = condition_reports[0]
    matched_success_rate = float(matched["success_rate"])
    matched_error = matched["object_position_error_mean_m"]
    matched_error_mean = None if matched_error is None else float(matched_error)
    for condition_report in condition_reports:
        condition_report["success_rate_delta_vs_matched"] = (
            float(condition_report["success_rate"]) - matched_success_rate
        )
        condition_error = condition_report["object_position_error_mean_m"]
        if matched_error_mean is None or condition_error is None:
            condition_report[
                "object_position_error_mean_delta_vs_matched_m"
            ] = None
        else:
            condition_report[
                "object_position_error_mean_delta_vs_matched_m"
            ] = float(condition_error) - matched_error_mean

    total_trials = sum(int(item["trial_count"]) for item in condition_reports)
    scored_trials = sum(
        int(item["scored_trial_count"]) for item in condition_reports
    )
    execution_failures = sum(
        int(item["execution_failure_count"]) for item in condition_reports
    )
    terminal_evaluated = scored_trials > 0
    if object_profile is None or object_profile_sha256 is None:
        raise PromptControlError("semantic prompt controls produced no object profile")
    if model_identity is None:
        raise PromptControlError("semantic prompt controls produced no model identity")
    identity_sources = sorted(model_identity_sources)

    directionality_evaluation = None
    direction_artifact = None
    if direction_plan is not None and direction_source is not None:
        directionality_evaluation = _evaluate_directionality(
            direction_plan,
            condition_reports,
        )
        direction_artifact = _copy_directionality_plan(
            direction_source,
            artifact_root,
        )

    directionality_preregistered = directionality_evaluation is not None
    limitations = [
        "dataset-to-object task semantics are declared by the manifest",
    ]
    if directionality_preregistered:
        limitations.extend(
            (
                "directionality is a local input-hash-bound object-state diagnostic",
                "external preregistration timing is not verified",
            )
        )
    else:
        limitations.append(
            "condition deltas are descriptive and not preregistered"
        )
    limitations.extend(
        (
            "MuJoCo prompt sensitivity is not proof of prompt causality",
            "simulation evidence does not authorize real robot output",
        )
    )
    if execution_failures:
        limitations.append(
            "execution failures do not have terminal object-state metrics"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": MUJOCO_SEMANTIC_PROMPT_CONTROL_GATE,
        "result": "complete",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "scope": MUJOCO_SEMANTIC_PROMPT_CONTROL_SCOPE,
        "result_semantics": (
            "all prompt conditions completed against a manifest-declared "
            "object-state criterion"
        ),
        "suite_id": prompt_manifest.suite_id,
        "robot_used": False,
        "recorded_live_context_used": False,
        "mujoco_live_context_used": True,
        "mujoco_terminal_success_evaluated": terminal_evaluated,
        "semantic_mujoco_object_state_evaluated": terminal_evaluated,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "scientific_pass_fail_evaluated": directionality_preregistered,
        "directionality_preregistered": directionality_preregistered,
        "prompt_control_manifest_sha256": prompt_control_sha256,
        "semantic_manifest_sha256": semantic_manifest_sha256,
        "training_report_sha256": training_report_sha256,
        "training_evidence_sha256": training_evidence,
        "source_config_sha256": project_config_sha256(source_config),
        "mujoco_config_sha256": mujoco_config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "benchmark_id": semantic_manifest.benchmark_id,
        "dataset_task": semantic_manifest.dataset_task,
        "dataset_task_index": semantic_manifest.dataset_task_index,
        "seed": prompt_manifest.seed,
        "condition_generator": CONDITION_GENERATOR,
        "condition_safety_basis": "mujoco_config",
        "task_disjoint_basis": TASK_DISJOINT_BASIS,
        "checkpoint_task_split": CHECKPOINT_TASK_SPLIT,
        "prompt_task_match": PROMPT_TASK_MATCH,
        "semantic_task_mapping": SEMANTIC_TASK_MAPPING,
        "object_body": semantic_manifest.heldout_tasks[0].object_body,
        "object_physical_profile": object_profile,
        "object_physical_profile_sha256": object_profile_sha256,
        "mujoco_model_identity": model_identity,
        "model_identity_sources": identity_sources,
        "source_episodes": source_records,
        "summary": {
            "condition_count": len(condition_reports),
            "trials_per_condition": len(expected_trials),
            "total_trial_count": total_trials,
            "scored_trial_count": scored_trials,
            "execution_failure_count": execution_failures,
            "matched_success_rate": matched_success_rate,
            "matched_object_position_error_mean_m": matched_error_mean,
            "mujoco_model_identity": model_identity,
            "model_identity_sources": identity_sources,
        },
        "conditions": condition_reports,
        "limitations": limitations,
    }
    if (
        directionality_evaluation is not None
        and direction_artifact is not None
        and direction_target is not None
        and direction_source_sha256 is not None
    ):
        report["preregistration_level"] = "local_input_hash_bound_before_rollout"
        report["external_preregistration_timestamp_verified"] = False
        report["directionality_evaluation"] = directionality_evaluation
        report["semantic_direction_preregistration_sha256"] = (
            direction_source_sha256
        )
        report["semantic_direction_preregistration_artifact"] = (
            direction_artifact.relative_to(artifact_root).as_posix()
        )
        report["semantic_direction_preregistration_artifact_sha256"] = (
            file_sha256(direction_artifact)
        )
    write_benchmark_report(report_target, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run CompactWAM prompt controls against a MuJoCo object-state task."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MUJOCO_CONFIG_PATH)
    parser.add_argument("--prompt-control-manifest", required=True, type=Path)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--direction-preregistration", type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        report = run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(args.config),
            prompt_control_manifest_path=args.prompt_control_manifest,
            semantic_manifest_path=args.semantic_manifest,
            training_report_path=args.training_report,
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
        RuntimeErrorState,
        TensorizerError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "MUJOCO_SEMANTIC_PROMPT_CONTROL_GATE",
    "MUJOCO_SEMANTIC_PROMPT_CONTROL_SCOPE",
    "main",
    "run_mujoco_semantic_prompt_control_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
