"""MuJoCo semantic controls driven by reviewed human-video task specs."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import (
    CheckpointError,
    load_compact_wam_bundle,
)
from .config import ConfigError, DEFAULT_MUJOCO_CONFIG_PATH, ProjectConfig
from .contracts import ContractError
from .deployment import canonical_json_sha256, file_sha256, project_config_sha256
from .model import ModelContractError
from .mujoco_benchmark import MujocoBenchmarkError
from .mujoco_cli import MujocoCLIError, _checkpoint_identity
from .mujoco_identity import MujocoIdentityError, MujocoModelIdentity
from .mujoco_semantic_benchmark import (
    SCHEMA_VERSION as SEMANTIC_REPORT_SCHEMA,
    TASK_SPEC_CHECKPOINT_SEMANTIC_GATE,
    TASK_SPEC_CHECKPOINT_SEMANTIC_POLICY,
    TASK_SPEC_CHECKPOINT_SEMANTIC_SCOPE,
    ModelIdentitySource,
    PromptTaskExpectation,
    SemanticManifest,
    SemanticTrialStatus,
    _validate_training_report,
    load_semantic_manifest_bytes,
    run_task_spec_checkpoint_benchmark,
    validate_failure_evidence,
    write_benchmark_report,
)
from .paired_data import (
    HumanRobotPairRecord,
    PairedDataError,
    SemanticMatchStatus,
    load_human_robot_pairs,
    paired_data_audit,
)
from .paired_task_specs import PairedTaskSpecError, human_prompt_from_pair
from .policy import PolicyError
from .rollout import (
    EXECUTABILITY_TRACE_SCHEMA,
    JOINT_LIMIT_MARGIN_FIELDS,
    RolloutError,
    validate_executability_trace,
    validate_mujoco_scored_trace,
)
from .runtime import RuntimeErrorState
from .task_specs import (
    HumanVideoFrame,
    HumanVideoPrompt,
    TaskSpecError,
    TaskSpecProvenance,
)
from .tensorizer import TensorizerError


SCHEMA_VERSION = 4
MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_GATE = (
    "G8-paired-human-video-semantic-object-state"
)
MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_SCOPE = (
    "diagnostic_paired_human_video_mujoco_object_state"
)
CONDITION_GENERATOR = "paired_human_video_prompt_controls_v1"
FUTURE_LATENT_OUTCOME_CLASSIFIER = (
    "lower_mse_and_lower_success_rate_vs_matched_v1"
)


class PairedSemanticPromptControlError(ValueError):
    """Raised when paired task-spec evidence is incomplete or inconsistent."""


class PairedSemanticPromptCondition(StrEnum):
    MATCHED_HUMAN_VIDEO = "matched_human_video"
    WRONG_TASK_HUMAN_VIDEO = "wrong_task_human_video"
    NULL_HUMAN_VIDEO = "null_human_video"


@dataclass(frozen=True, slots=True)
class _ConditionInput:
    condition: PairedSemanticPromptCondition
    prompt: HumanVideoPrompt
    task: str
    task_index: int
    expectation: PromptTaskExpectation
    source_pair: HumanRobotPairRecord | None
    derived_from: HumanRobotPairRecord | None = None


def _read_source(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PairedSemanticPromptControlError(
            f"failed to read {name}: {error}"
        ) from error


def _preflight_outputs(artifact_dir: Path, report_path: Path) -> None:
    if artifact_dir.exists():
        raise PairedSemanticPromptControlError(
            f"paired semantic artifact directory already exists: {artifact_dir}"
        )
    if report_path.exists():
        raise PairedSemanticPromptControlError(
            f"paired semantic report already exists: {report_path}"
        )


def _reviewed_prompts(
    pairs: Sequence[HumanRobotPairRecord],
) -> dict[str, HumanVideoPrompt]:
    prompts: dict[str, HumanVideoPrompt] = {}
    for pair in pairs:
        if pair.semantic_match is not SemanticMatchStatus.HUMAN_REVIEWED:
            raise PairedSemanticPromptControlError(
                "paired semantic controls require semantic_match=human_reviewed"
            )
        if pair.human_task_spec is None:
            raise PairedSemanticPromptControlError(
                "paired semantic controls require a checksum-bound human task-spec"
            )
        try:
            prompts[pair.pair_id] = human_prompt_from_pair(pair)
        except PairedTaskSpecError as error:
            raise PairedSemanticPromptControlError(str(error)) from error
    return prompts


def _null_prompt(
    prompt: HumanVideoPrompt,
    *,
    pair: HumanRobotPairRecord,
) -> HumanVideoPrompt:
    frames = tuple(
        HumanVideoFrame(
            timestamp_s=frame.timestamp_s,
            rgb=np.zeros_like(frame.rgb),
        )
        for frame in prompt.frames
    )
    digest = sha256()
    digest.update(b"so101_wam.null_human_video.v1")
    digest.update(pair.fingerprint.encode("ascii"))
    digest.update(pair.task.encode("utf-8"))
    for frame in frames:
        digest.update(np.float64(frame.timestamp_s).tobytes())
        digest.update(np.asarray(frame.rgb.shape, dtype=np.int32).tobytes())
        digest.update(frame.rgb.tobytes())
    return HumanVideoPrompt(
        frames=frames,
        provenance=TaskSpecProvenance(
            source_id=f"derived:null:{pair.fingerprint}",
            source_sha256=digest.hexdigest(),
        ),
        text_metadata=pair.task,
    )


def _condition_inputs(
    pairs: Sequence[HumanRobotPairRecord],
    prompts: Mapping[str, HumanVideoPrompt],
    manifest: SemanticManifest,
) -> tuple[_ConditionInput, ...]:
    identities = {(pair.task_index, pair.task) for pair in pairs}
    if len(identities) < 2:
        raise PairedSemanticPromptControlError(
            "paired semantic controls require a different validation task"
        )
    matched = sorted(
        (
            pair
            for pair in pairs
            if pair.task == manifest.dataset_task
            and pair.task_index == manifest.dataset_task_index
        ),
        key=lambda pair: pair.pair_id,
    )
    if not matched:
        raise PairedSemanticPromptControlError(
            "validation pairs require a matching semantic dataset task"
        )
    wrong = sorted(
        (
            pair
            for pair in pairs
            if pair.task != manifest.dataset_task
            and pair.task_index != manifest.dataset_task_index
        ),
        key=lambda pair: pair.pair_id,
    )
    if not wrong:
        raise PairedSemanticPromptControlError(
            "paired semantic controls require a different validation task"
        )

    matched_pair = matched[0]
    wrong_pair = wrong[0]
    matched_prompt = prompts[matched_pair.pair_id]
    return (
        _ConditionInput(
            PairedSemanticPromptCondition.MATCHED_HUMAN_VIDEO,
            matched_prompt,
            matched_pair.task,
            matched_pair.task_index,
            PromptTaskExpectation.MATCH_MANIFEST,
            matched_pair,
        ),
        _ConditionInput(
            PairedSemanticPromptCondition.WRONG_TASK_HUMAN_VIDEO,
            prompts[wrong_pair.pair_id],
            wrong_pair.task,
            wrong_pair.task_index,
            PromptTaskExpectation.EXPECTED_MISMATCH,
            wrong_pair,
        ),
        _ConditionInput(
            PairedSemanticPromptCondition.NULL_HUMAN_VIDEO,
            _null_prompt(matched_prompt, pair=matched_pair),
            matched_pair.task,
            matched_pair.task_index,
            PromptTaskExpectation.MATCH_MANIFEST,
            None,
            matched_pair,
        ),
    )


def _validate_prompt_resolution(
    prompts: Mapping[str, HumanVideoPrompt],
    config: ProjectConfig,
) -> None:
    expected = (config.mujoco.camera_height, config.mujoco.camera_width)
    mismatched = sorted(
        pair_id for pair_id, prompt in prompts.items() if prompt.resolution != expected
    )
    if mismatched:
        raise PairedSemanticPromptControlError(
            "human task-spec resolution must match MuJoCo cameras "
            f"{expected}; mismatched pairs={mismatched}"
        )


def _pair_inventory(
    pairs: Sequence[HumanRobotPairRecord],
) -> list[dict[str, str | int]]:
    inventory: list[dict[str, str | int]] = []
    for pair in sorted(pairs, key=lambda item: item.pair_id):
        if pair.human_task_spec is None:
            raise PairedSemanticPromptControlError(
                "paired semantic controls require a checksum-bound human task-spec"
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


def _json_object(source: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairedSemanticPromptControlError(
            f"{name} must be valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise PairedSemanticPromptControlError(f"{name} must be a JSON object")
    return value


def _model_identity_payload(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PairedSemanticPromptControlError(f"{name} model identity is invalid")
    try:
        return MujocoModelIdentity.from_payload(value).to_payload()
    except MujocoIdentityError as error:
        raise PairedSemanticPromptControlError(
            f"{name} model identity is invalid: {error}"
        ) from error


def _model_identity_sources(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise PairedSemanticPromptControlError(
            f"{name} model identity sources are invalid"
        )
    sources: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise PairedSemanticPromptControlError(
                f"{name} model identity source is invalid"
            )
        try:
            sources.append(ModelIdentitySource(item).value)
        except ValueError as error:
            raise PairedSemanticPromptControlError(
                f"{name} model identity source is invalid"
            ) from error
    return sources


def _model_identity_source(value: object, *, name: str) -> ModelIdentitySource:
    if not isinstance(value, str):
        raise PairedSemanticPromptControlError(
            f"{name} model identity source is invalid"
        )
    try:
        return ModelIdentitySource(value)
    except ValueError as error:
        raise PairedSemanticPromptControlError(
            f"{name} model identity source is invalid"
        ) from error


def _validate_candidate(
    *,
    config: ProjectConfig,
    checkpoint_path: Path,
    checkpoint_source: bytes,
    training_report_source: bytes,
    manifest: SemanticManifest,
    pairs: Sequence[HumanRobotPairRecord],
    pair_digest: str,
    device: str,
) -> tuple[str, str]:
    bundle = load_compact_wam_bundle(checkpoint_path, device=device)
    checkpoint_sha256, checkpoint_id = _checkpoint_identity(
        checkpoint_path,
        bundle.metadata,
    )
    if checkpoint_sha256 != sha256(checkpoint_source).hexdigest():
        raise PairedSemanticPromptControlError("checkpoint snapshot hash mismatch")
    if bundle.model.action_horizon != config.runtime.action_horizon:
        raise PairedSemanticPromptControlError(
            "checkpoint action_horizon does not match MuJoCo runtime"
        )
    try:
        training_evidence = _validate_training_report(
            training_report_source,
            checkpoint_metadata=bundle.metadata,
            checkpoint_architecture=bundle.architecture,
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_sha256,
            manifest=manifest,
        )
    except MujocoBenchmarkError as error:
        raise PairedSemanticPromptControlError(str(error)) from error
    report = _json_object(training_report_source, name="training report")
    protocol = report.get("protocol")
    if (
        bundle.metadata.get("paired_human_video") is not True
        or not isinstance(protocol, Mapping)
        or protocol.get("prompt_pairing") != "human_video_reviewed_pair"
        or protocol.get("prompt_modality") != "human_video_task_spec"
    ):
        raise PairedSemanticPromptControlError(
            "paired semantic controls require a paired human-video checkpoint"
        )
    optimization = report.get("optimization")
    if not isinstance(optimization, Mapping):
        raise PairedSemanticPromptControlError(
            "paired training report requires optimization settings"
        )
    for field, expected in (
        ("policy_hz", config.runtime.policy_hz),
        ("servo_hz", config.runtime.servo_hz),
    ):
        if optimization.get(field) != expected:
            raise PairedSemanticPromptControlError(
                f"paired training {field} does not match MuJoCo runtime"
            )
    data = report.get("data")
    if not isinstance(data, Mapping):
        raise PairedSemanticPromptControlError(
            "paired training report requires a data object"
        )
    if data.get("validation_pair_digest") != pair_digest:
        raise PairedSemanticPromptControlError(
            "validation pair digest does not match paired training report"
        )
    if data.get("validation_pair_count") != len(pairs):
        raise PairedSemanticPromptControlError(
            "validation pair count does not match paired training report"
        )
    if data.get("validation_pairs") != _pair_inventory(pairs):
        raise PairedSemanticPromptControlError(
            "validation pair inventory does not match paired training report"
        )
    return checkpoint_id, training_evidence


def _expected_identity(
    condition: _ConditionInput,
) -> tuple[bool, str]:
    if condition.expectation is PromptTaskExpectation.EXPECTED_MISMATCH:
        return False, "control_mismatch_verified"
    return True, "dataset_identity_verified"


def _validate_nested_report(
    report: Mapping[str, Any],
    *,
    condition: _ConditionInput,
    config: ProjectConfig,
    manifest: SemanticManifest,
    semantic_manifest_sha256: str,
    checkpoint_sha256: str,
    checkpoint_id: str,
    training_report_sha256: str,
    training_evidence_sha256: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    identity_matches, task_match = _expected_identity(condition)
    expected = {
        "schema_version": SEMANTIC_REPORT_SCHEMA,
        "gate": TASK_SPEC_CHECKPOINT_SEMANTIC_GATE,
        "policy": TASK_SPEC_CHECKPOINT_SEMANTIC_POLICY,
        "benchmark_scope": TASK_SPEC_CHECKPOINT_SEMANTIC_SCOPE,
        "benchmark_id": manifest.benchmark_id,
        "config_sha256": project_config_sha256(config),
        "manifest_sha256": semantic_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "training_report_sha256": training_report_sha256,
        "training_evidence_sha256": training_evidence_sha256,
        "task_spec_kind": "human_video",
        "task_spec_fingerprint": condition.prompt.fingerprint,
        "task_spec_source_id": condition.prompt.provenance.source_id,
        "task_spec_source_sha256": condition.prompt.provenance.source_sha256,
        "prompt_condition": condition.condition.value,
        "prompt_task": condition.task,
        "prompt_task_index": condition.task_index,
        "prompt_task_expectation": condition.expectation.value,
        "prompt_task_identity_matches_manifest": identity_matches,
        "prompt_task_match": task_match,
        "runtime_placeholder_prompt_used_by_model": False,
        "robot_used": False,
        "executability_trace_present": True,
        "future_visual_metric_available": False,
        "future_visual_success_claimed": False,
        "runtime_future_latent_success_claimed": False,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
    }
    for field, expected_value in expected.items():
        if report.get(field) != expected_value:
            raise PairedSemanticPromptControlError(
                f"{condition.condition.value} benchmark {field} mismatch"
            )
    model_identity = _model_identity_payload(
        report.get("mujoco_model_identity"),
        name=f"{condition.condition.value} benchmark",
    )
    model_identity_sources = _model_identity_sources(
        report.get("model_identity_sources"),
        name=f"{condition.condition.value} benchmark",
    )

    trials = report.get("trials")
    if not isinstance(trials, list):
        raise PairedSemanticPromptControlError(
            f"{condition.condition.value} benchmark trials must be a list"
        )
    expected_schedule = [
        (task.task_id, seed) for task in manifest.heldout_tasks for seed in task.seeds
    ]
    tasks_by_id = {task.task_id: task for task in manifest.heldout_tasks}
    actual_schedule: list[tuple[str, int]] = []
    errors: list[float] = []
    success_count = 0
    failure_counts: Counter[str] = Counter()
    trace_trial_count = 0
    trace_policy_steps = 0
    trace_servo_steps = 0
    trace_sent_actions = 0
    trace_shadow_steps = 0
    trace_safety_accepted = 0
    trace_safety_rejected = 0
    trace_safety_clipped = 0
    trace_joint_margins: dict[str, float | None] = {
        name: None for name in JOINT_LIMIT_MARGIN_FIELDS
    }
    trace_contact_progression = 0
    trace_contact_observations = 0
    trace_forbidden_contact_observations = 0
    trace_object_contact_observations = 0
    trace_minimum_contact_distance: float | None = None
    trace_collision_failures = 0
    future_latent_trace_trials = 0
    future_latent_predictions = 0
    future_latent_observations = 0
    future_latent_pairs = 0
    future_latent_censored = 0
    future_latent_elements = 0
    future_latent_squared_error = 0.0
    trial_identity_sources: set[str] = set()
    artifact_root = artifact_dir.resolve()
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise PairedSemanticPromptControlError("benchmark trial is invalid")
        task_id = trial.get("task_id")
        seed = trial.get("seed")
        if not isinstance(task_id, str) or not isinstance(seed, int):
            raise PairedSemanticPromptControlError(
                "benchmark trial identity is invalid"
            )
        semantic_task = tasks_by_id.get(task_id)
        if semantic_task is None:
            raise PairedSemanticPromptControlError("benchmark trial task is invalid")
        actual_schedule.append((task_id, seed))
        rollout_id = f"{task_id}:seed:{seed}"
        if trial.get("rollout_id") != rollout_id:
            raise PairedSemanticPromptControlError(
                "benchmark executability rollout identity is invalid"
            )
        status = trial.get("status")
        success = trial.get("success")
        failure_reason = trial.get("failure_reason")
        if not isinstance(success, bool):
            raise PairedSemanticPromptControlError("benchmark success is invalid")
        success_count += int(success)
        if failure_reason is not None:
            if not isinstance(failure_reason, str) or not failure_reason:
                raise PairedSemanticPromptControlError(
                    "benchmark failure reason is invalid"
                )
            failure_counts[failure_reason] += 1
        is_scored = status == SemanticTrialStatus.SCORED.value
        if is_scored:
            error = trial.get("object_position_error_m")
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not isfinite(float(error))
                or float(error) < 0.0
            ):
                raise PairedSemanticPromptControlError(
                    "benchmark object-position error is invalid"
                )
            errors.append(float(error))
            if trial.get("failure_evidence") is not None:
                raise PairedSemanticPromptControlError(
                    "scored benchmark trial has failure evidence"
                )
        elif status == SemanticTrialStatus.EXECUTION_FAILURE.value:
            if success or trial.get("object_position_error_m") is not None:
                raise PairedSemanticPromptControlError(
                    "execution failure has terminal outcome"
                )
            try:
                validate_failure_evidence(
                    str(failure_reason),
                    trial.get("failure_evidence"),
                )
            except MujocoBenchmarkError as error:
                raise PairedSemanticPromptControlError(str(error)) from error
        else:
            raise PairedSemanticPromptControlError("benchmark trial status is invalid")
        if success and failure_reason is not None:
            raise PairedSemanticPromptControlError(
                "successful benchmark trial has a failure reason"
            )
        if not success and (not isinstance(failure_reason, str) or not failure_reason):
            raise PairedSemanticPromptControlError(
                "failed benchmark trial requires a failure reason"
            )
        trial_identity = _model_identity_payload(
            trial.get("mujoco_model_identity"),
            name="benchmark trial",
        )
        if trial_identity != model_identity:
            raise PairedSemanticPromptControlError(
                "benchmark trial model identity mismatch"
            )
        trial_identity_source = _model_identity_source(
            trial.get("model_identity_source"),
            name="benchmark trial",
        )
        if is_scored and trial_identity_source is not ModelIdentitySource.ROLLOUT_SESSION:
            raise PairedSemanticPromptControlError(
                "scored benchmark trial model identity source mismatch"
            )
        trial_identity_sources.add(trial_identity_source.value)

        artifact = trial.get("artifact")
        artifact_sha256 = trial.get("artifact_sha256")
        if not isinstance(artifact, str) or not artifact:
            raise PairedSemanticPromptControlError("benchmark artifact is invalid")
        artifact_path = (artifact_dir / artifact).resolve()
        if not artifact_path.is_relative_to(artifact_root):
            raise PairedSemanticPromptControlError(
                "benchmark artifact escapes its directory"
            )
        if not artifact_path.is_file():
            raise PairedSemanticPromptControlError(
                "benchmark trial artifact hash mismatch"
            )
        artifact_source = _read_source(artifact_path, name="benchmark trial artifact")
        if sha256(artifact_source).hexdigest() != artifact_sha256:
            raise PairedSemanticPromptControlError(
                "benchmark trial artifact hash mismatch"
            )
        trace_artifact = trial.get("executability_trace_artifact")
        trace_sha256 = trial.get("executability_trace_sha256")
        if (
            trial.get("executability_trace_schema_version")
            != EXECUTABILITY_TRACE_SCHEMA
        ):
            raise PairedSemanticPromptControlError(
                "benchmark executability trace schema is invalid"
            )
        if not isinstance(trace_artifact, str) or not trace_artifact:
            raise PairedSemanticPromptControlError(
                "benchmark executability trace artifact is invalid"
            )
        trace_path = (artifact_dir / trace_artifact).resolve()
        if not trace_path.is_relative_to(artifact_root):
            raise PairedSemanticPromptControlError(
                "benchmark executability trace escapes its directory"
            )
        if not trace_path.is_file():
            raise PairedSemanticPromptControlError(
                "benchmark executability trace hash mismatch"
            )
        trace_source_bytes = _read_source(
            trace_path,
            name="benchmark executability trace",
        )
        if sha256(trace_source_bytes).hexdigest() != trace_sha256:
            raise PairedSemanticPromptControlError(
                "benchmark executability trace hash mismatch"
            )
        trace_source = _json_object(
            trace_source_bytes,
            name="benchmark executability trace",
        )
        try:
            trace = validate_executability_trace(
                trace_source,
                rollout_id=rollout_id,
            )
            if is_scored:
                validate_mujoco_scored_trace(
                    trace,
                    policy_steps=semantic_task.policy_steps,
                )
        except RolloutError as error:
            raise PairedSemanticPromptControlError(str(error)) from error
        trial_artifact = _json_object(artifact_source, name="benchmark trial artifact")
        artifact_identity = _model_identity_payload(
            trial_artifact.get("mujoco_model_identity"),
            name="benchmark trial artifact",
        )
        artifact_identity_source = _model_identity_source(
            trial_artifact.get("model_identity_source"),
            name="benchmark trial artifact",
        )
        if (
            artifact_identity != trial_identity
            or artifact_identity_source is not trial_identity_source
        ):
            raise PairedSemanticPromptControlError(
                "benchmark trial artifact model identity mismatch"
            )
        for field, expected_value in (
            ("rollout_id", rollout_id),
            ("executability_trace_schema_version", EXECUTABILITY_TRACE_SCHEMA),
            ("executability_trace_artifact", trace_artifact),
            ("executability_trace_sha256", trace_sha256),
            ("future_visual_metric_available", False),
            ("future_visual_success_claimed", False),
            (
                "runtime_future_latent_proxy_available",
                bool(trace["runtime_future_latent_proxy_available"]),
            ),
            ("runtime_future_latent_success_claimed", False),
        ):
            if trial_artifact.get(field) != expected_value:
                raise PairedSemanticPromptControlError(
                    f"benchmark trial artifact {field} mismatch"
                )
        trace_trial_count += 1
        trace_policy_steps += int(trace["policy_steps"])
        trace_servo_steps += int(trace["servo_steps"])
        trace_sent_actions += int(trace["sent_actions"])
        trace_shadow_steps += int(trace["shadow_steps"])
        trace_safety_accepted += int(trace["safety_accepted_steps"])
        trace_safety_rejected += int(trace["safety_rejected_steps"])
        trace_safety_clipped += int(trace["safety_clipped_steps"])
        for name, value in trace["joint_limit_margins"].items():
            if value is None:
                continue
            current = trace_joint_margins[name]
            trace_joint_margins[name] = (
                float(value) if current is None else min(current, float(value))
            )
        trace_contact_progression += int(trace["contact_progression_count"])
        trace_contact_observations += int(trace["contact_observation_count"])
        trace_forbidden_contact_observations += int(
            trace["forbidden_contact_observation_count"]
        )
        trace_object_contact_observations += int(
            trace["object_contact_observation_count"]
        )
        minimum_distance = trace["minimum_contact_distance_m"]
        if minimum_distance is not None:
            trace_minimum_contact_distance = (
                float(minimum_distance)
                if trace_minimum_contact_distance is None
                else min(trace_minimum_contact_distance, float(minimum_distance))
            )
        trace_collision_failures += int(trace["collision_failure_phase"] is not None)
        future_prediction_count = int(trace["future_latent_prediction_count"])
        future_latent_trace_trials += int(future_prediction_count > 0)
        future_latent_predictions += future_prediction_count
        future_latent_observations += int(trace["future_latent_observation_count"])
        future_latent_pairs += int(trace["future_latent_aligned_pair_count"])
        future_latent_censored += int(trace["future_latent_censored_pair_count"])
        future_latent_elements += int(trace["future_latent_element_count"])
        future_latent_squared_error += float(
            trace["future_latent_squared_error_sum"]
        )
    if actual_schedule != expected_schedule:
        raise PairedSemanticPromptControlError("benchmark trial schedule mismatch")
    if report.get("runtime_future_latent_proxy_available") is not bool(
        future_latent_pairs
    ):
        raise PairedSemanticPromptControlError(
            f"{condition.condition.value} benchmark "
            "runtime_future_latent_proxy_available mismatch"
        )

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise PairedSemanticPromptControlError("benchmark summary is invalid")
    if model_identity_sources != sorted(trial_identity_sources):
        raise PairedSemanticPromptControlError(
            "benchmark model identity sources mismatch"
        )
    if (
        _model_identity_payload(
            summary.get("mujoco_model_identity"),
            name="benchmark summary",
        )
        != model_identity
        or _model_identity_sources(
            summary.get("model_identity_sources"),
            name="benchmark summary",
        )
        != model_identity_sources
    ):
        raise PairedSemanticPromptControlError(
            "benchmark summary model identity mismatch"
        )
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise PairedSemanticPromptControlError("benchmark task summaries are invalid")
    if len(tasks) != len(manifest.heldout_tasks):
        raise PairedSemanticPromptControlError("benchmark task summaries mismatch")
    for task in tasks:
        if not isinstance(task, Mapping):
            raise PairedSemanticPromptControlError("benchmark task summary is invalid")
        if (
            _model_identity_payload(
                task.get("mujoco_model_identity"),
                name="benchmark task summary",
            )
            != model_identity
            or _model_identity_sources(
                task.get("model_identity_sources"),
                name="benchmark task summary",
            )
            != model_identity_sources
        ):
            raise PairedSemanticPromptControlError(
                "benchmark task model identity mismatch"
            )
    trial_count = len(trials)
    expected_summary = {
        "trial_count": trial_count,
        "scored_trial_count": len(errors),
        "execution_failure_count": trial_count - len(errors),
        "executability_trace_trial_count": trace_trial_count,
        "executability_policy_step_count": trace_policy_steps,
        "executability_servo_step_count": trace_servo_steps,
        "executability_sent_action_count": trace_sent_actions,
        "executability_shadow_step_count": trace_shadow_steps,
        "executability_safety_accepted_count": trace_safety_accepted,
        "executability_safety_rejected_count": trace_safety_rejected,
        "executability_safety_clipped_count": trace_safety_clipped,
        "executability_joint_limit_margins": trace_joint_margins,
        "executability_contact_progression_count": trace_contact_progression,
        "executability_contact_observation_count": trace_contact_observations,
        "executability_forbidden_contact_observation_count": (
            trace_forbidden_contact_observations
        ),
        "executability_object_contact_observation_count": (
            trace_object_contact_observations
        ),
        "executability_minimum_contact_distance_m": trace_minimum_contact_distance,
        "executability_collision_failure_count": trace_collision_failures,
        "future_latent_trace_trial_count": future_latent_trace_trials,
        "future_latent_prediction_count": future_latent_predictions,
        "future_latent_observation_count": future_latent_observations,
        "future_latent_aligned_pair_count": future_latent_pairs,
        "future_latent_censored_pair_count": future_latent_censored,
        "future_latent_element_count": future_latent_elements,
        "future_latent_squared_error_sum": future_latent_squared_error,
        "future_latent_mse_mean": (
            None
            if not future_latent_elements
            else future_latent_squared_error / future_latent_elements
        ),
        "success_count": success_count,
        "success_rate": success_count / trial_count,
        "object_position_error_mean_m": None if not errors else fmean(errors),
        "object_position_error_max_m": None if not errors else max(errors),
        "failure_counts": dict(sorted(failure_counts.items())),
        "mujoco_model_identity": model_identity,
        "model_identity_sources": model_identity_sources,
    }
    for field, expected_value in expected_summary.items():
        if summary.get(field) != expected_value:
            raise PairedSemanticPromptControlError(
                f"benchmark summary {field} mismatch"
            )
    expected_result = "pass" if success_count == trial_count else "fail"
    if report.get("result") != expected_result:
        raise PairedSemanticPromptControlError(
            "benchmark result disagrees with trial outcomes"
        )
    if report.get("semantic_mujoco_object_state_evaluated") is not bool(errors):
        raise PairedSemanticPromptControlError(
            "benchmark terminal criterion marker mismatch"
        )
    return expected_summary


def _condition_record(
    condition: _ConditionInput,
    *,
    benchmark_report_path: Path,
    artifact_root: Path,
    benchmark_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    source = condition.source_pair
    derived = condition.derived_from
    return {
        "condition": condition.condition.value,
        "source_pair_id": None if source is None else source.pair_id,
        "source_pair_fingerprint": None if source is None else source.fingerprint,
        "derived_from_pair_fingerprint": (
            None if derived is None else derived.fingerprint
        ),
        "human_video_sha256": None if source is None else source.human_video_sha256,
        "human_task_spec_sha256": (
            None
            if source is None or source.human_task_spec is None
            else source.human_task_spec.sha256
        ),
        "human_prompt_fingerprint": condition.prompt.fingerprint,
        "task_spec_source_id": condition.prompt.provenance.source_id,
        "task_spec_source_sha256": condition.prompt.provenance.source_sha256,
        "prompt_task": condition.task,
        "prompt_task_index": condition.task_index,
        "prompt_task_expectation": condition.expectation.value,
        "prompt_task_identity_matches_manifest": benchmark_report[
            "prompt_task_identity_matches_manifest"
        ],
        "prompt_task_match": benchmark_report["prompt_task_match"],
        "runtime_placeholder_prompt_used_by_model": False,
        "benchmark_report": benchmark_report_path.relative_to(artifact_root).as_posix(),
        "benchmark_report_sha256": file_sha256(benchmark_report_path),
        "object_body": benchmark_report["object_body"],
        "object_physical_profile_sha256": benchmark_report[
            "object_physical_profile_sha256"
        ],
        **metrics,
    }


def run_paired_semantic_prompt_controls(
    config: ProjectConfig,
    *,
    validation_pair_manifest_path: str | Path,
    semantic_manifest_path: str | Path,
    checkpoint_path: str | Path,
    training_report_path: str | Path,
    artifact_dir: str | Path,
    report_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run matched, wrong-task, and null reviewed human-video controls."""

    pair_target = Path(validation_pair_manifest_path)
    semantic_target = Path(semantic_manifest_path)
    checkpoint_target = Path(checkpoint_path)
    training_report_target = Path(training_report_path)
    artifact_root = Path(artifact_dir)
    report_target = Path(report_path)
    _preflight_outputs(artifact_root, report_target)

    pair_source = _read_source(pair_target, name="validation pair manifest")
    semantic_source = _read_source(semantic_target, name="semantic manifest")
    try:
        pairs = load_human_robot_pairs(pair_target)
        prompts = _reviewed_prompts(pairs)
        semantic_manifest = load_semantic_manifest_bytes(semantic_source)
    except (PairedDataError, MujocoBenchmarkError) as error:
        raise PairedSemanticPromptControlError(str(error)) from error
    if len(semantic_manifest.heldout_tasks) != 1:
        raise PairedSemanticPromptControlError(
            "paired semantic controls require exactly one held-out task"
        )
    if _read_source(pair_target, name="validation pair manifest") != pair_source:
        raise PairedSemanticPromptControlError(
            "validation pair manifest changed while it was being loaded"
        )
    _validate_prompt_resolution(prompts, config)
    conditions = _condition_inputs(pairs, prompts, semantic_manifest)

    checkpoint_source = _read_source(checkpoint_target, name="checkpoint")
    training_report_source = _read_source(
        training_report_target,
        name="training report",
    )
    pair_digest = str(paired_data_audit(pairs)["pair_digest"])
    checkpoint_sha256 = sha256(checkpoint_source).hexdigest()
    training_report_sha256 = sha256(training_report_source).hexdigest()
    semantic_manifest_sha256 = sha256(semantic_source).hexdigest()

    with TemporaryDirectory(prefix="so101-wam-paired-semantic-") as temp_dir:
        snapshot_root = Path(temp_dir)
        checkpoint_snapshot = snapshot_root / "checkpoint.pt"
        training_report_snapshot = snapshot_root / "training.json"
        semantic_snapshot = snapshot_root / "semantic.json"
        checkpoint_snapshot.write_bytes(checkpoint_source)
        training_report_snapshot.write_bytes(training_report_source)
        semantic_snapshot.write_bytes(semantic_source)
        try:
            checkpoint_id, training_evidence = _validate_candidate(
                config=config,
                checkpoint_path=checkpoint_snapshot,
                checkpoint_source=checkpoint_source,
                training_report_source=training_report_source,
                manifest=semantic_manifest,
                pairs=pairs,
                pair_digest=pair_digest,
                device=device,
            )
        except (CheckpointError, MujocoCLIError) as error:
            raise PairedSemanticPromptControlError(str(error)) from error

        trial_root = artifact_root / "semantic_benchmark_artifacts"
        benchmark_report_root = artifact_root / "semantic_benchmark_reports"
        condition_reports: list[dict[str, Any]] = []
        object_profile: Mapping[str, Any] | None = None
        object_profile_sha256: str | None = None
        model_identity: dict[str, object] | None = None
        model_identity_sources: set[str] = set()
        for condition in conditions:
            condition_trial_root = trial_root / condition.condition.value
            try:
                benchmark_report = run_task_spec_checkpoint_benchmark(
                    config,
                    manifest_path=semantic_snapshot,
                    artifact_dir=condition_trial_root,
                    checkpoint_path=checkpoint_snapshot,
                    training_report_path=training_report_snapshot,
                    task_spec=condition.prompt,
                    prompt_task=condition.task,
                    prompt_task_index=condition.task_index,
                    prompt_condition=condition.condition.value,
                    device=device,
                    prompt_task_expectation=condition.expectation,
                )
            except MujocoBenchmarkError as error:
                raise PairedSemanticPromptControlError(str(error)) from error
            metrics = _validate_nested_report(
                benchmark_report,
                condition=condition,
                config=config,
                manifest=semantic_manifest,
                semantic_manifest_sha256=semantic_manifest_sha256,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_id=checkpoint_id,
                training_report_sha256=training_report_sha256,
                training_evidence_sha256=training_evidence,
                artifact_dir=condition_trial_root,
            )
            observed_profile = benchmark_report.get("object_physical_profile")
            observed_profile_sha256 = benchmark_report.get(
                "object_physical_profile_sha256"
            )
            if not isinstance(observed_profile, Mapping) or (
                observed_profile_sha256 != canonical_json_sha256(observed_profile)
            ):
                raise PairedSemanticPromptControlError(
                    "benchmark object physical profile is invalid"
                )
            if object_profile_sha256 is None:
                object_profile = dict(observed_profile)
                object_profile_sha256 = str(observed_profile_sha256)
            elif observed_profile_sha256 != object_profile_sha256:
                raise PairedSemanticPromptControlError(
                    "object physical profile changed between conditions"
                )
            observed_model_identity = metrics["mujoco_model_identity"]
            if not isinstance(observed_model_identity, dict):
                raise PairedSemanticPromptControlError(
                    "benchmark model identity is invalid"
                )
            if model_identity is None:
                model_identity = observed_model_identity
            elif observed_model_identity != model_identity:
                raise PairedSemanticPromptControlError(
                    "model identity changed between conditions"
                )
            observed_sources = metrics["model_identity_sources"]
            if not isinstance(observed_sources, list):
                raise PairedSemanticPromptControlError(
                    "benchmark model identity sources are invalid"
                )
            for source in observed_sources:
                if not isinstance(source, str):
                    raise PairedSemanticPromptControlError(
                        "benchmark model identity source is invalid"
                    )
                model_identity_sources.add(source)

            benchmark_report_path = (
                benchmark_report_root / f"{condition.condition.value}.json"
            )
            write_benchmark_report(benchmark_report_path, benchmark_report)
            condition_reports.append(
                _condition_record(
                    condition,
                    benchmark_report_path=benchmark_report_path,
                    artifact_root=artifact_root,
                    benchmark_report=benchmark_report,
                    metrics=metrics,
                )
            )

    if model_identity is None:
        raise PairedSemanticPromptControlError("paired model identity is invalid")
    model_identity_source_list = sorted(model_identity_sources)
    matched = condition_reports[0]
    matched_success_rate = float(matched["success_rate"])
    matched_error = matched["object_position_error_mean_m"]
    matched_error_mean = None if matched_error is None else float(matched_error)
    matched_future_mse_value = matched["future_latent_mse_mean"]
    matched_future_mse = (
        None
        if matched_future_mse_value is None
        else float(matched_future_mse_value)
    )
    for condition_report in condition_reports:
        success_delta = (
            float(condition_report["success_rate"]) - matched_success_rate
        )
        condition_report["success_rate_delta_vs_matched"] = success_delta
        condition_error = condition_report["object_position_error_mean_m"]
        condition_report["object_position_error_mean_delta_vs_matched_m"] = (
            None
            if matched_error_mean is None or condition_error is None
            else float(condition_error) - matched_error_mean
        )
        condition_future_mse = condition_report["future_latent_mse_mean"]
        future_mse_delta = (
            None
            if matched_future_mse is None or condition_future_mse is None
            else float(condition_future_mse) - matched_future_mse
        )
        condition_report["future_latent_mse_delta_vs_matched"] = future_mse_delta
        condition_report["future_latent_proxy_outcome_mismatch"] = (
            future_mse_delta is not None
            and future_mse_delta < 0.0
            and success_delta < 0.0
        )
    total_trials = sum(int(item["trial_count"]) for item in condition_reports)
    scored_trials = sum(int(item["scored_trial_count"]) for item in condition_reports)
    execution_failures = sum(
        int(item["execution_failure_count"]) for item in condition_reports
    )
    joint_margins: dict[str, float | None] = {}
    for name in JOINT_LIMIT_MARGIN_FIELDS:
        values = [
            float(item["executability_joint_limit_margins"][name])
            for item in condition_reports
            if item["executability_joint_limit_margins"][name] is not None
        ]
        joint_margins[name] = min(values) if values else None
    contact_distances = [
        float(item["executability_minimum_contact_distance_m"])
        for item in condition_reports
        if item["executability_minimum_contact_distance_m"] is not None
    ]
    future_latent_elements = sum(
        int(item["future_latent_element_count"])
        for item in condition_reports
    )
    future_latent_squared_error = sum(
        float(item["future_latent_squared_error_sum"])
        for item in condition_reports
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_GATE,
        "result": "complete",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "scope": MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_SCOPE,
        "result_semantics": (
            "all paired human-video conditions completed against a "
            "manifest-declared object-state criterion"
        ),
        "condition_generator": CONDITION_GENERATOR,
        "future_latent_outcome_classifier": FUTURE_LATENT_OUTCOME_CLASSIFIER,
        "prompt_modality": "human_video_task_spec",
        "paired_human_video_checkpoint": True,
        "robot_used": False,
        "runtime_placeholder_prompt": "mujoco_home_hold",
        "runtime_placeholder_prompt_used_by_model": False,
        "executability_trace_present": True,
        "future_visual_metric_available": False,
        "future_visual_success_claimed": False,
        "runtime_future_latent_proxy_available": future_latent_elements > 0,
        "runtime_future_latent_success_claimed": False,
        "semantic_mujoco_object_state_evaluated": scored_trials > 0,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "validation_pair_manifest_sha256": sha256(pair_source).hexdigest(),
        "validation_pair_digest": pair_digest,
        "semantic_manifest_sha256": semantic_manifest_sha256,
        "training_report_sha256": training_report_sha256,
        "training_evidence_sha256": training_evidence,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "mujoco_config_sha256": project_config_sha256(config),
        "benchmark_id": semantic_manifest.benchmark_id,
        "dataset_task": semantic_manifest.dataset_task,
        "dataset_task_index": semantic_manifest.dataset_task_index,
        "object_body": semantic_manifest.heldout_tasks[0].object_body,
        "object_physical_profile": object_profile,
        "object_physical_profile_sha256": object_profile_sha256,
        "mujoco_model_identity": model_identity,
        "model_identity_sources": model_identity_source_list,
        "summary": {
            "condition_count": len(condition_reports),
            "trials_per_condition": len(semantic_manifest.heldout_tasks[0].seeds),
            "total_trial_count": total_trials,
            "scored_trial_count": scored_trials,
            "execution_failure_count": execution_failures,
            "executability_trace_trial_count": sum(
                int(item["executability_trace_trial_count"])
                for item in condition_reports
            ),
            "executability_policy_step_count": sum(
                int(item["executability_policy_step_count"])
                for item in condition_reports
            ),
            "executability_servo_step_count": sum(
                int(item["executability_servo_step_count"])
                for item in condition_reports
            ),
            "executability_sent_action_count": sum(
                int(item["executability_sent_action_count"])
                for item in condition_reports
            ),
            "executability_shadow_step_count": sum(
                int(item["executability_shadow_step_count"])
                for item in condition_reports
            ),
            "executability_safety_accepted_count": sum(
                int(item["executability_safety_accepted_count"])
                for item in condition_reports
            ),
            "executability_safety_rejected_count": sum(
                int(item["executability_safety_rejected_count"])
                for item in condition_reports
            ),
            "executability_safety_clipped_count": sum(
                int(item["executability_safety_clipped_count"])
                for item in condition_reports
            ),
            "executability_joint_limit_margins": joint_margins,
            "executability_contact_progression_count": sum(
                int(item["executability_contact_progression_count"])
                for item in condition_reports
            ),
            "executability_contact_observation_count": sum(
                int(item["executability_contact_observation_count"])
                for item in condition_reports
            ),
            "executability_forbidden_contact_observation_count": sum(
                int(item["executability_forbidden_contact_observation_count"])
                for item in condition_reports
            ),
            "executability_object_contact_observation_count": sum(
                int(item["executability_object_contact_observation_count"])
                for item in condition_reports
            ),
            "executability_minimum_contact_distance_m": (
                min(contact_distances) if contact_distances else None
            ),
            "executability_collision_failure_count": sum(
                int(item["executability_collision_failure_count"])
                for item in condition_reports
            ),
            "future_latent_trace_trial_count": sum(
                int(item["future_latent_trace_trial_count"])
                for item in condition_reports
            ),
            "future_latent_prediction_count": sum(
                int(item["future_latent_prediction_count"])
                for item in condition_reports
            ),
            "future_latent_observation_count": sum(
                int(item["future_latent_observation_count"])
                for item in condition_reports
            ),
            "future_latent_aligned_pair_count": sum(
                int(item["future_latent_aligned_pair_count"])
                for item in condition_reports
            ),
            "future_latent_censored_pair_count": sum(
                int(item["future_latent_censored_pair_count"])
                for item in condition_reports
            ),
            "future_latent_element_count": future_latent_elements,
            "future_latent_squared_error_sum": future_latent_squared_error,
            "future_latent_mse_mean": (
                None
                if not future_latent_elements
                else future_latent_squared_error / future_latent_elements
            ),
            "future_latent_proxy_outcome_mismatch_count": sum(
                int(item["future_latent_proxy_outcome_mismatch"])
                for item in condition_reports
            ),
            "mujoco_model_identity": model_identity,
            "model_identity_sources": model_identity_source_list,
            "matched_success_rate": matched_success_rate,
            "matched_object_position_error_mean_m": matched_error_mean,
        },
        "conditions": condition_reports,
        "limitations": [
            "human semantic matches are imported reviewer declarations",
            "the physical runtime prompt is a simulator-only context placeholder",
            "condition deltas are descriptive and do not prove prompt causality",
            "simulation evidence does not authorize real robot output",
        ],
    }
    write_benchmark_report(report_target, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired human-video prompt controls against a MuJoCo object-state task."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MUJOCO_CONFIG_PATH)
    parser.add_argument("--validation-pair-manifest", required=True, type=Path)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        report = run_paired_semantic_prompt_controls(
            ProjectConfig.load(args.config),
            validation_pair_manifest_path=args.validation_pair_manifest,
            semantic_manifest_path=args.semantic_manifest,
            checkpoint_path=args.checkpoint,
            training_report_path=args.training_report,
            artifact_dir=args.artifact_dir,
            report_path=args.report,
            device=args.device,
        )
    except (
        CheckpointError,
        ConfigError,
        ContractError,
        ModelContractError,
        MujocoAdapterError,
        MujocoBenchmarkError,
        MujocoCLIError,
        OSError,
        PairedDataError,
        PairedSemanticPromptControlError,
        PairedTaskSpecError,
        PolicyError,
        RolloutError,
        RuntimeErrorState,
        TaskSpecError,
        TensorizerError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "FUTURE_LATENT_OUTCOME_CLASSIFIER",
    "MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_GATE",
    "MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_SCOPE",
    "PairedSemanticPromptCondition",
    "PairedSemanticPromptControlError",
    "main",
    "run_paired_semantic_prompt_controls",
]


if __name__ == "__main__":
    raise SystemExit(main())
