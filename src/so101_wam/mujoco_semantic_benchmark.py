"""MuJoCo benchmark with a named-object position terminal criterion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .adapters.mujoco import MujocoAdapterError, MujocoCollisionError
from .checkpoint import (
    CheckpointError,
    load_compact_wam_bundle,
)
from .config import ConfigError, DEFAULT_MUJOCO_CONFIG_PATH, ProjectConfig
from .constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from .dataset import DatasetError, load_episode
from .deployment import canonical_json_sha256, file_sha256, project_config_sha256
from .mujoco_benchmark import (
    MujocoBenchmarkError,
    _failure_counts,
    _positive_float,
    _positive_int,
    _seed_tuple,
    _seeded_home_joint_position,
    _string_tuple,
    _wilson_interval,
    _write_evidence,
)
from .mujoco_cli import (
    MujocoCLIError,
    _checkpoint_identity,
    _require_mujoco_backend,
    inspect_mujoco_scene,
    run_mujoco_checkpoint_session,
    run_mujoco_task_spec_checkpoint_session,
)
from .mujoco_identity import MujocoIdentityError, MujocoModelIdentity
from .policy import PolicyError
from .rollout import (
    EXECUTABILITY_TRACE_SCHEMA,
    ExecutabilityTrace,
    JOINT_LIMIT_MARGIN_FIELDS,
    RolloutError,
    SafetyRejectedError,
    validate_executability_trace,
    validate_mujoco_scored_trace,
)
from .runtime import RuntimeErrorState
from .task_specs import TaskSpec, task_spec_kind
from .training import TRAINING_REPORT_SCHEMA


SCHEMA_VERSION = 5
MANIFEST_SCHEMA_VERSION = 3
TERMINAL_CRITERION = "object_body_position"
TERMINAL_FAILURE_REASON = "object_body_position_tolerance"
SEMANTIC_BENCHMARK_GATE = "G8-semantic-object-state-fixture"
SEMANTIC_BENCHMARK_POLICY = "injected_trial_runner"
SEMANTIC_BENCHMARK_SCOPE = "mujoco_object_state_report_contract_only"
CHECKPOINT_SEMANTIC_GATE = "G8-candidate-object-state"
CHECKPOINT_SEMANTIC_POLICY = "compact_wam"
CHECKPOINT_SEMANTIC_SCOPE = "candidate_object_state_checkpoint_split_verified"
TASK_SPEC_CHECKPOINT_SEMANTIC_GATE = "G8-candidate-task-spec-object-state"
TASK_SPEC_CHECKPOINT_SEMANTIC_POLICY = "compact_wam_task_spec"
TASK_SPEC_CHECKPOINT_SEMANTIC_SCOPE = (
    "candidate_object_state_task_spec_checkpoint_split_verified"
)
_MANIFEST_FIELDS = {
    "schema_version",
    "benchmark_id",
    "dataset_task",
    "train_task_ids",
    "heldout_tasks",
}
_DATASET_TASK_FIELDS = {"task_index", "task"}
_TASK_FIELDS = {
    "task_id",
    "label",
    "policy_steps",
    "seeds",
    "object_body",
    "initial_object_positions",
    "target_object_position",
    "position_tolerance_m",
}
_INITIAL_POSITION_FIELDS = {"seed", "position"}
RECOVERABLE_FAILURE_REASONS = frozenset(
    {
        "mujoco_collision",
        "mujoco_adapter_error",
        "policy_model_error",
        "safety_joint_limit",
        "safety_rejection",
        "safety_watchdog",
    }
)

MujocoSemanticBenchmarkError = MujocoBenchmarkError


class PromptTaskExpectation(StrEnum):
    MATCH_MANIFEST = "match_manifest"
    EXPECTED_MISMATCH = "expected_mismatch"


class SemanticTrialStatus(StrEnum):
    SCORED = "scored"
    EXECUTION_FAILURE = "execution_failure"


class ModelIdentitySource(StrEnum):
    ROLLOUT_SESSION = "rollout_session"
    POST_FAILURE_INSPECTION = "post_failure_inspection"


class _ModelIdentityRequirement(StrEnum):
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class SemanticTask:
    task_id: str
    label: str
    policy_steps: int
    seeds: tuple[int, ...]
    object_body: str
    initial_object_positions: tuple[tuple[int, tuple[float, float, float]], ...]
    target_object_position: tuple[float, float, float]
    position_tolerance_m: float

    def initial_position(self, seed: int) -> tuple[float, float, float]:
        for item_seed, position in self.initial_object_positions:
            if item_seed == seed:
                return position
        raise MujocoBenchmarkError(
            f"task {self.task_id!r} has no initial position for seed {seed}"
        )

    @property
    def initial_task_block_positions(
        self,
    ) -> tuple[tuple[int, tuple[float, float, float]], ...]:
        return self.initial_object_positions

    @property
    def target_task_block_position(self) -> tuple[float, float, float]:
        return self.target_object_position


@dataclass(frozen=True, slots=True)
class SemanticManifest:
    benchmark_id: str
    dataset_task: str
    dataset_task_index: int
    train_task_ids: tuple[str, ...]
    heldout_tasks: tuple[SemanticTask, ...]


@dataclass(frozen=True, slots=True)
class ObjectTerminalOutcome:
    success: bool
    object_position_error_m: float
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class SemanticTrialResult:
    success: bool
    object_position_error_m: float | None
    failure_reason: str | None
    rollout: Mapping[str, Any]
    initial_joint_position: tuple[float, ...]
    final_joint_position: tuple[float, ...] | None
    initial_object_position: tuple[float, float, float]
    final_object_position: tuple[float, float, float] | None
    status: SemanticTrialStatus = SemanticTrialStatus.SCORED
    failure_evidence: Mapping[str, Any] | None = None
    object_physical_profile: Mapping[str, Any] | None = None
    object_physical_profile_sha256: str | None = None
    executability_trace: Mapping[str, Any] | None = None
    mujoco_model_identity: Mapping[str, Any] | None = None
    model_identity_source: ModelIdentitySource | str | None = None

    @property
    def initial_task_block_position(self) -> tuple[float, float, float]:
        return self.initial_object_position

    @property
    def final_task_block_position(self) -> tuple[float, float, float] | None:
        return self.final_object_position


@dataclass(frozen=True, slots=True)
class _CheckpointEvidence:
    checkpoint_sha256: str
    checkpoint_id: str
    training_report_sha256: str
    training_evidence_sha256: str
    prompt_npz_sha256: str
    prompt_manifest_sha256: str
    prompt_fingerprint: str
    prompt_task: str
    prompt_task_index: int
    prompt_task_expectation: str
    prompt_task_identity_matches_manifest: bool
    prompt_task_match: str


@dataclass(frozen=True, slots=True)
class _TaskSpecCheckpointEvidence:
    checkpoint_sha256: str
    checkpoint_id: str
    training_report_sha256: str
    training_evidence_sha256: str
    task_spec_kind: str
    task_spec_fingerprint: str
    task_spec_source_id: str
    task_spec_source_sha256: str
    prompt_condition: str
    prompt_task: str
    prompt_task_index: int
    prompt_task_expectation: str
    prompt_task_identity_matches_manifest: bool
    prompt_task_match: str
    runtime_placeholder_prompt_used_by_model: bool = False


class TrialRunner(Protocol):
    def __call__(
        self,
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult: ...


def _json_object(source: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MujocoBenchmarkError(f"{name} has duplicate field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise MujocoBenchmarkError(
            f"{name} contains non-standard numeric constant: {value}"
        )

    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MujocoBenchmarkError(f"{name} must use UTF-8") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise MujocoBenchmarkError(f"{name} must be valid JSON") from error
    if not isinstance(payload, dict):
        raise MujocoBenchmarkError(f"{name} must be a JSON object")
    return payload


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    fields = set(value)
    missing = sorted(expected - fields)
    extra = sorted(fields - expected)
    if missing:
        raise MujocoBenchmarkError(f"{name} is missing fields: {missing}")
    if extra:
        raise MujocoBenchmarkError(f"{name} has unknown fields: {extra}")


def _source_bytes(path: str | Path, *, name: str) -> bytes:
    target = Path(path)
    try:
        return target.read_bytes()
    except OSError as error:
        raise MujocoBenchmarkError(f"failed to read {name}: {target}") from error


def _dataset_task(value: object) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise MujocoBenchmarkError("dataset_task must be an object")
    _exact_fields(value, _DATASET_TASK_FIELDS, name="dataset_task")
    task = value.get("task")
    task_index = value.get("task_index")
    if not isinstance(task, str) or not task.strip():
        raise MujocoBenchmarkError("dataset_task.task must be a non-empty string")
    if (
        not isinstance(task_index, int)
        or isinstance(task_index, bool)
        or task_index < 0
    ):
        raise MujocoBenchmarkError("dataset_task.task_index must be non-negative")
    return task, task_index


def _numeric_tuple(
    value: object,
    *,
    length: int,
    length_label: str,
    name: str,
) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (bytes, str))
        or len(value) != length
    ):
        raise MujocoBenchmarkError(f"{name} must contain {length_label} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise MujocoBenchmarkError(f"{name} must be numeric") from error
    if not all(isfinite(item) for item in result):
        raise MujocoBenchmarkError(f"{name} must be finite")
    return result


def _position(value: object, *, name: str) -> tuple[float, float, float]:
    result = _numeric_tuple(value, length=3, length_label="three", name=name)
    return result[0], result[1], result[2]


def _initial_positions(
    value: object,
    *,
    seeds: tuple[int, ...],
) -> tuple[tuple[int, tuple[float, float, float]], ...]:
    if not isinstance(value, list):
        raise MujocoBenchmarkError("initial_object_positions must be a list")
    by_seed: dict[int, tuple[float, float, float]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise MujocoBenchmarkError(
                "initial_object_positions entries must be objects"
            )
        _exact_fields(
            item,
            _INITIAL_POSITION_FIELDS,
            name=f"initial_object_positions[{index}]",
        )
        seed = item.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise MujocoBenchmarkError("initial object seed must be non-negative")
        if seed in by_seed:
            raise MujocoBenchmarkError(f"initial object seed is duplicated: {seed}")
        by_seed[seed] = _position(
            item.get("position"),
            name="initial object position",
        )
    if set(by_seed) != set(seeds):
        raise MujocoBenchmarkError(
            "initial_object_positions requires exactly one position per task seed"
        )
    return tuple((seed, by_seed[seed]) for seed in seeds)


def _task(value: object) -> SemanticTask:
    if not isinstance(value, dict):
        raise MujocoBenchmarkError("heldout_tasks entries must be objects")
    _exact_fields(value, _TASK_FIELDS, name="heldout task")
    task_id = value.get("task_id")
    label = value.get("label")
    if not isinstance(task_id, str) or not task_id.strip():
        raise MujocoBenchmarkError("held-out semantic task requires task_id")
    if not isinstance(label, str) or not label.strip():
        raise MujocoBenchmarkError("held-out semantic task requires label")
    if not task_id.replace("_", "").replace("-", "").isalnum():
        raise MujocoBenchmarkError("held-out semantic task_id is artifact-unsafe")
    object_body = value.get("object_body")
    if not isinstance(object_body, str) or not object_body.strip():
        raise MujocoBenchmarkError("held-out semantic task requires object_body")
    seeds = _seed_tuple(value.get("seeds"))
    return SemanticTask(
        task_id=task_id.strip(),
        label=label.strip(),
        policy_steps=_positive_int(value.get("policy_steps"), name="policy_steps"),
        object_body=object_body.strip(),
        seeds=seeds,
        initial_object_positions=_initial_positions(
            value.get("initial_object_positions"),
            seeds=seeds,
        ),
        target_object_position=_position(
            value.get("target_object_position"),
            name="target object position",
        ),
        position_tolerance_m=_positive_float(
            value.get("position_tolerance_m"),
            name="position_tolerance_m",
        ),
    )


def load_semantic_manifest_bytes(source: bytes) -> SemanticManifest:
    payload = _json_object(source, name="semantic manifest")
    _exact_fields(payload, _MANIFEST_FIELDS, name="semantic manifest")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise MujocoBenchmarkError(
            f"semantic benchmark requires schema_version={MANIFEST_SCHEMA_VERSION}"
        )
    benchmark_id = payload.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise MujocoBenchmarkError("semantic benchmark requires benchmark_id")
    dataset_task, dataset_task_index = _dataset_task(payload.get("dataset_task"))
    train_task_ids = _string_tuple(payload.get("train_task_ids"), name="train_task_ids")
    tasks_value = payload.get("heldout_tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise MujocoBenchmarkError("heldout_tasks must be a non-empty list")
    tasks = tuple(_task(item) for item in tasks_value)
    heldout_ids = tuple(task.task_id for task in tasks)
    if len(set(heldout_ids)) != len(heldout_ids):
        raise MujocoBenchmarkError("held-out semantic task ids must be unique")
    if set(train_task_ids).intersection(heldout_ids):
        raise MujocoBenchmarkError(
            "train and held-out semantic task ids must be disjoint"
        )
    return SemanticManifest(
        benchmark_id=benchmark_id.strip(),
        dataset_task=dataset_task,
        dataset_task_index=dataset_task_index,
        train_task_ids=train_task_ids,
        heldout_tasks=tasks,
    )


def load_semantic_manifest(path: str | Path) -> SemanticManifest:
    return load_semantic_manifest_bytes(_source_bytes(path, name="semantic manifest"))


def score_object_position(
    task: SemanticTask,
    final_object_position: Sequence[float],
) -> ObjectTerminalOutcome:
    final = np.asarray(final_object_position, dtype=np.float64)
    if final.shape != (3,):
        raise MujocoBenchmarkError(
            "final object position must contain three values"
        )
    if not np.isfinite(final).all():
        raise MujocoBenchmarkError("final object position must be finite")
    target = np.asarray(task.target_object_position, dtype=np.float64)
    error = float(np.linalg.norm(final - target))
    success = error <= task.position_tolerance_m
    return ObjectTerminalOutcome(
        success,
        error,
        None if success else TERMINAL_FAILURE_REASON,
    )


def _vector(value: object, *, length: int, name: str) -> tuple[float, ...]:
    return _numeric_tuple(value, length=length, length_label=str(length), name=name)


def _safety_reason(reasons: Sequence[str]) -> str:
    if "stale_action" in reasons:
        return "safety_watchdog"
    if any("_joint_limit:" in reason for reason in reasons):
        return "safety_joint_limit"
    return "safety_rejection"


def _execution_failure(
    error: MujocoAdapterError | SafetyRejectedError | PolicyError,
) -> tuple[str, dict[str, Any]]:
    if isinstance(error, MujocoCollisionError):
        return "mujoco_collision", {
            "error_type": type(error).__name__,
            "phase": error.phase,
            "contacts": [asdict(contact) for contact in error.contacts],
        }
    if isinstance(error, MujocoAdapterError):
        return "mujoco_adapter_error", {
            "error_type": type(error).__name__,
            "message": str(error),
        }
    if isinstance(error, SafetyRejectedError):
        return _safety_reason(error.reasons), {
            "error_type": type(error).__name__,
            "reasons": list(error.reasons),
        }
    return "policy_model_error", {
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _record_collision_trace(
    trace: ExecutabilityTrace,
    error: MujocoAdapterError | SafetyRejectedError | PolicyError,
) -> None:
    if not isinstance(error, MujocoCollisionError):
        return
    trace.on_collision_failure(
        phase=error.phase,
        contacts=tuple(asdict(contact) for contact in error.contacts),
    )


def _string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MujocoBenchmarkError(f"{name} must be a list")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise MujocoBenchmarkError(f"{name} must contain non-empty strings")
    return result


def validate_failure_evidence(reason: str, evidence: object) -> None:
    if reason not in RECOVERABLE_FAILURE_REASONS:
        raise MujocoBenchmarkError(f"unknown recoverable failure reason: {reason}")
    if not isinstance(evidence, Mapping):
        raise MujocoBenchmarkError("recoverable failure evidence must be an object")

    if reason == "mujoco_collision":
        if set(evidence) != {"error_type", "phase", "contacts"}:
            raise MujocoBenchmarkError("collision failure evidence fields are invalid")
        if evidence.get("error_type") != "MujocoCollisionError":
            raise MujocoBenchmarkError("collision failure evidence type is invalid")
        phase = evidence.get("phase")
        contacts = evidence.get("contacts")
        if not isinstance(phase, str) or not phase:
            raise MujocoBenchmarkError("collision failure phase is invalid")
        if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
            raise MujocoBenchmarkError("collision failure contacts must be a list")
        fields = {
            "geom1",
            "geom2",
            "body1",
            "body2",
            "category1",
            "category2",
            "distance",
        }
        for contact in contacts:
            if not isinstance(contact, Mapping) or set(contact) != fields:
                raise MujocoBenchmarkError("collision contact evidence is invalid")
            for field in fields - {"distance"}:
                if not isinstance(contact.get(field), str):
                    raise MujocoBenchmarkError("collision contact identity is invalid")
            distance = contact.get("distance")
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not isfinite(float(distance))
            ):
                raise MujocoBenchmarkError("collision contact distance is invalid")
        return

    if reason == "mujoco_adapter_error":
        if set(evidence) != {"error_type", "message"}:
            raise MujocoBenchmarkError("adapter failure evidence fields are invalid")
        if evidence.get("error_type") != "MujocoAdapterError":
            raise MujocoBenchmarkError("adapter failure evidence type is invalid")
        message = evidence.get("message")
        if not isinstance(message, str) or not message:
            raise MujocoBenchmarkError("adapter failure message is invalid")
        return

    if reason.startswith("safety_"):
        if set(evidence) != {"error_type", "reasons"}:
            raise MujocoBenchmarkError("safety failure evidence fields are invalid")
        if evidence.get("error_type") != "SafetyRejectedError":
            raise MujocoBenchmarkError("safety failure evidence type is invalid")
        reasons = _string_list(evidence.get("reasons"), name="safety failure reasons")
        if _safety_reason(reasons) != reason:
            raise MujocoBenchmarkError("safety failure category disagrees with reasons")
        return

    if set(evidence) != {"error_type", "message"}:
        raise MujocoBenchmarkError("policy failure evidence fields are invalid")
    if evidence.get("error_type") != "PolicyError":
        raise MujocoBenchmarkError("policy failure evidence type is invalid")
    message = evidence.get("message")
    if not isinstance(message, str) or not message:
        raise MujocoBenchmarkError("policy failure message is invalid")


def run_checkpoint_task_trial(
    config: ProjectConfig,
    task: SemanticTask,
    *,
    seed: int,
    checkpoint_path: str | Path,
    prompt_path: str | Path,
    prompt_manifest_path: str | Path | None = None,
    device: str = "cpu",
) -> SemanticTrialResult:
    _require_mujoco_backend(config)
    initial_joints = _seeded_home_joint_position(config, seed=seed)
    initial_object = task.initial_position(seed)
    trial_config = replace(
        config,
        mujoco=replace(config.mujoco, home_joint_position=initial_joints),
    )
    trace = ExecutabilityTrace(
        _trial_id(task, seed),
        joint_lower=config.safety.joint_lower,
        joint_upper=config.safety.joint_upper,
        observation_joint_lower=config.safety.observation_joint_lower,
        observation_joint_upper=config.safety.observation_joint_upper,
    )
    try:
        report = run_mujoco_checkpoint_session(
            trial_config,
            checkpoint_path=checkpoint_path,
            prompt_path=prompt_path,
            manifest_path=prompt_manifest_path,
            policy_steps=task.policy_steps,
            device=device,
            rollout_observer=trace,
            semantic_object_body=task.object_body,
            initial_object_position=initial_object,
        )
    except (MujocoAdapterError, SafetyRejectedError, PolicyError) as error:
        _record_collision_trace(trace, error)
        failure_reason, failure_evidence = _execution_failure(error)
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason=failure_reason,
            rollout={"execution_status": "aborted"},
            initial_joint_position=initial_joints,
            final_joint_position=None,
            initial_object_position=initial_object,
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence=failure_evidence,
            executability_trace=trace.snapshot(),
        )
    rollout = report.get("rollout")
    if not isinstance(rollout, Mapping):
        raise MujocoBenchmarkError("checkpoint session report requires rollout object")
    final_joints = _vector(
        report.get("terminal_joint_position"),
        length=ACTION_DIM,
        name="terminal joint position",
    )
    if report.get("object_body") != task.object_body:
        raise MujocoBenchmarkError("checkpoint session object body disagrees with task")
    model_identity = _model_identity_payload(report.get("mujoco_model_identity"))
    object_profile = report.get("object_physical_profile")
    object_profile_sha256 = report.get("object_physical_profile_sha256")
    if not isinstance(object_profile, Mapping):
        raise MujocoBenchmarkError("checkpoint session requires object physical profile")
    if object_profile_sha256 != canonical_json_sha256(object_profile):
        raise MujocoBenchmarkError("checkpoint session object profile SHA-256 is invalid")

    final_object_values = _vector(
        report.get("terminal_object_position"),
        length=3,
        name="terminal object position",
    )
    final_object = (
        final_object_values[0],
        final_object_values[1],
        final_object_values[2],
    )
    outcome = score_object_position(task, final_object)
    return SemanticTrialResult(
        outcome.success,
        outcome.object_position_error_m,
        outcome.failure_reason,
        dict(rollout),
        initial_joints,
        final_joints,
        initial_object,
        final_object,
        object_physical_profile=dict(object_profile),
        object_physical_profile_sha256=str(object_profile_sha256),
        executability_trace=trace.snapshot(),
        mujoco_model_identity=model_identity,
        model_identity_source=ModelIdentitySource.ROLLOUT_SESSION,
    )


def run_task_spec_checkpoint_task_trial(
    config: ProjectConfig,
    task: SemanticTask,
    *,
    seed: int,
    checkpoint_path: str | Path,
    task_spec: TaskSpec,
    device: str = "cpu",
) -> SemanticTrialResult:
    """Run one semantic trial with an action-free task specification."""

    _require_mujoco_backend(config)
    initial_joints = _seeded_home_joint_position(config, seed=seed)
    initial_object = task.initial_position(seed)
    trial_config = replace(
        config,
        mujoco=replace(config.mujoco, home_joint_position=initial_joints),
    )
    trace = ExecutabilityTrace(
        _trial_id(task, seed),
        joint_lower=config.safety.joint_lower,
        joint_upper=config.safety.joint_upper,
        observation_joint_lower=config.safety.observation_joint_lower,
        observation_joint_upper=config.safety.observation_joint_upper,
    )
    try:
        report = run_mujoco_task_spec_checkpoint_session(
            trial_config,
            checkpoint_path=checkpoint_path,
            task_spec=task_spec,
            policy_steps=task.policy_steps,
            device=device,
            rollout_observer=trace,
            semantic_object_body=task.object_body,
            initial_object_position=initial_object,
        )
    except (MujocoAdapterError, SafetyRejectedError, PolicyError) as error:
        _record_collision_trace(trace, error)
        failure_reason, failure_evidence = _execution_failure(error)
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason=failure_reason,
            rollout={"execution_status": "aborted"},
            initial_joint_position=initial_joints,
            final_joint_position=None,
            initial_object_position=initial_object,
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence=failure_evidence,
            executability_trace=trace.snapshot(),
        )
    if report.get("runtime_placeholder_prompt_used_by_model") is not False:
        raise MujocoBenchmarkError(
            "task-spec session must exclude its runtime placeholder prompt"
        )
    if report.get("task_spec_fingerprint") != task_spec.fingerprint:
        raise MujocoBenchmarkError("task-spec session fingerprint mismatch")

    rollout = report.get("rollout")
    if not isinstance(rollout, Mapping):
        raise MujocoBenchmarkError("task-spec session report requires rollout object")
    final_joints = _vector(
        report.get("terminal_joint_position"),
        length=ACTION_DIM,
        name="terminal joint position",
    )
    if report.get("object_body") != task.object_body:
        raise MujocoBenchmarkError("task-spec session object body disagrees with task")
    model_identity = _model_identity_payload(report.get("mujoco_model_identity"))
    object_profile = report.get("object_physical_profile")
    object_profile_sha256 = report.get("object_physical_profile_sha256")
    if not isinstance(object_profile, Mapping):
        raise MujocoBenchmarkError("task-spec session requires object physical profile")
    if object_profile_sha256 != canonical_json_sha256(object_profile):
        raise MujocoBenchmarkError(
            "task-spec session object profile SHA-256 is invalid"
        )

    final_object_values = _vector(
        report.get("terminal_object_position"),
        length=3,
        name="terminal object position",
    )
    final_object = (
        final_object_values[0],
        final_object_values[1],
        final_object_values[2],
    )
    outcome = score_object_position(task, final_object)
    return SemanticTrialResult(
        outcome.success,
        outcome.object_position_error_m,
        outcome.failure_reason,
        dict(rollout),
        initial_joints,
        final_joints,
        initial_object,
        final_object,
        object_physical_profile=dict(object_profile),
        object_physical_profile_sha256=str(object_profile_sha256),
        executability_trace=trace.snapshot(),
        mujoco_model_identity=model_identity,
        model_identity_source=ModelIdentitySource.ROLLOUT_SESSION,
    )


def _matches_expected(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    return actual == expected


def _require_markers(
    value: Mapping[str, Any],
    expected: Mapping[str, object],
    *,
    name: str,
) -> None:
    for key, expected_value in expected.items():
        if not _matches_expected(value.get(key), expected_value):
            raise MujocoBenchmarkError(f"{name} requires {key}={expected_value!r}")


def _task_inventory(
    data: Mapping[str, Any],
    *,
    field: str,
    count_field: str,
    name: str,
) -> tuple[tuple[int, str], ...]:
    value = data.get(field)
    if not isinstance(value, list) or not value:
        raise MujocoBenchmarkError(f"{name} must be a non-empty list")

    identities: list[tuple[int, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise MujocoBenchmarkError(f"{name}[{index}] must be an object")
        _exact_fields(item, _DATASET_TASK_FIELDS, name=f"{name}[{index}]")
        task_index = item.get("task_index")
        task = item.get("task")
        if (
            not isinstance(task_index, int)
            or isinstance(task_index, bool)
            or task_index < 0
        ):
            raise MujocoBenchmarkError(
                f"{name}[{index}].task_index must be non-negative"
            )
        if not isinstance(task, str) or not task.strip():
            raise MujocoBenchmarkError(
                f"{name}[{index}].task must be a non-empty string"
            )
        identities.append((task_index, task))

    count = data.get(count_field)
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(identities)
    ):
        raise MujocoBenchmarkError(f"{name} count does not match {count_field}")
    if identities != sorted(set(identities)):
        raise MujocoBenchmarkError(f"{name} must be sorted and unique")
    return tuple(identities)


def _require_task_split(
    report: Mapping[str, Any],
    manifest: SemanticManifest,
) -> None:
    data = report.get("data")
    if not isinstance(data, Mapping):
        raise MujocoBenchmarkError("training report requires a data object")
    train = _task_inventory(
        data,
        field="train_tasks",
        count_field="train_task_count",
        name="checkpoint train task inventory",
    )
    validation = _task_inventory(
        data,
        field="validation_tasks",
        count_field="validation_task_count",
        name="checkpoint validation task inventory",
    )

    combined = (*train, *validation)
    by_index: dict[int, str] = {}
    by_task: dict[str, int] = {}
    for task_index, task in combined:
        if task_index in by_index and by_index[task_index] != task:
            raise MujocoBenchmarkError(
                "checkpoint task inventory maps one index to multiple labels"
            )
        if task in by_task and by_task[task] != task_index:
            raise MujocoBenchmarkError(
                "checkpoint task inventory maps one label to multiple indices"
            )
        by_index[task_index] = task
        by_task[task] = task_index

    train_indices = {task_index for task_index, _ in train}
    train_labels = {task for _, task in train}
    validation_indices = {task_index for task_index, _ in validation}
    validation_labels = {task for _, task in validation}
    if train_indices & validation_indices or train_labels & validation_labels:
        raise MujocoBenchmarkError(
            "checkpoint train and validation task inventories must be disjoint"
        )

    identity = (manifest.dataset_task_index, manifest.dataset_task)
    if (
        manifest.dataset_task_index in train_indices
        or manifest.dataset_task in train_labels
    ):
        raise MujocoBenchmarkError(
            "manifest dataset task appears in the checkpoint train task inventory"
        )
    if identity not in validation:
        raise MujocoBenchmarkError(
            "manifest dataset task is absent from the checkpoint validation task inventory"
        )


def _validate_training_report(
    source: bytes,
    *,
    checkpoint_metadata: Mapping[str, Any],
    checkpoint_architecture: Mapping[str, int | str],
    checkpoint_id: str,
    checkpoint_sha256: str,
    manifest: SemanticManifest,
) -> str:
    report = _json_object(source, name="training report")
    offline_markers = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
    }
    _require_markers(
        checkpoint_metadata,
        offline_markers,
        name="checkpoint metadata",
    )
    _require_markers(
        report,
        {
            "schema_version": TRAINING_REPORT_SCHEMA,
            "result": "pass",
            **offline_markers,
        },
        name="training report",
    )
    if report.get("checkpoint_id") != checkpoint_id:
        raise MujocoBenchmarkError(
            "training report checkpoint_id does not match checkpoint metadata"
        )

    training_evidence = report.get("training_evidence_sha256")
    if (
        not isinstance(training_evidence, str)
        or len(training_evidence) != 64
        or any(character not in "0123456789abcdef" for character in training_evidence)
    ):
        raise MujocoBenchmarkError(
            "training report training_evidence_sha256 must be lowercase SHA-256"
        )
    if checkpoint_metadata.get("training_evidence_sha256") != training_evidence:
        raise MujocoBenchmarkError(
            "training report digest does not match checkpoint metadata"
        )
    report_core = {
        key: value
        for key, value in report.items()
        if key not in {"training_evidence_sha256", "artifacts"}
    }
    if canonical_json_sha256(report_core) != training_evidence:
        raise MujocoBenchmarkError("training report core digest is invalid")

    artifacts = report.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise MujocoBenchmarkError(
            "training report does not bind the exact checkpoint bytes"
        )
    if report.get("model") != dict(checkpoint_architecture):
        raise MujocoBenchmarkError(
            "training report architecture does not match the checkpoint"
        )
    protocol = report.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("split") != "task_disjoint"
        or protocol.get("real_output_authorized") is not False
    ):
        raise MujocoBenchmarkError(
            "training report protocol is not task-disjoint and offline-only"
        )

    _require_task_split(report, manifest)
    return training_evidence


def _prompt_task_status(
    *,
    task: str,
    task_index: int,
    manifest: SemanticManifest,
    expectation: PromptTaskExpectation,
) -> tuple[bool, str]:
    if not isinstance(expectation, PromptTaskExpectation):
        raise MujocoBenchmarkError("prompt task expectation is invalid")
    task_matches = task == manifest.dataset_task
    index_matches = task_index == manifest.dataset_task_index
    identity_matches = task_matches and index_matches
    if expectation is PromptTaskExpectation.MATCH_MANIFEST:
        if not identity_matches:
            raise MujocoBenchmarkError(
                "prompt task identity does not match manifest dataset_task"
            )
        return True, "dataset_identity_verified"
    if task_matches or index_matches:
        raise MujocoBenchmarkError(
            "expected control prompt task identity must differ in label and index"
        )
    return False, "control_mismatch_verified"


def _claims() -> dict[str, bool]:
    return {
        "robot_used": False,
        "future_visual_metric_available": False,
        "future_visual_success_claimed": False,
        "runtime_future_latent_proxy_available": False,
        "runtime_future_latent_success_claimed": False,
        "semantic_mujoco_object_state_evaluated": True,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
    }


def _trial_id(task: SemanticTask, seed: int) -> str:
    return f"{task.task_id}:seed:{seed}"


def _validated_trace(
    result: SemanticTrialResult,
    *,
    rollout_id: str,
) -> dict[str, Any] | None:
    if result.executability_trace is None:
        return None
    try:
        return validate_executability_trace(
            result.executability_trace,
            rollout_id=rollout_id,
        )
    except RolloutError as error:
        raise MujocoBenchmarkError(str(error)) from error


def _model_identity_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MujocoBenchmarkError("semantic trial requires MuJoCo model identity")
    try:
        return MujocoModelIdentity.from_payload(value).to_payload()
    except MujocoIdentityError as error:
        raise MujocoBenchmarkError(f"invalid MuJoCo model identity: {error}") from error


def _identity_source(value: object) -> ModelIdentitySource | None:
    if value is None:
        return None
    try:
        return ModelIdentitySource(str(value))
    except ValueError as error:
        raise MujocoBenchmarkError("semantic trial model identity source is invalid") from error


def _trial_identity(
    result: SemanticTrialResult,
    *,
    requirement: _ModelIdentityRequirement,
) -> tuple[dict[str, object] | None, ModelIdentitySource | None]:
    source = _identity_source(result.model_identity_source)
    if result.mujoco_model_identity is None:
        if source is not None:
            raise MujocoBenchmarkError("model identity source requires model identity")
        if requirement is _ModelIdentityRequirement.REQUIRED:
            raise MujocoBenchmarkError("semantic trial requires MuJoCo model identity")
        return None, None

    identity = _model_identity_payload(result.mujoco_model_identity)
    if source is None:
        raise MujocoBenchmarkError("MuJoCo model identity requires source")
    if result.status is SemanticTrialStatus.SCORED and source is not (
        ModelIdentitySource.ROLLOUT_SESSION
    ):
        raise MujocoBenchmarkError("scored trial model identity must come from rollout")
    return identity, source


def _failure_scene_needed(result: SemanticTrialResult) -> bool:
    _trial_identity(result, requirement=_ModelIdentityRequirement.OPTIONAL)
    if result.object_physical_profile is None:
        return True
    if result.mujoco_model_identity is None:
        return True
    return False


def _with_failure_scene(
    result: SemanticTrialResult,
    scene: tuple[dict[str, object], str, dict[str, object]],
) -> SemanticTrialResult:
    profile, profile_sha256, scene_identity = scene
    trial_identity, trial_source = _trial_identity(
        result,
        requirement=_ModelIdentityRequirement.OPTIONAL,
    )
    scene_identity = _model_identity_payload(scene_identity)
    if trial_identity is not None and trial_identity != scene_identity:
        raise MujocoBenchmarkError("inspected model identity disagrees with trial")

    if (
        result.object_physical_profile is not None
        and result.object_physical_profile != profile
    ):
        raise MujocoBenchmarkError("inspected object profile disagrees with trial")
    if (
        result.object_physical_profile_sha256 is not None
        and result.object_physical_profile_sha256 != profile_sha256
    ):
        raise MujocoBenchmarkError("inspected object profile hash disagrees with trial")

    identity = trial_identity if trial_identity is not None else scene_identity
    identity_source = (
        trial_source
        if trial_source is not None
        else ModelIdentitySource.POST_FAILURE_INSPECTION
    )

    return replace(
        result,
        object_physical_profile=(
            profile if result.object_physical_profile is None else result.object_physical_profile
        ),
        object_physical_profile_sha256=(
            profile_sha256
            if result.object_physical_profile_sha256 is None
            else result.object_physical_profile_sha256
        ),
        mujoco_model_identity=identity,
        model_identity_source=identity_source,
    )


def _validate_trial(
    task: SemanticTask,
    seed: int,
    result: SemanticTrialResult,
    *,
    identity_requirement: _ModelIdentityRequirement = _ModelIdentityRequirement.OPTIONAL,
) -> tuple[dict[str, object] | None, ModelIdentitySource | None]:
    if result.initial_task_block_position != task.initial_position(seed):
        raise MujocoBenchmarkError(
            "semantic trial initial object position disagrees with manifest"
        )
    _vector(result.initial_joint_position, length=ACTION_DIM, name="initial joints")
    if not isinstance(result.rollout, Mapping):
        raise MujocoBenchmarkError("semantic trial rollout must be an object")
    trace = _validated_trace(result, rollout_id=_trial_id(task, seed))
    if result.object_physical_profile is None:
        if result.object_physical_profile_sha256 is not None:
            raise MujocoBenchmarkError("object profile hash requires object profile")
    elif (
        result.object_physical_profile_sha256
        != canonical_json_sha256(result.object_physical_profile)
    ):
        raise MujocoBenchmarkError("object profile SHA-256 disagrees with profile")
    model_identity = _trial_identity(result, requirement=identity_requirement)

    if result.status is SemanticTrialStatus.EXECUTION_FAILURE:
        if (
            result.success
            or result.object_position_error_m is not None
            or result.final_joint_position is not None
            or result.final_task_block_position is not None
            or result.failure_reason is None
        ):
            raise MujocoBenchmarkError(
                "recoverable failure cannot contain terminal outcome evidence"
            )
        validate_failure_evidence(result.failure_reason, result.failure_evidence)
        return model_identity

    if result.status is not SemanticTrialStatus.SCORED:
        raise MujocoBenchmarkError("semantic trial status is invalid")
    if trace is None:
        raise MujocoBenchmarkError("scored trial requires complete executability trace")
    try:
        validate_mujoco_scored_trace(trace, policy_steps=task.policy_steps)
    except RolloutError as error:
        raise MujocoBenchmarkError(
            f"scored trial requires complete executability trace: {error}"
        ) from error
    if (
        result.object_position_error_m is None
        or result.final_joint_position is None
        or result.final_task_block_position is None
        or result.failure_evidence is not None
    ):
        raise MujocoBenchmarkError("scored trial requires terminal outcome evidence")

    expected = score_object_position(task, result.final_task_block_position)
    if (
        result.success is not expected.success
        or result.failure_reason != expected.failure_reason
        or not isclose(
            result.object_position_error_m,
            expected.object_position_error_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise MujocoBenchmarkError(
            "semantic trial result disagrees with object terminal criterion"
        )
    _vector(result.final_joint_position, length=ACTION_DIM, name="final joints")
    return model_identity


def _summaries(
    manifest: SemanticManifest,
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for task in manifest.heldout_tasks:
        task_trials = [trial for trial in trials if trial["task_id"] == task.task_id]
        successes, count, scored, errors = _trial_totals(task_trials)
        error_mean, error_max = _error_metrics(errors)
        summaries.append(
            {
                "task_id": task.task_id,
                "label": task.label,
                "object_body": task.object_body,
                "object_physical_profile_sha256": _single_value(
                    task_trials,
                    "object_physical_profile_sha256",
                ),
                "mujoco_model_identity": _single_model_identity(task_trials),
                "model_identity_sources": _identity_sources(task_trials),
                "trial_count": count,
                "scored_trial_count": scored,
                "execution_failure_count": count - scored,
                "success_count": successes,
                "success_rate": successes / count,
                "success_rate_95ci": list(_wilson_interval(successes, count)),
                "object_position_error_mean_m": error_mean,
                "object_position_error_max_m": error_max,
                "failure_counts": _failure_counts(task_trials),
            }
        )
    return summaries


def _single_value(
    values: Sequence[Mapping[str, Any]],
    field: str,
) -> object:
    unique = {item.get(field) for item in values}
    if len(unique) == 1:
        return next(iter(unique))
    return None


def _single_model_identity(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, object] | None:
    identities = [
        item.get("mujoco_model_identity")
        for item in trials
        if item.get("mujoco_model_identity") is not None
    ]
    if not identities:
        return None

    first = identities[0]
    if not isinstance(first, Mapping):
        raise MujocoBenchmarkError("MuJoCo model identity is invalid")
    payload = MujocoModelIdentity.from_payload(first).to_payload()
    if any(
        not isinstance(identity, Mapping)
        or MujocoModelIdentity.from_payload(identity).to_payload() != payload
        for identity in identities[1:]
    ):
        raise MujocoBenchmarkError("mixed MuJoCo model identity across trials")
    return payload


def _identity_sources(
    trials: Sequence[Mapping[str, Any]],
) -> list[str]:
    return sorted(
        {
            str(source)
            for trial in trials
            if (source := trial.get("model_identity_source")) is not None
        }
    )


def _trial_totals(
    trials: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, tuple[float, ...]]:
    successes = sum(int(trial["success"]) for trial in trials)
    errors = tuple(
        float(trial["object_position_error_m"])
        for trial in trials
        if trial["status"] == SemanticTrialStatus.SCORED.value
    )
    return successes, len(trials), len(errors), errors


def _error_metrics(errors: Sequence[float]) -> tuple[float | None, float | None]:
    if not errors:
        return None, None
    return fmean(errors), max(errors)


def _run_benchmark(
    config: ProjectConfig,
    *,
    manifest: SemanticManifest,
    manifest_sha256: str,
    artifact_dir: str | Path,
    trial_runner: TrialRunner,
    checkpoint_evidence: _CheckpointEvidence | None = None,
    task_spec_checkpoint_evidence: _TaskSpecCheckpointEvidence | None = None,
) -> dict[str, Any]:
    _require_mujoco_backend(config)
    if checkpoint_evidence is not None and task_spec_checkpoint_evidence is not None:
        raise MujocoBenchmarkError("benchmark evidence modes are exclusive")
    artifacts = Path(artifact_dir)
    paths = {
        (task.task_id, seed): artifacts / f"{task.task_id}-seed-{seed}.json"
        for task in manifest.heldout_tasks
        for seed in task.seeds
    }
    trace_paths = {
        identity: path.with_suffix(".executability.json")
        for identity, path in paths.items()
    }
    all_paths = (*paths.values(), *trace_paths.values())
    existing = next((path for path in all_paths if path.exists()), None)
    if existing is not None:
        raise MujocoBenchmarkError(
            f"MuJoCo semantic benchmark artifact already exists: {existing}"
        )

    gate = SEMANTIC_BENCHMARK_GATE
    policy = SEMANTIC_BENCHMARK_POLICY
    scope = SEMANTIC_BENCHMARK_SCOPE
    evidence: dict[str, Any] = {}
    if checkpoint_evidence is not None:
        gate = CHECKPOINT_SEMANTIC_GATE
        policy = CHECKPOINT_SEMANTIC_POLICY
        scope = CHECKPOINT_SEMANTIC_SCOPE
        evidence = {
            **asdict(checkpoint_evidence),
            "task_disjoint_basis": "checkpoint_bound_task_inventory",
            "checkpoint_task_split": "training_report_verified",
            "semantic_task_mapping": "manifest_declared",
        }
    if task_spec_checkpoint_evidence is not None:
        gate = TASK_SPEC_CHECKPOINT_SEMANTIC_GATE
        policy = TASK_SPEC_CHECKPOINT_SEMANTIC_POLICY
        scope = TASK_SPEC_CHECKPOINT_SEMANTIC_SCOPE
        evidence = {
            **asdict(task_spec_checkpoint_evidence),
            "task_disjoint_basis": "checkpoint_bound_task_inventory",
            "checkpoint_task_split": "training_report_verified",
            "semantic_task_mapping": "manifest_declared",
        }

    config_hash = project_config_sha256(config)
    claims = _claims()
    trials: list[dict[str, Any]] = []
    identity_requirement = (
        _ModelIdentityRequirement.REQUIRED
        if checkpoint_evidence is not None or task_spec_checkpoint_evidence is not None
        else _ModelIdentityRequirement.OPTIONAL
    )
    model_identity: dict[str, object] | None = None
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
    for task in manifest.heldout_tasks:
        for seed in task.seeds:
            result = trial_runner(config, task, seed=seed)
            trial_identity, trial_identity_source = _validate_trial(
                task,
                seed,
                result,
                identity_requirement=identity_requirement,
            )
            if trial_identity is not None:
                if model_identity is None:
                    model_identity = trial_identity
                elif trial_identity != model_identity:
                    raise MujocoBenchmarkError(
                        "mixed MuJoCo model identity across trials"
                    )
            trial_id = _trial_id(task, seed)
            artifact_path = paths[(task.task_id, seed)]
            trace_path = trace_paths[(task.task_id, seed)]
            trace = _validated_trace(result, rollout_id=trial_id)
            trace_artifact: str | None = None
            trace_sha256: str | None = None
            if trace is not None:
                _write_evidence(trace_path, trace)
                trace_artifact = trace_path.name
                trace_sha256 = file_sha256(trace_path)
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
                        float(value)
                        if current is None
                        else min(current, float(value))
                    )
                trace_contact_progression += int(
                    trace["contact_progression_count"]
                )
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
                trace_collision_failures += int(
                    trace["collision_failure_phase"] is not None
                )
                future_prediction_count = int(
                    trace["future_latent_prediction_count"]
                )
                future_latent_trace_trials += int(future_prediction_count > 0)
                future_latent_predictions += future_prediction_count
                future_latent_observations += int(
                    trace["future_latent_observation_count"]
                )
                future_latent_pairs += int(
                    trace["future_latent_aligned_pair_count"]
                )
                future_latent_censored += int(
                    trace["future_latent_censored_pair_count"]
                )
                future_latent_elements += int(
                    trace["future_latent_element_count"]
                )
                future_latent_squared_error += float(
                    trace["future_latent_squared_error_sum"]
                )
            trial_claims = {
                **claims,
                "runtime_future_latent_proxy_available": bool(
                    trace is not None
                    and trace["runtime_future_latent_proxy_available"]
                ),
            }
            _write_evidence(
                artifact_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "trial_id": trial_id,
                    "rollout_id": trial_id,
                    "task_id": task.task_id,
                    "seed": seed,
                    "status": result.status.value,
                    "success": result.success,
                    "criterion": TERMINAL_CRITERION,
                    "object_body": task.object_body,
                    "object_physical_profile": (
                        None
                        if result.object_physical_profile is None
                        else dict(result.object_physical_profile)
                    ),
                    "object_physical_profile_sha256": (
                        result.object_physical_profile_sha256
                    ),
                    "mujoco_model_identity": trial_identity,
                    "model_identity_source": (
                        None
                        if trial_identity_source is None
                        else trial_identity_source.value
                    ),
                    "object_position_error_m": result.object_position_error_m,
                    "position_tolerance_m": task.position_tolerance_m,
                    "failure_reason": result.failure_reason,
                    "failure_evidence": (
                        None
                        if result.failure_evidence is None
                        else dict(result.failure_evidence)
                    ),
                    "policy": policy,
                    "benchmark_scope": scope,
                    "config_sha256": config_hash,
                    "manifest_sha256": manifest_sha256,
                    "dataset_task": manifest.dataset_task,
                    "dataset_task_index": manifest.dataset_task_index,
                    "initial_joint_position": list(result.initial_joint_position),
                    "final_joint_position": (
                        None
                        if result.final_joint_position is None
                        else list(result.final_joint_position)
                    ),
                    "initial_object_position": list(result.initial_object_position),
                    "target_object_position": list(task.target_object_position),
                    "final_object_position": (
                        None
                        if result.final_object_position is None
                        else list(result.final_object_position)
                    ),
                    "rollout": dict(result.rollout),
                    "executability_trace_schema_version": (
                        None if trace is None else EXECUTABILITY_TRACE_SCHEMA
                    ),
                    "executability_trace_artifact": trace_artifact,
                    "executability_trace_sha256": trace_sha256,
                    **trial_claims,
                    "semantic_mujoco_object_state_evaluated": (
                        result.status is SemanticTrialStatus.SCORED
                    ),
                    **evidence,
                },
            )
            trials.append(
                {
                    "trial_id": trial_id,
                    "rollout_id": trial_id,
                    "task_id": task.task_id,
                    "seed": seed,
                    "status": result.status.value,
                    "success": result.success,
                    "object_body": task.object_body,
                    "object_physical_profile_sha256": (
                        result.object_physical_profile_sha256
                    ),
                    "mujoco_model_identity": trial_identity,
                    "model_identity_source": (
                        None
                        if trial_identity_source is None
                        else trial_identity_source.value
                    ),
                    "object_position_error_m": result.object_position_error_m,
                    "failure_reason": result.failure_reason,
                    "failure_evidence": (
                        None
                        if result.failure_evidence is None
                        else dict(result.failure_evidence)
                    ),
                    "executability_trace_schema_version": (
                        None if trace is None else EXECUTABILITY_TRACE_SCHEMA
                    ),
                    "executability_trace_artifact": trace_artifact,
                    "executability_trace_sha256": trace_sha256,
                    "artifact": artifact_path.name,
                    "artifact_sha256": file_sha256(artifact_path),
                }
            )

    successes, count, scored, errors = _trial_totals(trials)
    error_mean, error_max = _error_metrics(errors)
    all_successful = successes == count
    object_bodies = {task.object_body for task in manifest.heldout_tasks}
    profile_hash = _single_value(trials, "object_physical_profile_sha256")
    top_object_body = next(iter(object_bodies)) if len(object_bodies) == 1 else None
    profile = None
    if (
        checkpoint_evidence is not None or task_spec_checkpoint_evidence is not None
    ) and len(manifest.heldout_tasks) == 1:
        first_task_id = manifest.heldout_tasks[0].task_id
        for trial in trials:
            if trial["task_id"] != first_task_id:
                continue
            artifact = json.loads((artifacts / str(trial["artifact"])).read_text())
            profile = artifact.get("object_physical_profile")
            break
    claims["runtime_future_latent_proxy_available"] = future_latent_pairs > 0
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "result": "pass" if all_successful else "fail",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "benchmark_id": manifest.benchmark_id,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_sha256,
        "dataset_task": manifest.dataset_task,
        "dataset_task_index": manifest.dataset_task_index,
        "policy": policy,
        "benchmark_scope": scope,
        "criterion": TERMINAL_CRITERION,
        "object_body": top_object_body,
        "object_physical_profile": profile,
        "object_physical_profile_sha256": profile_hash,
        "mujoco_model_identity": model_identity,
        "model_identity_sources": _identity_sources(trials),
        "task_disjoint": True,
        "train_task_ids": list(manifest.train_task_ids),
        "heldout_task_ids": [task.task_id for task in manifest.heldout_tasks],
        "primary_cameras": list(PRIMARY_CAMERA_KEYS),
        "executability_trace_present": trace_trial_count > 0,
        **claims,
        **evidence,
        "semantic_mujoco_object_state_evaluated": scored > 0,
        "summary": {
            "task_count": len(manifest.heldout_tasks),
            "trial_count": count,
            "scored_trial_count": scored,
            "execution_failure_count": count - scored,
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
            "executability_minimum_contact_distance_m": (
                trace_minimum_contact_distance
            ),
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
            "success_count": successes,
            "success_rate": successes / count,
            "success_rate_95ci": list(_wilson_interval(successes, count)),
            "all_trials_successful": all_successful,
            "object_position_error_mean_m": error_mean,
            "object_position_error_max_m": error_max,
            "failure_counts": _failure_counts(trials),
            "mujoco_model_identity": model_identity,
            "model_identity_sources": _identity_sources(trials),
        },
        "tasks": _summaries(manifest, trials),
        "trials": trials,
    }


def run_benchmark(
    config: ProjectConfig,
    *,
    manifest_path: str | Path,
    artifact_dir: str | Path,
    trial_runner: TrialRunner,
) -> dict[str, Any]:
    manifest_source = _source_bytes(manifest_path, name="semantic manifest")
    manifest = load_semantic_manifest_bytes(manifest_source)
    return _run_benchmark(
        config,
        manifest=manifest,
        manifest_sha256=sha256(manifest_source).hexdigest(),
        artifact_dir=artifact_dir,
        trial_runner=trial_runner,
    )


def run_checkpoint_benchmark(
    config: ProjectConfig,
    *,
    manifest_path: str | Path,
    artifact_dir: str | Path,
    checkpoint_path: str | Path,
    training_report_path: str | Path,
    prompt_path: str | Path,
    prompt_manifest_path: str | Path | None = None,
    device: str = "cpu",
    prompt_task_expectation: PromptTaskExpectation = (
        PromptTaskExpectation.MATCH_MANIFEST
    ),
) -> dict[str, Any]:
    _require_mujoco_backend(config)
    manifest_source = _source_bytes(manifest_path, name="semantic manifest")
    manifest = load_semantic_manifest_bytes(manifest_source)
    if len(manifest.heldout_tasks) != 1:
        raise MujocoBenchmarkError(
            "checkpoint semantic benchmark requires exactly one held-out task per prompt"
        )

    prompt_manifest_source_path = (
        Path(prompt_manifest_path)
        if prompt_manifest_path is not None
        else Path(prompt_path).with_suffix(".json")
    )
    checkpoint_source = _source_bytes(checkpoint_path, name="checkpoint")
    training_report_source = _source_bytes(
        training_report_path,
        name="training report",
    )
    prompt_source = _source_bytes(prompt_path, name="prompt NPZ")
    prompt_manifest_source = _source_bytes(
        prompt_manifest_source_path,
        name="prompt manifest",
    )

    with TemporaryDirectory(prefix="so101-wam-semantic-") as temp_dir:
        snapshot_root = Path(temp_dir)
        checkpoint_snapshot = snapshot_root / "checkpoint.pt"
        prompt_snapshot = snapshot_root / "prompt.npz"
        prompt_manifest_snapshot = snapshot_root / "prompt.json"
        checkpoint_snapshot.write_bytes(checkpoint_source)
        prompt_snapshot.write_bytes(prompt_source)
        prompt_manifest_snapshot.write_bytes(prompt_manifest_source)

        bundle = load_compact_wam_bundle(checkpoint_snapshot, device=device)
        snapshot_hash, checkpoint_id = _checkpoint_identity(
            checkpoint_snapshot,
            bundle.metadata,
        )
        checkpoint_hash = sha256(checkpoint_source).hexdigest()
        if snapshot_hash != checkpoint_hash:
            raise MujocoBenchmarkError("checkpoint snapshot hash mismatch")
        training_evidence = _validate_training_report(
            training_report_source,
            checkpoint_metadata=bundle.metadata,
            checkpoint_architecture=bundle.architecture,
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_hash,
            manifest=manifest,
        )
        prompt = load_episode(prompt_snapshot, prompt_manifest_snapshot)
        prompt_identity_matches, prompt_task_match = _prompt_task_status(
            task=prompt.task,
            task_index=prompt.task_index,
            manifest=manifest,
            expectation=prompt_task_expectation,
        )
        evidence = _CheckpointEvidence(
            checkpoint_sha256=checkpoint_hash,
            checkpoint_id=checkpoint_id,
            training_report_sha256=sha256(training_report_source).hexdigest(),
            training_evidence_sha256=training_evidence,
            prompt_npz_sha256=sha256(prompt_source).hexdigest(),
            prompt_manifest_sha256=sha256(prompt_manifest_source).hexdigest(),
            prompt_fingerprint=prompt.fingerprint,
            prompt_task=prompt.task,
            prompt_task_index=prompt.task_index,
            prompt_task_expectation=prompt_task_expectation.value,
            prompt_task_identity_matches_manifest=prompt_identity_matches,
            prompt_task_match=prompt_task_match,
        )
        failure_scene: tuple[dict[str, object], str, dict[str, object]] | None = None

        def checkpoint_runner(
            config,
            task: SemanticTask,
            *,
            seed: int,
        ) -> SemanticTrialResult:
            nonlocal failure_scene
            result = run_checkpoint_task_trial(
                config,
                task,
                seed=seed,
                checkpoint_path=checkpoint_snapshot,
                prompt_path=prompt_snapshot,
                prompt_manifest_path=prompt_manifest_snapshot,
                device=device,
            )
            if result.status is not SemanticTrialStatus.EXECUTION_FAILURE:
                return result
            if not _failure_scene_needed(result):
                return result

            if failure_scene is None:
                profile, profile_sha256, model_identity = inspect_mujoco_scene(
                    config,
                    body_name=task.object_body,
                )
                if profile_sha256 != canonical_json_sha256(profile):
                    raise MujocoBenchmarkError(
                        "inspected object physical profile SHA-256 is invalid"
                    )
                failure_scene = profile, profile_sha256, model_identity
            return _with_failure_scene(result, failure_scene)

        return _run_benchmark(
            config,
            manifest=manifest,
            manifest_sha256=sha256(manifest_source).hexdigest(),
            artifact_dir=artifact_dir,
            trial_runner=checkpoint_runner,
            checkpoint_evidence=evidence,
        )


def run_task_spec_checkpoint_benchmark(
    config: ProjectConfig,
    *,
    manifest_path: str | Path,
    artifact_dir: str | Path,
    checkpoint_path: str | Path,
    training_report_path: str | Path,
    task_spec: TaskSpec,
    prompt_task: str,
    prompt_task_index: int,
    prompt_condition: str,
    device: str = "cpu",
    prompt_task_expectation: PromptTaskExpectation = (
        PromptTaskExpectation.MATCH_MANIFEST
    ),
) -> dict[str, Any]:
    """Run one paired task-spec condition against a semantic object task."""

    _require_mujoco_backend(config)
    manifest_source = _source_bytes(manifest_path, name="semantic manifest")
    manifest = load_semantic_manifest_bytes(manifest_source)
    if len(manifest.heldout_tasks) != 1:
        raise MujocoBenchmarkError(
            "task-spec semantic benchmark requires exactly one held-out task"
        )
    if not isinstance(prompt_condition, str) or not prompt_condition.strip():
        raise MujocoBenchmarkError("prompt_condition must be a non-empty string")

    kind = task_spec_kind(task_spec)
    source_sha256 = task_spec.provenance.source_sha256
    if source_sha256 is None:
        raise MujocoBenchmarkError(
            "task-spec semantic benchmark requires checksum-bound provenance"
        )
    checkpoint_source = _source_bytes(checkpoint_path, name="checkpoint")
    training_report_source = _source_bytes(
        training_report_path,
        name="training report",
    )

    with TemporaryDirectory(prefix="so101-wam-task-spec-semantic-") as temp_dir:
        checkpoint_snapshot = Path(temp_dir) / "checkpoint.pt"
        checkpoint_snapshot.write_bytes(checkpoint_source)
        bundle = load_compact_wam_bundle(checkpoint_snapshot, device=device)
        snapshot_hash, checkpoint_id = _checkpoint_identity(
            checkpoint_snapshot,
            bundle.metadata,
        )
        checkpoint_hash = sha256(checkpoint_source).hexdigest()
        if snapshot_hash != checkpoint_hash:
            raise MujocoBenchmarkError("checkpoint snapshot hash mismatch")
        training_evidence = _validate_training_report(
            training_report_source,
            checkpoint_metadata=bundle.metadata,
            checkpoint_architecture=bundle.architecture,
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_hash,
            manifest=manifest,
        )
        training_report = _json_object(
            training_report_source,
            name="training report",
        )
        protocol = training_report.get("protocol")
        if (
            bundle.metadata.get("paired_human_video") is not True
            or not isinstance(protocol, Mapping)
            or protocol.get("prompt_pairing") != "human_video_reviewed_pair"
            or protocol.get("prompt_modality") != "human_video_task_spec"
        ):
            raise MujocoBenchmarkError(
                "task-spec semantic benchmark requires a paired human-video checkpoint"
            )
        prompt_identity_matches, prompt_task_match = _prompt_task_status(
            task=prompt_task,
            task_index=prompt_task_index,
            manifest=manifest,
            expectation=prompt_task_expectation,
        )
        evidence = _TaskSpecCheckpointEvidence(
            checkpoint_sha256=checkpoint_hash,
            checkpoint_id=checkpoint_id,
            training_report_sha256=sha256(training_report_source).hexdigest(),
            training_evidence_sha256=training_evidence,
            task_spec_kind=kind.value,
            task_spec_fingerprint=task_spec.fingerprint,
            task_spec_source_id=task_spec.provenance.source_id,
            task_spec_source_sha256=source_sha256,
            prompt_condition=prompt_condition.strip(),
            prompt_task=prompt_task,
            prompt_task_index=prompt_task_index,
            prompt_task_expectation=prompt_task_expectation.value,
            prompt_task_identity_matches_manifest=prompt_identity_matches,
            prompt_task_match=prompt_task_match,
        )
        failure_scene: tuple[dict[str, object], str, dict[str, object]] | None = None

        def task_spec_runner(
            config: ProjectConfig,
            task: SemanticTask,
            *,
            seed: int,
        ) -> SemanticTrialResult:
            nonlocal failure_scene
            result = run_task_spec_checkpoint_task_trial(
                config,
                task,
                seed=seed,
                checkpoint_path=checkpoint_snapshot,
                task_spec=task_spec,
                device=device,
            )
            if result.status is not SemanticTrialStatus.EXECUTION_FAILURE:
                return result
            if not _failure_scene_needed(result):
                return result

            if failure_scene is None:
                profile, profile_sha256, model_identity = inspect_mujoco_scene(
                    config,
                    body_name=task.object_body,
                )
                if profile_sha256 != canonical_json_sha256(profile):
                    raise MujocoBenchmarkError(
                        "inspected object physical profile SHA-256 is invalid"
                    )
                failure_scene = profile, profile_sha256, model_identity
            return _with_failure_scene(result, failure_scene)

        return _run_benchmark(
            config,
            manifest=manifest,
            manifest_sha256=sha256(manifest_source).hexdigest(),
            artifact_dir=artifact_dir,
            trial_runner=task_spec_runner,
            task_spec_checkpoint_evidence=evidence,
        )


def write_benchmark_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    return _write_evidence(Path(path), report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a MuJoCo named-object position benchmark."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MUJOCO_CONFIG_PATH)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error(f"MuJoCo semantic benchmark report already exists: {args.report}")
    try:
        config = ProjectConfig.load(args.config)
        report = run_checkpoint_benchmark(
            config,
            manifest_path=args.manifest,
            artifact_dir=args.artifact_dir,
            checkpoint_path=args.checkpoint,
            training_report_path=args.training_report,
            prompt_path=args.prompt,
            prompt_manifest_path=args.prompt_manifest,
            device=args.device,
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
    "CHECKPOINT_SEMANTIC_GATE",
    "CHECKPOINT_SEMANTIC_POLICY",
    "CHECKPOINT_SEMANTIC_SCOPE",
    "SEMANTIC_BENCHMARK_GATE",
    "SEMANTIC_BENCHMARK_POLICY",
    "SEMANTIC_BENCHMARK_SCOPE",
    "TASK_SPEC_CHECKPOINT_SEMANTIC_GATE",
    "TASK_SPEC_CHECKPOINT_SEMANTIC_POLICY",
    "TASK_SPEC_CHECKPOINT_SEMANTIC_SCOPE",
    "MujocoSemanticBenchmarkError",
    "ModelIdentitySource",
    "ObjectTerminalOutcome",
    "PromptTaskExpectation",
    "RECOVERABLE_FAILURE_REASONS",
    "SemanticManifest",
    "SemanticTask",
    "SemanticTrialResult",
    "SemanticTrialStatus",
    "load_semantic_manifest",
    "load_semantic_manifest_bytes",
    "main",
    "run_benchmark",
    "run_checkpoint_benchmark",
    "run_checkpoint_task_trial",
    "run_task_spec_checkpoint_benchmark",
    "run_task_spec_checkpoint_task_trial",
    "score_object_position",
    "validate_failure_evidence",
    "write_benchmark_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
