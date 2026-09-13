"""Aggregate separately prompted semantic MuJoCo benchmark cases."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isclose, isfinite
import os
from pathlib import Path, PurePosixPath
from statistics import fmean
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import CheckpointError
from .config import ConfigError, DEFAULT_MUJOCO_CONFIG_PATH, ProjectConfig
from .dataset import DatasetError
from .deployment import canonical_json_sha256, project_config_sha256
from .mujoco_benchmark import MujocoBenchmarkError, _wilson_interval
from .mujoco_cli import MujocoCLIError
from .mujoco_identity import MujocoIdentityError, MujocoModelIdentity
from .mujoco_semantic_benchmark import (
    CHECKPOINT_SEMANTIC_GATE,
    CHECKPOINT_SEMANTIC_POLICY,
    CHECKPOINT_SEMANTIC_SCOPE,
    ModelIdentitySource,
    SCHEMA_VERSION as SEMANTIC_REPORT_SCHEMA,
    TERMINAL_CRITERION,
    SemanticManifest,
    SemanticTask,
    SemanticTrialStatus,
    load_semantic_manifest_bytes,
    run_checkpoint_benchmark,
    validate_failure_evidence,
)
from .paired_data import SemanticMatchStatus
from .policy import PolicyError
from .rollout import RolloutError
from .runtime import RuntimeErrorState
from .training import TRAINING_REPORT_SCHEMA


SEMANTIC_SUITE_SCHEMA = 1
SEMANTIC_MAPPING_SCHEMA = 3
SEMANTIC_SUITE_REPORT_SCHEMA = 5
OBJECT_TASK_SIGNATURE_SCHEMA = 2
SEMANTIC_MAPPING_SCOPE = "local_dataset_object_mapping_attestation_only"
SEMANTIC_SUITE_GATE = "G8-candidate-semantic-suite"
MIN_SEMANTIC_CASES = 2

_SUITE_FIELDS = {"schema_version", "suite_id", "cases"}
_CASE_FIELDS = {
    "case_id",
    "semantic_manifest",
    "prompt",
    "prompt_manifest",
}
_MAPPING_FIELDS = {"schema_version", "suite_id", "scope", "mappings"}
_MAPPING_ITEM_FIELDS = {
    "case_id",
    "semantic_match",
    "reviewer_id",
    "dataset_task",
    "object_task_id",
    "object_task_signature_sha256",
    "semantic_manifest_sha256",
    "criterion",
    "object_body",
    "mapping_basis",
}
_DATASET_TASK_FIELDS = {"task_index", "task"}
_LOWER_SHA256 = frozenset("0123456789abcdef")


class MujocoSemanticSuiteError(ValueError):
    """Raised when a semantic suite cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class SemanticSuiteCase:
    case_id: str
    semantic_manifest_path: Path
    prompt_path: Path
    prompt_manifest_path: Path


@dataclass(frozen=True, slots=True)
class SemanticSuite:
    suite_id: str
    cases: tuple[SemanticSuiteCase, ...]


@dataclass(frozen=True, slots=True)
class SemanticMapping:
    case_id: str
    semantic_match: SemanticMatchStatus
    reviewer_id: str | None
    dataset_task: str
    dataset_task_index: int
    object_task_id: str
    object_task_signature_sha256: str
    semantic_manifest_sha256: str
    criterion: str
    object_body: str
    mapping_basis: str


@dataclass(frozen=True, slots=True)
class SemanticMappingManifest:
    suite_id: str
    scope: str
    mappings: tuple[SemanticMapping, ...]


@dataclass(frozen=True, slots=True)
class _FrozenCase:
    case: SemanticSuiteCase
    mapping: SemanticMapping
    manifest: SemanticManifest
    semantic_manifest: bytes
    prompt: bytes
    prompt_manifest: bytes


@dataclass(frozen=True, slots=True)
class _TrainingIdentity:
    checkpoint_id: str
    training_evidence_sha256: str


def _json_object(source: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MujocoSemanticSuiteError(
                    f"{name} has duplicate field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise MujocoSemanticSuiteError(
            f"{name} contains non-standard numeric constant: {value}"
        )

    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MujocoSemanticSuiteError(f"{name} must use UTF-8") from error

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise MujocoSemanticSuiteError(f"{name} must be valid JSON") from error
    if not isinstance(payload, dict):
        raise MujocoSemanticSuiteError(f"{name} must be a JSON object")
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
        raise MujocoSemanticSuiteError(f"{name} is missing fields: {missing}")
    if extra:
        raise MujocoSemanticSuiteError(f"{name} has unknown fields: {extra}")


def _schema(value: object, expected: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise MujocoSemanticSuiteError(
            f"{name} requires schema_version={expected}"
        )


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MujocoSemanticSuiteError(f"{name} must be a non-empty string")
    return value.strip()


def _safe_id(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if not result.replace("_", "").replace("-", "").isalnum():
        raise MujocoSemanticSuiteError(f"{name} is artifact-unsafe")
    return result


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise MujocoSemanticSuiteError(f"{name} must be lowercase SHA-256")
    return value


def _source_bytes(path: str | Path, *, name: str) -> bytes:
    target = Path(path)
    try:
        return target.read_bytes()
    except OSError as error:
        raise MujocoSemanticSuiteError(f"failed to read {name}: {target}") from error


def _bundle_path(root: str | Path, value: object, *, name: str) -> Path:
    relative = _string(value, name=name)
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise MujocoSemanticSuiteError(f"{name} must be a normalized bundle-relative path")

    bundle_root = Path(root).resolve()
    target = (bundle_root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(bundle_root)
    except ValueError as error:
        raise MujocoSemanticSuiteError(
            f"{name} must stay inside the bundle root"
        ) from error
    return target


def load_semantic_suite_bytes(
    source: bytes,
    *,
    root: str | Path,
) -> SemanticSuite:
    """Load one strict suite of separately prompted semantic cases."""

    payload = _json_object(source, name="semantic suite")
    _exact_fields(payload, _SUITE_FIELDS, name="semantic suite")
    _schema(
        payload.get("schema_version"),
        SEMANTIC_SUITE_SCHEMA,
        name="semantic suite",
    )
    suite_id = _safe_id(payload.get("suite_id"), name="suite_id")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < MIN_SEMANTIC_CASES:
        raise MujocoSemanticSuiteError(
            f"semantic suite requires at least {MIN_SEMANTIC_CASES} cases"
        )

    cases: list[SemanticSuiteCase] = []
    seen_case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise MujocoSemanticSuiteError(f"cases[{index}] must be an object")
        _exact_fields(raw_case, _CASE_FIELDS, name=f"cases[{index}]")
        case_id = _safe_id(raw_case.get("case_id"), name=f"cases[{index}].case_id")
        if case_id in seen_case_ids:
            raise MujocoSemanticSuiteError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        cases.append(
            SemanticSuiteCase(
                case_id=case_id,
                semantic_manifest_path=_bundle_path(
                    root,
                    raw_case.get("semantic_manifest"),
                    name=f"cases[{index}].semantic_manifest",
                ),
                prompt_path=_bundle_path(
                    root,
                    raw_case.get("prompt"),
                    name=f"cases[{index}].prompt",
                ),
                prompt_manifest_path=_bundle_path(
                    root,
                    raw_case.get("prompt_manifest"),
                    name=f"cases[{index}].prompt_manifest",
                ),
            )
        )
    return SemanticSuite(suite_id=suite_id, cases=tuple(cases))


def _dataset_task(value: object, *, name: str) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise MujocoSemanticSuiteError(f"{name} must be an object")
    _exact_fields(value, _DATASET_TASK_FIELDS, name=name)
    task = _string(value.get("task"), name=f"{name}.task")
    task_index = value.get("task_index")
    if (
        not isinstance(task_index, int)
        or isinstance(task_index, bool)
        or task_index < 0
    ):
        raise MujocoSemanticSuiteError(f"{name}.task_index must be non-negative")
    return task, task_index


def _semantic_match(
    value: object,
    reviewer_id: object,
    *,
    name: str,
) -> tuple[SemanticMatchStatus, str | None]:
    if not isinstance(value, str):
        raise MujocoSemanticSuiteError(
            f"{name}.semantic_match must be unverified or human_reviewed"
        )
    try:
        status = SemanticMatchStatus(value)
    except ValueError as error:
        raise MujocoSemanticSuiteError(
            f"{name}.semantic_match must be unverified or human_reviewed"
        ) from error

    if status is SemanticMatchStatus.UNVERIFIED:
        if reviewer_id is not None:
            raise MujocoSemanticSuiteError(
                f"{name}.reviewer_id must be null for unverified mappings"
            )
        return status, None

    reviewer = _string(reviewer_id, name=f"{name}.reviewer_id")
    return status, reviewer


def object_task_signature(task: SemanticTask) -> str:
    """Hash object-state geometry without task labels or identifiers."""

    initial_positions = [
        {
            "seed": seed,
            "position": list(position),
        }
        for seed, position in sorted(task.initial_object_positions)
    ]
    canonical = {
        "schema_version": OBJECT_TASK_SIGNATURE_SCHEMA,
        "criterion": TERMINAL_CRITERION,
        "object_body": task.object_body,
        "initial_object_positions": initial_positions,
        "target_object_position": list(task.target_object_position),
        "position_tolerance_m": task.position_tolerance_m,
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_semantic_mapping_bytes(
    source: bytes,
    *,
    root: str | Path,
    suite: SemanticSuite,
) -> SemanticMappingManifest:
    """Load and verify one local dataset-to-object mapping attestation."""

    payload = _json_object(source, name="semantic mapping")
    _exact_fields(payload, _MAPPING_FIELDS, name="semantic mapping")
    _schema(
        payload.get("schema_version"),
        SEMANTIC_MAPPING_SCHEMA,
        name="semantic mapping",
    )
    suite_id = _safe_id(payload.get("suite_id"), name="semantic mapping suite_id")
    if suite_id != suite.suite_id:
        raise MujocoSemanticSuiteError("semantic mapping suite_id does not match suite")
    scope = _string(payload.get("scope"), name="semantic mapping scope")
    if scope != SEMANTIC_MAPPING_SCOPE:
        raise MujocoSemanticSuiteError(
            f"semantic mapping scope must be {SEMANTIC_MAPPING_SCOPE!r}"
        )

    raw_mappings = payload.get("mappings")
    if not isinstance(raw_mappings, list):
        raise MujocoSemanticSuiteError("semantic mapping mappings must be a list")

    cases_by_id = {case.case_id: case for case in suite.cases}
    mappings_by_case: dict[str, SemanticMapping] = {}
    dataset_indices: set[int] = set()
    dataset_tasks: set[str] = set()
    object_task_ids: set[str] = set()
    object_task_signatures: set[str] = set()
    for index, raw_mapping in enumerate(raw_mappings):
        name = f"mappings[{index}]"
        if not isinstance(raw_mapping, dict):
            raise MujocoSemanticSuiteError(f"{name} must be an object")
        _exact_fields(raw_mapping, _MAPPING_ITEM_FIELDS, name=name)
        case_id = _safe_id(raw_mapping.get("case_id"), name=f"{name}.case_id")
        if case_id not in cases_by_id or case_id in mappings_by_case:
            raise MujocoSemanticSuiteError(
                "semantic mapping requires exact 1:1 case coverage"
            )
        semantic_match, reviewer_id = _semantic_match(
            raw_mapping.get("semantic_match"),
            raw_mapping.get("reviewer_id"),
            name=name,
        )
        dataset_task, dataset_task_index = _dataset_task(
            raw_mapping.get("dataset_task"),
            name=f"{name}.dataset_task",
        )
        if dataset_task_index in dataset_indices or dataset_task in dataset_tasks:
            raise MujocoSemanticSuiteError(
                "semantic mapping requires unique dataset identities"
            )
        dataset_indices.add(dataset_task_index)
        dataset_tasks.add(dataset_task)

        object_task_id = _safe_id(
            raw_mapping.get("object_task_id"),
            name=f"{name}.object_task_id",
        )
        if object_task_id in object_task_ids:
            raise MujocoSemanticSuiteError(
                "semantic mapping requires unique object_task_id values"
            )
        object_task_ids.add(object_task_id)

        manifest_hash = _sha256(
            raw_mapping.get("semantic_manifest_sha256"),
            name=f"{name}.semantic_manifest_sha256",
        )
        case = cases_by_id[case_id]
        semantic_source = _source_bytes(
            case.semantic_manifest_path,
            name=f"semantic manifest for {case_id}",
        )
        actual_manifest_hash = sha256(semantic_source).hexdigest()
        if manifest_hash != actual_manifest_hash:
            raise MujocoSemanticSuiteError(
                f"{name}.semantic_manifest_sha256 does not match suite input"
            )
        try:
            manifest = load_semantic_manifest_bytes(semantic_source)
        except MujocoBenchmarkError as error:
            raise MujocoSemanticSuiteError(
                f"{name} semantic manifest is invalid: {error}"
            ) from error
        if len(manifest.heldout_tasks) != 1:
            raise MujocoSemanticSuiteError(
                f"{name} requires exactly one held-out task per prompt"
            )
        if (
            dataset_task != manifest.dataset_task
            or dataset_task_index != manifest.dataset_task_index
        ):
            raise MujocoSemanticSuiteError(
                f"{name}.dataset_task does not match semantic manifest"
            )
        if object_task_id != manifest.heldout_tasks[0].task_id:
            raise MujocoSemanticSuiteError(
                f"{name}.object_task_id does not match semantic manifest"
            )

        criterion = _string(raw_mapping.get("criterion"), name=f"{name}.criterion")
        if criterion != TERMINAL_CRITERION:
            raise MujocoSemanticSuiteError(
                f"{name}.criterion must be {TERMINAL_CRITERION!r}"
            )
        object_body = _string(
            raw_mapping.get("object_body"),
            name=f"{name}.object_body",
        )
        if object_body != manifest.heldout_tasks[0].object_body:
            raise MujocoSemanticSuiteError(
                f"{name}.object_body does not match semantic manifest"
            )
        declared_task_signature = _sha256(
            raw_mapping.get("object_task_signature_sha256"),
            name=f"{name}.object_task_signature_sha256",
        )
        actual_task_signature = object_task_signature(manifest.heldout_tasks[0])
        if declared_task_signature != actual_task_signature:
            raise MujocoSemanticSuiteError(
                f"{name}.object_task_signature_sha256 does not match semantic manifest"
            )
        if actual_task_signature in object_task_signatures:
            raise MujocoSemanticSuiteError(
                "semantic mapping requires unique object task signatures"
            )
        object_task_signatures.add(actual_task_signature)
        mapping_basis = _string(
            raw_mapping.get("mapping_basis"),
            name=f"{name}.mapping_basis",
        )
        mappings_by_case[case_id] = SemanticMapping(
            case_id=case_id,
            semantic_match=semantic_match,
            reviewer_id=reviewer_id,
            dataset_task=dataset_task,
            dataset_task_index=dataset_task_index,
            object_task_id=object_task_id,
            object_task_signature_sha256=actual_task_signature,
            semantic_manifest_sha256=manifest_hash,
            criterion=criterion,
            object_body=object_body,
            mapping_basis=mapping_basis,
        )

    suite_case_ids = tuple(case.case_id for case in suite.cases)
    if set(mappings_by_case) != set(suite_case_ids):
        raise MujocoSemanticSuiteError(
            "semantic mapping requires exact 1:1 case coverage"
        )
    return SemanticMappingManifest(
        suite_id=suite_id,
        scope=scope,
        mappings=tuple(mappings_by_case[case_id] for case_id in suite_case_ids),
    )


def _training_identity(source: bytes, checkpoint_sha256: str) -> _TrainingIdentity:
    report = _json_object(source, name="training report")
    if report.get("schema_version") != TRAINING_REPORT_SCHEMA:
        raise MujocoSemanticSuiteError("training report schema is invalid")
    expected = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
    }
    for field, value in expected.items():
        if report.get(field) != value or isinstance(report.get(field), bool) != isinstance(value, bool):
            raise MujocoSemanticSuiteError(
                f"training report requires {field}={value!r}"
            )
    checkpoint_id = _string(report.get("checkpoint_id"), name="checkpoint_id")
    training_evidence = _sha256(
        report.get("training_evidence_sha256"),
        name="training_evidence_sha256",
    )
    artifacts = report.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise MujocoSemanticSuiteError(
            "training report does not bind the exact checkpoint bytes"
        )
    return _TrainingIdentity(checkpoint_id, training_evidence)


def _freeze_cases(
    suite: SemanticSuite,
    mapping: SemanticMappingManifest,
) -> tuple[_FrozenCase, ...]:
    mappings = {item.case_id: item for item in mapping.mappings}
    frozen: list[_FrozenCase] = []
    prompt_hashes: set[str] = set()
    prompt_manifest_hashes: set[str] = set()
    for case in suite.cases:
        semantic_source = _source_bytes(
            case.semantic_manifest_path,
            name=f"semantic manifest for {case.case_id}",
        )
        manifest = load_semantic_manifest_bytes(semantic_source)
        if len(manifest.heldout_tasks) != 1:
            raise MujocoSemanticSuiteError(
                f"case {case.case_id!r} requires exactly one held-out task per prompt"
            )
        item = mappings[case.case_id]
        if (
            item.dataset_task != manifest.dataset_task
            or item.dataset_task_index != manifest.dataset_task_index
        ):
            raise MujocoSemanticSuiteError(
                f"case {case.case_id!r} mapping dataset_task does not match semantic manifest"
            )
        if sha256(semantic_source).hexdigest() != item.semantic_manifest_sha256:
            raise MujocoSemanticSuiteError(
                f"case {case.case_id!r} semantic_manifest_sha256 changed during validation"
            )
        prompt_source = _source_bytes(
            case.prompt_path,
            name=f"prompt for {case.case_id}",
        )
        prompt_hash = sha256(prompt_source).hexdigest()
        if prompt_hash in prompt_hashes:
            raise MujocoSemanticSuiteError(
                "semantic suite cannot reuse prompt NPZ bytes across cases"
            )
        prompt_hashes.add(prompt_hash)
        prompt_manifest_source = _source_bytes(
            case.prompt_manifest_path,
            name=f"prompt manifest for {case.case_id}",
        )
        prompt_manifest_hash = sha256(prompt_manifest_source).hexdigest()
        if prompt_manifest_hash in prompt_manifest_hashes:
            raise MujocoSemanticSuiteError(
                "semantic suite cannot reuse prompt manifest bytes across cases"
            )
        prompt_manifest_hashes.add(prompt_manifest_hash)
        frozen.append(
            _FrozenCase(
                case=case,
                mapping=item,
                manifest=manifest,
                semantic_manifest=semantic_source,
                prompt=prompt_source,
                prompt_manifest=prompt_manifest_source,
            )
        )
    return tuple(frozen)


def _strict_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MujocoSemanticSuiteError(f"nested {name} must be non-negative")
    return value


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MujocoSemanticSuiteError(f"nested {name} must be numeric")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise MujocoSemanticSuiteError(f"nested {name} must be finite and non-negative")
    return result


def _nested_error(message: str) -> MujocoSemanticSuiteError:
    return MujocoSemanticSuiteError(f"nested semantic report {message}")


def _validate_interval(
    value: object,
    expected: tuple[float, float],
    *,
    name: str,
) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise _nested_error(f"{name} must contain two values")
    for index, expected_value in enumerate(expected):
        actual = _strict_float(value[index], name=f"{name}[{index}]")
        if not isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise _nested_error(f"{name} disagrees with trials")


def _validate_metric_summary(
    value: object,
    *,
    name: str,
    successes: int,
    count: int,
    scored: int,
    errors: tuple[float, ...],
    failure_counts: Mapping[str, int],
    top_level: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise _nested_error(f"{name} must be an object")
    exact = {
        "trial_count": count,
        "scored_trial_count": scored,
        "execution_failure_count": count - scored,
        "success_count": successes,
        "failure_counts": dict(failure_counts),
    }
    if top_level:
        exact.update(
            {
                "task_count": 1,
                "all_trials_successful": successes == count,
            }
        )
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise _nested_error(f"{name}.{field} disagrees with trials")

    actual_rate = _strict_float(value.get("success_rate"), name=f"{name}.success_rate")
    if not isclose(actual_rate, successes / count, rel_tol=0.0, abs_tol=1e-12):
        raise _nested_error(f"{name}.success_rate disagrees with trials")

    metrics = {
        "object_position_error_mean_m": None if not errors else fmean(errors),
        "object_position_error_max_m": None if not errors else max(errors),
    }
    for field, expected in metrics.items():
        if expected is None:
            if value.get(field) is not None:
                raise _nested_error(f"{name}.{field} disagrees with trials")
            continue
        actual = _strict_float(value.get(field), name=f"{name}.{field}")
        if not isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise _nested_error(f"{name}.{field} disagrees with trials")
    _validate_interval(
        value.get("success_rate_95ci"),
        _wilson_interval(successes, count),
        name=f"{name}.success_rate_95ci",
    )


def _artifact_path(root: Path, value: object) -> Path:
    try:
        target = _bundle_path(root, value, name="nested trial artifact")
    except MujocoSemanticSuiteError as error:
        raise _nested_error(str(error)) from error
    return target


def _model_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise _nested_error("compiled model identity must be an object")
    try:
        return MujocoModelIdentity.from_payload(value).to_payload()
    except MujocoIdentityError as error:
        raise _nested_error(f"invalid model identity: {error}") from error


def _validate_nested_report(
    report: Mapping[str, Any],
    *,
    frozen: _FrozenCase,
    config_sha256: str,
    checkpoint_sha256: str,
    training_report_sha256: str,
    training_identity: _TrainingIdentity,
    artifact_dir: Path,
) -> dict[str, Any]:
    task = frozen.manifest.heldout_tasks[0]
    expected = {
        "schema_version": SEMANTIC_REPORT_SCHEMA,
        "gate": CHECKPOINT_SEMANTIC_GATE,
        "mode": "mujoco",
        "evidence_level": "simulation",
        "benchmark_id": frozen.manifest.benchmark_id,
        "config_sha256": config_sha256,
        "manifest_sha256": sha256(frozen.semantic_manifest).hexdigest(),
        "dataset_task": frozen.manifest.dataset_task,
        "dataset_task_index": frozen.manifest.dataset_task_index,
        "policy": CHECKPOINT_SEMANTIC_POLICY,
        "benchmark_scope": CHECKPOINT_SEMANTIC_SCOPE,
        "criterion": TERMINAL_CRITERION,
        "object_body": task.object_body,
        "task_disjoint": True,
        "robot_used": False,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": training_identity.checkpoint_id,
        "training_report_sha256": training_report_sha256,
        "training_evidence_sha256": training_identity.training_evidence_sha256,
        "prompt_npz_sha256": sha256(frozen.prompt).hexdigest(),
        "prompt_manifest_sha256": sha256(frozen.prompt_manifest).hexdigest(),
        "prompt_task": frozen.manifest.dataset_task,
        "prompt_task_index": frozen.manifest.dataset_task_index,
        "prompt_task_expectation": "match_manifest",
        "prompt_task_identity_matches_manifest": True,
        "prompt_task_match": "dataset_identity_verified",
        "task_disjoint_basis": "checkpoint_bound_task_inventory",
        "checkpoint_task_split": "training_report_verified",
        "semantic_task_mapping": "manifest_declared",
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise _nested_error(f"requires {field}={value!r}")

    profile = report.get("object_physical_profile")
    if not isinstance(profile, Mapping):
        raise _nested_error("object_physical_profile must be an object")
    profile_hash = _sha256(
        report.get("object_physical_profile_sha256"),
        name="object_physical_profile_sha256",
    )
    try:
        actual_profile_hash = canonical_json_sha256(dict(profile))
    except (TypeError, ValueError) as error:
        raise _nested_error("object_physical_profile is not canonical JSON") from error
    if profile_hash != actual_profile_hash:
        raise _nested_error("object physical profile SHA-256 does not match payload")
    model_identity = _model_identity(report.get("mujoco_model_identity"))

    result = report.get("result")
    if result not in {"pass", "fail"}:
        raise _nested_error("result must be pass or fail")
    heldout_ids = report.get("heldout_task_ids")
    expected_task_id = task.task_id
    if heldout_ids != [expected_task_id]:
        raise _nested_error("heldout task identity does not match manifest")
    if report.get("train_task_ids") != list(frozen.manifest.train_task_ids):
        raise _nested_error("train task identities do not match manifest")

    raw_trials = report.get("trials")
    if not isinstance(raw_trials, list):
        raise _nested_error("trials must be a list")
    expected_seeds = task.seeds
    if len(raw_trials) != len(expected_seeds):
        raise _nested_error("trial schedule does not match semantic manifest")

    trials: list[dict[str, Any]] = []
    trial_ids: set[str] = set()
    seen_seeds: list[int] = []
    for index, trial in enumerate(raw_trials):
        if not isinstance(trial, Mapping):
            raise _nested_error(f"trials[{index}] must be an object")
        trial_id = _string(trial.get("trial_id"), name=f"trials[{index}].trial_id")
        if trial_id in trial_ids:
            raise _nested_error("trial IDs must be unique")
        trial_ids.add(trial_id)
        if trial.get("task_id") != expected_task_id:
            raise _nested_error("trial task_id does not match semantic manifest")
        if trial.get("object_body") != task.object_body:
            raise _nested_error("trial object_body does not match semantic manifest")
        if trial.get("object_physical_profile_sha256") != profile_hash:
            raise _nested_error("trial object physical profile does not match report")
        if _model_identity(trial.get("mujoco_model_identity")) != model_identity:
            raise _nested_error("trial model identity does not match report")
        seed = _strict_int(trial.get("seed"), name=f"trials[{index}].seed")
        seen_seeds.append(seed)
        success = trial.get("success")
        if not isinstance(success, bool):
            raise _nested_error(f"trials[{index}].success must be boolean")
        status = trial.get("status")
        if status not in {
            SemanticTrialStatus.SCORED.value,
            SemanticTrialStatus.EXECUTION_FAILURE.value,
        }:
            raise _nested_error(f"trials[{index}].status is invalid")
        identity_source = trial.get("model_identity_source")
        if (
            not isinstance(identity_source, str)
            or identity_source not in {item.value for item in ModelIdentitySource}
            or (
                status == SemanticTrialStatus.SCORED.value
                and identity_source != ModelIdentitySource.ROLLOUT_SESSION.value
            )
        ):
            raise _nested_error("trial model identity source disagrees with status")
        failure_reason = trial.get("failure_reason")
        failure_evidence = trial.get("failure_evidence")
        if status == SemanticTrialStatus.EXECUTION_FAILURE.value:
            if success or trial.get("object_position_error_m") is not None:
                raise _nested_error(
                    "execution failure cannot contain a terminal outcome"
                )
            if not isinstance(failure_reason, str) or not failure_reason:
                raise _nested_error("execution failure requires failure_reason")
            try:
                validate_failure_evidence(failure_reason, failure_evidence)
            except MujocoBenchmarkError as error:
                raise _nested_error(str(error)) from error
            object_error = None
        else:
            object_error = _strict_float(
                trial.get("object_position_error_m"),
                name=f"trials[{index}].object_position_error_m",
            )
            if failure_evidence is not None:
                raise _nested_error("scored trial cannot have failure_evidence")
        if success:
            if failure_reason is not None:
                raise _nested_error("successful trial cannot have failure_reason")
        elif not isinstance(failure_reason, str) or not failure_reason:
            raise _nested_error("failed trial requires failure_reason")
        artifact = _artifact_path(artifact_dir, trial.get("artifact"))
        artifact_hash = _sha256(
            trial.get("artifact_sha256"),
            name=f"trials[{index}].artifact_sha256",
        )
        artifact_source = _source_bytes(artifact, name="nested trial artifact")
        if sha256(artifact_source).hexdigest() != artifact_hash:
            raise _nested_error("trial artifact SHA-256 does not match bytes")
        artifact_payload = _json_object(artifact_source, name="nested trial artifact")
        if (
            _model_identity(artifact_payload.get("mujoco_model_identity")) != model_identity
            or artifact_payload.get("model_identity_source") != identity_source
        ):
            raise _nested_error("trial artifact model identity disagrees with index")
        trials.append(
            {
                "trial_id": trial_id,
                "task_id": expected_task_id,
                "object_body": task.object_body,
                "object_physical_profile_sha256": profile_hash,
                "mujoco_model_identity": model_identity,
                "model_identity_source": identity_source,
                "seed": seed,
                "status": status,
                "success": success,
                "object_position_error_m": object_error,
                "failure_reason": failure_reason,
                "failure_evidence": failure_evidence,
                "artifact": artifact.name,
                "artifact_sha256": artifact_hash,
            }
        )

    if tuple(seen_seeds) != expected_seeds:
        raise _nested_error("trial seed schedule does not match semantic manifest")

    successes = sum(int(trial["success"]) for trial in trials)
    count = len(trials)
    errors = tuple(
        float(trial["object_position_error_m"])
        for trial in trials
        if trial["status"] == SemanticTrialStatus.SCORED.value
    )
    scored = len(errors)
    if report.get("semantic_mujoco_object_state_evaluated") is not (scored > 0):
        raise _nested_error(
            "semantic_mujoco_object_state_evaluated disagrees with trials"
        )
    failure_counts = dict(
        sorted(
            Counter(
                str(trial["failure_reason"])
                for trial in trials
                if trial["failure_reason"] is not None
            ).items()
        )
    )
    expected_result = "pass" if successes == count else "fail"
    if result != expected_result:
        raise _nested_error("result disagrees with trial outcomes")

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise _nested_error("summary must be an object")
    if _model_identity(summary.get("mujoco_model_identity")) != model_identity:
        raise _nested_error("summary model identity does not match report")
    identity_sources = sorted({trial["model_identity_source"] for trial in trials})
    if summary.get("model_identity_sources") != identity_sources:
        raise _nested_error("summary model identity sources do not match trials")
    _validate_metric_summary(
        summary,
        name="summary",
        successes=successes,
        count=count,
        scored=scored,
        errors=errors,
        failure_counts=failure_counts,
        top_level=True,
    )
    task_summaries = report.get("tasks")
    if not isinstance(task_summaries, list) or len(task_summaries) != 1:
        raise _nested_error("task summary requires exactly one task")
    task_summary = task_summaries[0]
    if not isinstance(task_summary, Mapping):
        raise _nested_error("task summary must be an object")
    if task_summary.get("task_id") != task.task_id:
        raise _nested_error("task summary task_id does not match manifest")
    if task_summary.get("label") != task.label:
        raise _nested_error("task summary label does not match manifest")
    if task_summary.get("object_body") != task.object_body:
        raise _nested_error("task summary object_body does not match manifest")
    if task_summary.get("object_physical_profile_sha256") != profile_hash:
        raise _nested_error("task summary object physical profile does not match report")
    if _model_identity(task_summary.get("mujoco_model_identity")) != model_identity:
        raise _nested_error("task summary model identity does not match report")
    if (
        task_summary.get("model_identity_sources") != identity_sources
        or report.get("model_identity_sources") != identity_sources
    ):
        raise _nested_error("model identity sources do not match trials")
    _validate_metric_summary(
        task_summary,
        name="task summary",
        successes=successes,
        count=count,
        scored=scored,
        errors=errors,
        failure_counts=failure_counts,
        top_level=False,
    )
    return {
        "result": result,
        "trial_count": count,
        "scored_trial_count": scored,
        "execution_failure_count": count - scored,
        "success_count": successes,
        "success_rate": successes / count,
        "object_position_error_mean_m": None if not errors else fmean(errors),
        "object_position_error_max_m": None if not errors else max(errors),
        "failure_counts": failure_counts,
        "object_body": task.object_body,
        "object_physical_profile": dict(profile),
        "object_physical_profile_sha256": profile_hash,
        "mujoco_model_identity": model_identity,
        "model_identity_sources": identity_sources,
        "trials": trials,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> Path:
    encoded = (
        json.dumps(
            dict(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise MujocoSemanticSuiteError(
                f"semantic suite report already exists: {path}"
            ) from error
    except OSError as error:
        raise MujocoSemanticSuiteError(
            f"failed to write semantic suite report: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _snapshot(path: Path, source: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
    return path


def _run_frozen_cases(
    config: ProjectConfig,
    *,
    frozen_cases: tuple[_FrozenCase, ...],
    snapshot_root: Path,
    artifacts: Path,
    checkpoint_source: bytes,
    training_source: bytes,
    checkpoint_sha256: str,
    training_report_sha256: str,
    training_identity: _TrainingIdentity,
    config_sha256: str,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    checkpoint_snapshot = _snapshot(
        snapshot_root / "checkpoint.pt",
        checkpoint_source,
    )
    training_snapshot = _snapshot(
        snapshot_root / "training-report.json",
        training_source,
    )
    case_reports: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    mapping_status_counts: Counter[str] = Counter()
    for frozen in frozen_cases:
        case = frozen.case
        case_snapshot = snapshot_root / "cases" / case.case_id
        manifest_snapshot = _snapshot(
            case_snapshot / "semantic" / case.semantic_manifest_path.name,
            frozen.semantic_manifest,
        )
        prompt_snapshot = _snapshot(
            case_snapshot / "prompt" / case.prompt_path.name,
            frozen.prompt,
        )
        prompt_manifest_snapshot = _snapshot(
            case_snapshot / "prompt" / case.prompt_manifest_path.name,
            frozen.prompt_manifest,
        )
        case_artifacts = artifacts / case.case_id
        nested = run_checkpoint_benchmark(
            config,
            manifest_path=manifest_snapshot,
            artifact_dir=case_artifacts,
            checkpoint_path=checkpoint_snapshot,
            training_report_path=training_snapshot,
            prompt_path=prompt_snapshot,
            prompt_manifest_path=prompt_manifest_snapshot,
            device=device,
        )
        validated = _validate_nested_report(
            nested,
            frozen=frozen,
            config_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            training_report_sha256=training_report_sha256,
            training_identity=training_identity,
            artifact_dir=case_artifacts,
        )
        nested_report_path = case_artifacts / "benchmark-report.json"
        _write_report(nested_report_path, nested)
        nested_report_hash = sha256(nested_report_path.read_bytes()).hexdigest()
        mapping_status_counts[frozen.mapping.semantic_match.value] += 1
        case_reports.append(
            {
                "case_id": case.case_id,
                "result": validated["result"],
                "dataset_task": frozen.manifest.dataset_task,
                "dataset_task_index": frozen.manifest.dataset_task_index,
                "object_task_id": frozen.mapping.object_task_id,
                "object_task_signature_sha256": (
                    frozen.mapping.object_task_signature_sha256
                ),
                "object_body": validated["object_body"],
                "object_physical_profile": validated[
                    "object_physical_profile"
                ],
                "object_physical_profile_sha256": validated[
                    "object_physical_profile_sha256"
                ],
                "mujoco_model_identity": validated["mujoco_model_identity"],
                "model_identity_sources": validated["model_identity_sources"],
                "semantic_match": frozen.mapping.semantic_match.value,
                "reviewer_id": frozen.mapping.reviewer_id,
                "mapping_basis": frozen.mapping.mapping_basis,
                "semantic_manifest_sha256": sha256(
                    frozen.semantic_manifest
                ).hexdigest(),
                "prompt_npz_sha256": sha256(frozen.prompt).hexdigest(),
                "prompt_manifest_sha256": sha256(
                    frozen.prompt_manifest
                ).hexdigest(),
                "benchmark_report": f"{case.case_id}/benchmark-report.json",
                "benchmark_report_sha256": nested_report_hash,
                "summary": {
                    key: value
                    for key, value in validated.items()
                    if key != "trials"
                },
            }
        )
        all_trials.extend(validated["trials"])
    return case_reports, all_trials, mapping_status_counts


def run_mujoco_semantic_suite(
    config: ProjectConfig,
    *,
    suite_path: str | Path,
    mapping_path: str | Path,
    checkpoint_path: str | Path,
    training_report_path: str | Path,
    artifact_dir: str | Path,
    report_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run all suite cases and publish one hash-bound aggregate report."""

    output = Path(report_path)
    artifacts = Path(artifact_dir)
    if output.exists():
        raise MujocoSemanticSuiteError(
            f"semantic suite report already exists: {output}"
        )
    if artifacts.exists():
        raise MujocoSemanticSuiteError(
            f"semantic suite artifact directory already exists: {artifacts}"
        )

    suite_source = _source_bytes(suite_path, name="semantic suite")
    mapping_source = _source_bytes(mapping_path, name="semantic mapping")
    checkpoint_source = _source_bytes(checkpoint_path, name="checkpoint")
    training_source = _source_bytes(training_report_path, name="training report")
    suite = load_semantic_suite_bytes(suite_source, root=Path(suite_path).parent)
    mapping = load_semantic_mapping_bytes(
        mapping_source,
        root=Path(mapping_path).parent,
        suite=suite,
    )
    frozen_cases = _freeze_cases(suite, mapping)
    checkpoint_hash = sha256(checkpoint_source).hexdigest()
    training_hash = sha256(training_source).hexdigest()
    training_identity = _training_identity(training_source, checkpoint_hash)
    config_hash = project_config_sha256(config)

    # All mapping and input validation finishes before any output is created.
    try:
        artifacts.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise MujocoSemanticSuiteError(
            f"semantic suite artifact directory already exists: {artifacts}"
        ) from error
    except OSError as error:
        raise MujocoSemanticSuiteError(
            f"failed to create semantic suite artifact directory: {artifacts}"
        ) from error

    with TemporaryDirectory(prefix="so101-wam-semantic-suite-") as temp_dir:
        case_reports, all_trials, mapping_status_counts = _run_frozen_cases(
            config,
            frozen_cases=frozen_cases,
            snapshot_root=Path(temp_dir),
            artifacts=artifacts,
            checkpoint_source=checkpoint_source,
            training_source=training_source,
            checkpoint_sha256=checkpoint_hash,
            training_report_sha256=training_hash,
            training_identity=training_identity,
            config_sha256=config_hash,
            device=device,
        )

    model_identity = case_reports[0]["mujoco_model_identity"]
    if any(case["mujoco_model_identity"] != model_identity for case in case_reports):
        raise MujocoSemanticSuiteError("semantic cases use different model identities")
    identity_sources = sorted({trial["model_identity_source"] for trial in all_trials})
    success_count = sum(int(trial["success"]) for trial in all_trials)
    trial_count = len(all_trials)
    errors = tuple(
        float(trial["object_position_error_m"])
        for trial in all_trials
        if trial["status"] == SemanticTrialStatus.SCORED.value
    )
    scored_count = len(errors)
    failure_counts = dict(
        sorted(
            Counter(
                str(trial["failure_reason"])
                for trial in all_trials
                if trial["failure_reason"] is not None
            ).items()
        )
    )
    report = {
        "schema_version": SEMANTIC_SUITE_REPORT_SCHEMA,
        "gate": SEMANTIC_SUITE_GATE,
        "result": "complete",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "suite_id": suite.suite_id,
        "case_count": len(case_reports),
        "distinct_object_task_count": len(
            {
                item.object_task_signature_sha256
                for item in mapping.mappings
            }
        ),
        "distinct_object_body_count": len(
            {case["object_body"] for case in case_reports}
        ),
        "distinct_object_physical_profile_count": len(
            {
                case["object_physical_profile_sha256"]
                for case in case_reports
            }
        ),
        "physical_object_diversity_observed": (
            len({case["object_body"] for case in case_reports}) > 1
            and len(
                {
                    case["object_physical_profile_sha256"]
                    for case in case_reports
                }
            )
            > 1
        ),
        "suite_sha256": sha256(suite_source).hexdigest(),
        "mapping_sha256": sha256(mapping_source).hexdigest(),
        "mapping_scope": mapping.scope,
        "mapping_status_counts": dict(sorted(mapping_status_counts.items())),
        "semantic_task_mapping": "local_mapping_manifest",
        "independent_mapping_verified": False,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_id": training_identity.checkpoint_id,
        "training_report_sha256": training_hash,
        "training_evidence_sha256": training_identity.training_evidence_sha256,
        "mujoco_config_sha256": config_hash,
        "mujoco_model_identity": model_identity,
        "model_identity_sources": identity_sources,
        "robot_used": False,
        "semantic_mujoco_object_state_evaluated": scored_count > 0,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "summary": {
            "task_count": len(case_reports),
            "total_trial_count": trial_count,
            "scored_trial_count": scored_count,
            "execution_failure_count": trial_count - scored_count,
            "success_count": success_count,
            "success_rate": success_count / trial_count,
            "success_rate_95ci": list(
                _wilson_interval(success_count, trial_count)
            ),
            "all_cases_passed": all(
                case["result"] == "pass" for case in case_reports
            ),
            "object_position_error_mean_m": None if not errors else fmean(errors),
            "object_position_error_max_m": None if not errors else max(errors),
            "failure_counts": failure_counts,
        },
        "cases": case_reports,
    }
    _write_report(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a multi-case MuJoCo semantic benchmark suite."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MUJOCO_CONFIG_PATH)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        config = ProjectConfig.load(args.config)
        report = run_mujoco_semantic_suite(
            config,
            suite_path=args.suite,
            mapping_path=args.mapping,
            checkpoint_path=args.checkpoint,
            training_report_path=args.training_report,
            artifact_dir=args.artifact_dir,
            report_path=args.report,
            device=args.device,
        )
    except (
        CheckpointError,
        ConfigError,
        DatasetError,
        MujocoAdapterError,
        MujocoBenchmarkError,
        MujocoCLIError,
        MujocoSemanticSuiteError,
        PolicyError,
        RolloutError,
        RuntimeErrorState,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "MIN_SEMANTIC_CASES",
    "OBJECT_TASK_SIGNATURE_SCHEMA",
    "SEMANTIC_MAPPING_SCHEMA",
    "SEMANTIC_MAPPING_SCOPE",
    "SEMANTIC_SUITE_GATE",
    "SEMANTIC_SUITE_REPORT_SCHEMA",
    "SEMANTIC_SUITE_SCHEMA",
    "MujocoSemanticSuiteError",
    "SemanticMapping",
    "SemanticMappingManifest",
    "SemanticSuite",
    "SemanticSuiteCase",
    "load_semantic_mapping_bytes",
    "load_semantic_suite_bytes",
    "main",
    "object_task_signature",
    "run_mujoco_semantic_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
