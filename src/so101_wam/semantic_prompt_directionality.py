"""Preregistered direction checks for semantic MuJoCo prompt controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SEMANTIC_DIRECTION_PLAN_SCHEMA = 1
SEMANTIC_DIRECTION_CLAIM_SCOPE = (
    "preregistered_mujoco_object_state_directionality_only"
)
PRIMARY_ENDPOINT = "success_rate"
SECONDARY_ENDPOINT = "object_position_error_mean_m"
_HASH_FIELDS = (
    "prompt_control_manifest_sha256",
    "semantic_manifest_sha256",
    "training_report_sha256",
    "checkpoint_sha256",
    "mujoco_config_sha256",
)
_NEGATIVE_CONDITIONS = {"wrong_task", "null", "counterfactual"}
_DESCRIPTIVE_CONDITIONS = {"temporal_shuffle", "image_frame_shuffle"}
_SAME_TASK_CONDITION = "same_task_alternate"
_MATCHED_CONDITION = "matched"
_ALL_CONDITIONS = {
    _MATCHED_CONDITION,
    _SAME_TASK_CONDITION,
    *_NEGATIVE_CONDITIONS,
    *_DESCRIPTIVE_CONDITIONS,
}
_PLAN_FIELDS = {
    "schema_version",
    "suite_id",
    "hypothesis_id",
    "claim_scope",
    *_HASH_FIELDS,
    "minimum_trials_per_condition",
    "primary_endpoint",
    "secondary_endpoint",
    "same_task_noninferiority_margin",
    "negative_success_margin",
    "negative_object_position_error_margin_m",
    "negative_conditions",
    "descriptive_conditions",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SemanticPromptDirectionError(ValueError):
    """Raised when a semantic direction plan or evaluation is invalid."""


@dataclass(frozen=True, slots=True)
class SemanticDirectionPlan:
    suite_id: str
    hypothesis_id: str
    prompt_control_manifest_sha256: str
    semantic_manifest_sha256: str
    training_report_sha256: str
    checkpoint_sha256: str
    mujoco_config_sha256: str
    minimum_trials_per_condition: int
    primary_endpoint: str
    secondary_endpoint: str
    same_task_noninferiority_margin: float
    negative_success_margin: float
    negative_object_position_error_margin_m: float
    negative_conditions: tuple[str, ...]
    descriptive_conditions: tuple[str, ...]


def _json_object(source: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, value in pairs:
            if name in payload:
                raise SemanticPromptDirectionError(
                    f"semantic direction plan has duplicate field: {name}"
                )
            payload[name] = value
        return payload

    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SemanticPromptDirectionError(
            "semantic direction plan must use UTF-8"
        ) from error
    try:
        payload = json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise SemanticPromptDirectionError(
            "semantic direction plan must be valid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise SemanticPromptDirectionError(
            "semantic direction plan must be a JSON object"
        )
    return payload


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticPromptDirectionError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if _SAFE_ID.fullmatch(result) is None:
        raise SemanticPromptDirectionError(
            f"{name} contains unsupported characters"
        )
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticPromptDirectionError(
            f"{name} must be a SHA-256 hex string"
        )
    return value


def _margin(value: object, *, name: str, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SemanticPromptDirectionError(
            f"{name} must be a finite non-negative number"
        )
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise SemanticPromptDirectionError(
            f"{name} must be a finite non-negative number"
        )
    if maximum is not None and result > maximum:
        raise SemanticPromptDirectionError(f"{name} must be in [0, {maximum:g}]")
    return result


def _condition_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SemanticPromptDirectionError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise SemanticPromptDirectionError(
            f"{name} must contain non-empty strings"
        )
    conditions = tuple(str(item).strip() for item in value)
    if len(set(conditions)) != len(conditions):
        raise SemanticPromptDirectionError(f"{name} must not contain duplicates")
    return conditions


def _minimum_trials(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 3:
        raise SemanticPromptDirectionError(
            "minimum_trials_per_condition must be at least 3"
        )
    return int(value)


def _validate_partition(
    negative_conditions: Sequence[str],
    descriptive_conditions: Sequence[str],
) -> None:
    negative = set(negative_conditions)
    descriptive = set(descriptive_conditions)
    if negative != _NEGATIVE_CONDITIONS:
        raise SemanticPromptDirectionError("condition partition is invalid")
    if descriptive != _DESCRIPTIVE_CONDITIONS:
        raise SemanticPromptDirectionError("condition partition is invalid")
    if negative & descriptive:
        raise SemanticPromptDirectionError("condition partition is invalid")


def load_semantic_direction_plan_bytes(
    source: bytes,
    *,
    expected_suite_id: str,
    expected_prompt_control_manifest_sha256: str,
    expected_semantic_manifest_sha256: str,
    expected_training_report_sha256: str,
    expected_checkpoint_sha256: str,
    expected_mujoco_config_sha256: str,
) -> SemanticDirectionPlan:
    """Load a strict plan bound to the exact semantic suite inputs."""

    payload = _json_object(source)
    unexpected = sorted(set(payload) - _PLAN_FIELDS)
    missing = sorted(_PLAN_FIELDS - set(payload))
    if unexpected:
        raise SemanticPromptDirectionError(
            f"semantic direction plan has unexpected fields: {unexpected}"
        )
    if missing:
        raise SemanticPromptDirectionError(
            f"semantic direction plan is missing fields: {missing}"
        )
    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SEMANTIC_DIRECTION_PLAN_SCHEMA
    ):
        raise SemanticPromptDirectionError(
            "semantic direction plan requires schema_version=1"
        )

    suite_id = _identifier(payload.get("suite_id"), name="suite_id")
    if suite_id != expected_suite_id:
        raise SemanticPromptDirectionError("suite_id mismatch")
    if payload.get("claim_scope") != SEMANTIC_DIRECTION_CLAIM_SCOPE:
        raise SemanticPromptDirectionError("claim_scope mismatch")

    expected_hashes = {
        "prompt_control_manifest_sha256": (
            expected_prompt_control_manifest_sha256
        ),
        "semantic_manifest_sha256": expected_semantic_manifest_sha256,
        "training_report_sha256": expected_training_report_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "mujoco_config_sha256": expected_mujoco_config_sha256,
    }
    for field in _HASH_FIELDS:
        actual = _sha256(payload.get(field), name=field)
        if actual != expected_hashes[field]:
            raise SemanticPromptDirectionError(f"{field} mismatch")

    negative_conditions = _condition_tuple(
        payload.get("negative_conditions"),
        name="negative_conditions",
    )
    descriptive_conditions = _condition_tuple(
        payload.get("descriptive_conditions"),
        name="descriptive_conditions",
    )
    _validate_partition(negative_conditions, descriptive_conditions)

    primary_endpoint = _string(
        payload.get("primary_endpoint"),
        name="primary_endpoint",
    )
    if primary_endpoint != PRIMARY_ENDPOINT:
        raise SemanticPromptDirectionError(
            f"primary_endpoint must be {PRIMARY_ENDPOINT!r}"
        )
    secondary_endpoint = _string(
        payload.get("secondary_endpoint"),
        name="secondary_endpoint",
    )
    if secondary_endpoint != SECONDARY_ENDPOINT:
        raise SemanticPromptDirectionError(
            f"secondary_endpoint must be {SECONDARY_ENDPOINT!r}"
        )

    return SemanticDirectionPlan(
        suite_id=suite_id,
        hypothesis_id=_identifier(
            payload.get("hypothesis_id"),
            name="hypothesis_id",
        ),
        prompt_control_manifest_sha256=expected_hashes[
            "prompt_control_manifest_sha256"
        ],
        semantic_manifest_sha256=expected_hashes["semantic_manifest_sha256"],
        training_report_sha256=expected_hashes["training_report_sha256"],
        checkpoint_sha256=expected_hashes["checkpoint_sha256"],
        mujoco_config_sha256=expected_hashes["mujoco_config_sha256"],
        minimum_trials_per_condition=_minimum_trials(
            payload.get("minimum_trials_per_condition")
        ),
        primary_endpoint=primary_endpoint,
        secondary_endpoint=secondary_endpoint,
        same_task_noninferiority_margin=_margin(
            payload.get("same_task_noninferiority_margin"),
            name="same_task_noninferiority_margin",
            maximum=1.0,
        ),
        negative_success_margin=_margin(
            payload.get("negative_success_margin"),
            name="negative_success_margin",
            maximum=1.0,
        ),
        negative_object_position_error_margin_m=_margin(
            payload.get("negative_object_position_error_margin_m"),
            name="negative_object_position_error_margin_m",
        ),
        negative_conditions=negative_conditions,
        descriptive_conditions=descriptive_conditions,
    )


def load_semantic_direction_plan(
    path: str | Path,
    *,
    expected_suite_id: str,
    expected_prompt_control_manifest_sha256: str,
    expected_semantic_manifest_sha256: str,
    expected_training_report_sha256: str,
    expected_checkpoint_sha256: str,
    expected_mujoco_config_sha256: str,
) -> SemanticDirectionPlan:
    return load_semantic_direction_plan_bytes(
        Path(path).read_bytes(),
        expected_suite_id=expected_suite_id,
        expected_prompt_control_manifest_sha256=(
            expected_prompt_control_manifest_sha256
        ),
        expected_semantic_manifest_sha256=expected_semantic_manifest_sha256,
        expected_training_report_sha256=expected_training_report_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_mujoco_config_sha256=expected_mujoco_config_sha256,
    )


def _conditions_by_name(
    conditions: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for condition in conditions:
        name = condition.get("condition")
        if not isinstance(name, str) or not name:
            raise SemanticPromptDirectionError("condition name is invalid")
        if name in result:
            raise SemanticPromptDirectionError(f"duplicate condition: {name}")
        result[name] = condition
    if set(result) != _ALL_CONDITIONS:
        raise SemanticPromptDirectionError("condition set is invalid")
    return result


def _metric(
    condition: Mapping[str, object],
    name: str,
    *,
    maximum: float | None = None,
) -> float:
    value = condition.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SemanticPromptDirectionError(f"{name} is invalid")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise SemanticPromptDirectionError(f"{name} is invalid")
    if maximum is not None and result > maximum:
        raise SemanticPromptDirectionError(f"{name} is invalid")
    return result


def _trial_count(condition: Mapping[str, object]) -> int:
    value = condition.get("trial_count")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SemanticPromptDirectionError("trial_count is invalid")
    return int(value)


def _optional_metric(
    condition: Mapping[str, object],
    name: str,
    *,
    maximum: float | None = None,
) -> float | None:
    if condition.get(name) is None:
        return None
    return _metric(condition, name, maximum=maximum)


def _rule(rule_id: str, passed: bool, **extra: object) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "result": "pass" if passed else "fail",
        **extra,
    }


def evaluate_semantic_directionality(
    plan: SemanticDirectionPlan,
    conditions: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Evaluate the four preregistered object-state direction rules."""

    by_name = _conditions_by_name(conditions)
    matched = by_name[_MATCHED_CONDITION]
    same_task = by_name[_SAME_TASK_CONDITION]
    negative = [by_name[name] for name in plan.negative_conditions]
    named_conditions = [
        by_name[_MATCHED_CONDITION],
        by_name[_SAME_TASK_CONDITION],
        *negative,
        *(by_name[name] for name in plan.descriptive_conditions),
    ]

    trial_counts = [_trial_count(condition) for condition in named_conditions]
    observed_minimum_trials = min(trial_counts)
    trials_passed = (
        observed_minimum_trials >= plan.minimum_trials_per_condition
    )

    matched_success = _metric(matched, PRIMARY_ENDPOINT, maximum=1.0)
    same_task_success = _metric(same_task, PRIMARY_ENDPOINT, maximum=1.0)
    same_task_margin = same_task_success - matched_success
    same_task_passed = same_task_margin <= plan.same_task_noninferiority_margin

    negative_success = [
        (
            name,
            _metric(by_name[name], PRIMARY_ENDPOINT, maximum=1.0),
        )
        for name in plan.negative_conditions
    ]
    best_negative_name, best_negative_success = max(
        negative_success,
        key=lambda item: item[1],
    )
    observed_success_margin = matched_success - best_negative_success
    negative_success_passed = (
        observed_success_margin >= plan.negative_success_margin
    )

    matched_error = _optional_metric(matched, SECONDARY_ENDPOINT)
    negative_errors = [
        (name, _optional_metric(by_name[name], SECONDARY_ENDPOINT))
        for name in plan.negative_conditions
    ]
    error_endpoint_available = matched_error is not None and all(
        value is not None for _, value in negative_errors
    )
    closest_negative_name: str | None = None
    observed_error_margin: float | None = None
    negative_error_passed = False
    if error_endpoint_available and matched_error is not None:
        negative_error_margins = [
            (name, float(value) - matched_error)
            for name, value in negative_errors
            if value is not None
        ]
        closest_negative_name, observed_error_margin = min(
            negative_error_margins,
            key=lambda item: item[1],
        )
        negative_error_passed = (
            observed_error_margin
            >= plan.negative_object_position_error_margin_m
        )

    rules = [
        _rule(
            "minimum_trials",
            trials_passed,
            observed_minimum=observed_minimum_trials,
            required_minimum=plan.minimum_trials_per_condition,
        ),
        _rule(
            "same_task_noninferiority",
            same_task_passed,
            observed_margin=same_task_margin,
            required_maximum=plan.same_task_noninferiority_margin,
        ),
        _rule(
            "negative_success_margin",
            negative_success_passed,
            observed_margin=observed_success_margin,
            required_minimum=plan.negative_success_margin,
            best_negative_condition=best_negative_name,
        ),
        _rule(
            "negative_object_position_error_margin",
            negative_error_passed,
            endpoint_available=error_endpoint_available,
            observed_margin_m=observed_error_margin,
            required_minimum_m=(
                plan.negative_object_position_error_margin_m
            ),
            closest_negative_condition=closest_negative_name,
        ),
    ]
    passed_rule_count = sum(int(rule["result"] == "pass") for rule in rules)

    return {
        "result": "pass" if passed_rule_count == len(rules) else "fail",
        "claim_scope": SEMANTIC_DIRECTION_CLAIM_SCOPE,
        "hypothesis_id": plan.hypothesis_id,
        "primary_endpoint": plan.primary_endpoint,
        "secondary_endpoint": plan.secondary_endpoint,
        "rule_count": len(rules),
        "passed_rule_count": passed_rule_count,
        "statistical_significance_evaluated": False,
        "rules": rules,
        "descriptive_conditions": [
            {
                "condition": name,
                "success_rate": _metric(
                    by_name[name],
                    PRIMARY_ENDPOINT,
                    maximum=1.0,
                ),
                "object_position_error_mean_m": _optional_metric(
                    by_name[name], SECONDARY_ENDPOINT
                ),
            }
            for name in plan.descriptive_conditions
        ],
    }


__all__ = [
    "SEMANTIC_DIRECTION_CLAIM_SCOPE",
    "SEMANTIC_DIRECTION_PLAN_SCHEMA",
    "SemanticDirectionPlan",
    "SemanticPromptDirectionError",
    "evaluate_semantic_directionality",
    "load_semantic_direction_plan",
    "load_semantic_direction_plan_bytes",
]
