"""Preregistered prompt-direction checks for MuJoCo prompt controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


DIRECTION_PLAN_SCHEMA = 1
DIRECTION_CLAIM_SCOPE = "preregistered_mujoco_joint_proxy_directionality_only"
PRIMARY_ENDPOINT = "success_rate"
SECONDARY_ENDPOINT = "final_error_mean"
_HASH_FIELDS = (
    "prompt_control_manifest_sha256",
    "benchmark_manifest_sha256",
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
    "negative_error_margin",
    "negative_conditions",
    "descriptive_conditions",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class PromptDirectionError(ValueError):
    """Raised when a prompt-direction plan or evaluation is invalid."""


@dataclass(frozen=True, slots=True)
class DirectionPlan:
    suite_id: str
    hypothesis_id: str
    prompt_control_manifest_sha256: str
    benchmark_manifest_sha256: str
    checkpoint_sha256: str
    mujoco_config_sha256: str
    minimum_trials_per_condition: int
    primary_endpoint: str
    secondary_endpoint: str
    same_task_noninferiority_margin: float
    negative_success_margin: float
    negative_error_margin: float
    negative_conditions: tuple[str, ...]
    descriptive_conditions: tuple[str, ...]


def _json_object(source: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, value in pairs:
            if name in payload:
                raise PromptDirectionError(
                    f"direction plan has duplicate field: {name}"
                )
            payload[name] = value
        return payload

    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PromptDirectionError("direction plan must use UTF-8") from error
    try:
        payload = json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise PromptDirectionError("direction plan must be valid JSON") from error
    if not isinstance(payload, dict):
        raise PromptDirectionError("direction plan must be a JSON object")
    return payload


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptDirectionError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if _SAFE_ID.fullmatch(result) is None:
        raise PromptDirectionError(f"{name} contains unsupported characters")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PromptDirectionError(f"{name} must be a SHA-256 hex string")
    return value


def _margin(value: object, *, name: str, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PromptDirectionError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise PromptDirectionError(f"{name} must be a finite non-negative number")
    if maximum is not None and result > maximum:
        raise PromptDirectionError(f"{name} must be in [0, {maximum:g}]")
    return result


def _condition_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PromptDirectionError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise PromptDirectionError(f"{name} must contain non-empty strings")
    conditions = tuple(str(item).strip() for item in value)
    if len(set(conditions)) != len(conditions):
        raise PromptDirectionError(f"{name} must not contain duplicates")
    return conditions


def _minimum_trials(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 3:
        raise PromptDirectionError("minimum_trials_per_condition must be at least 3")
    return int(value)


def _validate_partition(
    negative_conditions: Sequence[str],
    descriptive_conditions: Sequence[str],
) -> None:
    negative = set(negative_conditions)
    descriptive = set(descriptive_conditions)
    if negative != _NEGATIVE_CONDITIONS:
        raise PromptDirectionError("condition partition is invalid")
    if descriptive != _DESCRIPTIVE_CONDITIONS:
        raise PromptDirectionError("condition partition is invalid")
    if negative & descriptive:
        raise PromptDirectionError("condition partition is invalid")


def load_direction_plan_bytes(
    source: bytes,
    *,
    expected_suite_id: str,
    expected_prompt_control_manifest_sha256: str,
    expected_benchmark_manifest_sha256: str,
    expected_checkpoint_sha256: str,
    expected_mujoco_config_sha256: str,
) -> DirectionPlan:
    payload = _json_object(source)
    unexpected = sorted(set(payload) - _PLAN_FIELDS)
    missing = sorted(_PLAN_FIELDS - set(payload))
    if unexpected:
        raise PromptDirectionError(
            f"direction plan has unexpected fields: {unexpected}"
        )
    if missing:
        raise PromptDirectionError(f"direction plan is missing fields: {missing}")
    if payload.get("schema_version") != DIRECTION_PLAN_SCHEMA:
        raise PromptDirectionError("direction plan requires schema_version=1")
    suite_id = _identifier(payload.get("suite_id"), name="suite_id")
    if suite_id != expected_suite_id:
        raise PromptDirectionError("suite_id mismatch")
    if payload.get("claim_scope") != DIRECTION_CLAIM_SCOPE:
        raise PromptDirectionError("claim_scope mismatch")

    expected_hashes = {
        "prompt_control_manifest_sha256": expected_prompt_control_manifest_sha256,
        "benchmark_manifest_sha256": expected_benchmark_manifest_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "mujoco_config_sha256": expected_mujoco_config_sha256,
    }
    for field in _HASH_FIELDS:
        actual = _sha256(payload.get(field), name=field)
        if actual != expected_hashes[field]:
            raise PromptDirectionError(f"{field} mismatch")

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
        raise PromptDirectionError(f"primary_endpoint must be {PRIMARY_ENDPOINT!r}")
    secondary_endpoint = _string(
        payload.get("secondary_endpoint"),
        name="secondary_endpoint",
    )
    if secondary_endpoint != SECONDARY_ENDPOINT:
        raise PromptDirectionError(f"secondary_endpoint must be {SECONDARY_ENDPOINT!r}")

    return DirectionPlan(
        suite_id=suite_id,
        hypothesis_id=_identifier(
            payload.get("hypothesis_id"),
            name="hypothesis_id",
        ),
        prompt_control_manifest_sha256=expected_hashes[
            "prompt_control_manifest_sha256"
        ],
        benchmark_manifest_sha256=expected_hashes["benchmark_manifest_sha256"],
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
        negative_error_margin=_margin(
            payload.get("negative_error_margin"),
            name="negative_error_margin",
        ),
        negative_conditions=negative_conditions,
        descriptive_conditions=descriptive_conditions,
    )


def load_direction_plan(
    path: str | Path,
    *,
    expected_suite_id: str,
    expected_prompt_control_manifest_sha256: str,
    expected_benchmark_manifest_sha256: str,
    expected_checkpoint_sha256: str,
    expected_mujoco_config_sha256: str,
) -> DirectionPlan:
    return load_direction_plan_bytes(
        Path(path).read_bytes(),
        expected_suite_id=expected_suite_id,
        expected_prompt_control_manifest_sha256=(
            expected_prompt_control_manifest_sha256
        ),
        expected_benchmark_manifest_sha256=expected_benchmark_manifest_sha256,
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
            raise PromptDirectionError("condition name is invalid")
        if name in result:
            raise PromptDirectionError(f"duplicate condition: {name}")
        result[name] = condition
    if set(result) != _ALL_CONDITIONS:
        raise PromptDirectionError("condition set is invalid")
    return result


def _metric(
    condition: Mapping[str, object],
    name: str,
    *,
    maximum: float | None = None,
) -> float:
    value = condition.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PromptDirectionError(f"{name} is invalid")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise PromptDirectionError(f"{name} is invalid")
    if maximum is not None and result > maximum:
        raise PromptDirectionError(f"{name} is invalid")
    return result


def _trial_count(condition: Mapping[str, object]) -> int:
    value = condition.get("trial_count")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PromptDirectionError("trial_count is invalid")
    return int(value)


def _rule(rule_id: str, passed: bool, **extra: object) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "result": "pass" if passed else "fail",
        **extra,
    }


def evaluate_directionality(
    plan: DirectionPlan,
    conditions: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
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
    trials_passed = all(
        _trial_count(condition) >= plan.minimum_trials_per_condition
        for condition in named_conditions
    )

    matched_success = _metric(matched, "success_rate", maximum=1.0)
    same_task_success = _metric(same_task, "success_rate", maximum=1.0)
    same_task_margin = same_task_success - matched_success
    same_task_passed = same_task_margin <= plan.same_task_noninferiority_margin

    negative_success_margins = [
        matched_success - _metric(condition, "success_rate", maximum=1.0)
        for condition in negative
    ]
    negative_success_margin = min(negative_success_margins)
    negative_success_passed = negative_success_margin >= plan.negative_success_margin

    matched_error = _metric(matched, "final_error_mean")
    negative_error_margins = [
        _metric(condition, "final_error_mean") - matched_error for condition in negative
    ]
    negative_error_margin = min(negative_error_margins)
    negative_error_passed = negative_error_margin >= plan.negative_error_margin

    rules = [
        _rule("minimum_trials", trials_passed),
        _rule(
            "same_task_noninferiority",
            same_task_passed,
            observed_margin=same_task_margin,
            required_maximum=plan.same_task_noninferiority_margin,
        ),
        _rule(
            "negative_success_margin",
            negative_success_passed,
            observed_margin=negative_success_margin,
            required_minimum=plan.negative_success_margin,
        ),
        _rule(
            "negative_error_margin",
            negative_error_passed,
            observed_margin=negative_error_margin,
            required_minimum=plan.negative_error_margin,
        ),
    ]
    passed_rule_count = sum(int(rule["result"] == "pass") for rule in rules)

    return {
        "result": "pass" if passed_rule_count == len(rules) else "fail",
        "claim_scope": DIRECTION_CLAIM_SCOPE,
        "hypothesis_id": plan.hypothesis_id,
        "rule_count": len(rules),
        "passed_rule_count": passed_rule_count,
        "statistical_significance_evaluated": False,
        "rules": rules,
        "descriptive_conditions": [
            {
                "condition": name,
                "success_rate": _metric(
                    by_name[name],
                    "success_rate",
                    maximum=1.0,
                ),
                "final_error_mean": _metric(by_name[name], "final_error_mean"),
            }
            for name in plan.descriptive_conditions
        ],
    }


__all__ = [
    "DIRECTION_CLAIM_SCOPE",
    "DIRECTION_PLAN_SCHEMA",
    "DirectionPlan",
    "PromptDirectionError",
    "evaluate_directionality",
    "load_direction_plan",
    "load_direction_plan_bytes",
]
