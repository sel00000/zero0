from __future__ import annotations

import json
from pathlib import Path

import pytest

from so101_wam.semantic_prompt_directionality import (
    SEMANTIC_DIRECTION_CLAIM_SCOPE,
    SEMANTIC_DIRECTION_PLAN_SCHEMA,
    SemanticPromptDirectionError,
    evaluate_semantic_directionality,
    load_semantic_direction_plan,
)


HASHES = {
    "prompt_control_manifest_sha256": "a" * 64,
    "semantic_manifest_sha256": "b" * 64,
    "training_report_sha256": "c" * 64,
    "checkpoint_sha256": "d" * 64,
    "mujoco_config_sha256": "e" * 64,
}


def _payload() -> dict[str, object]:
    return {
        "schema_version": SEMANTIC_DIRECTION_PLAN_SCHEMA,
        "suite_id": "semantic-prompt-control-test",
        "hypothesis_id": "object-state-direction-v1",
        "claim_scope": SEMANTIC_DIRECTION_CLAIM_SCOPE,
        **HASHES,
        "minimum_trials_per_condition": 3,
        "primary_endpoint": "success_rate",
        "secondary_endpoint": "object_position_error_mean_m",
        "same_task_noninferiority_margin": 0.34,
        "negative_success_margin": 0.34,
        "negative_object_position_error_margin_m": 0.05,
        "negative_conditions": ["wrong_task", "null", "counterfactual"],
        "descriptive_conditions": [
            "temporal_shuffle",
            "image_frame_shuffle",
        ],
    }


def _write_plan(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "semantic-direction-plan.json"
    path.write_text(json.dumps(payload or _payload()), encoding="utf-8")
    return path


def _conditions() -> list[dict[str, object]]:
    values = {
        "matched": (3, 1.0, 0.020),
        "same_task_alternate": (3, 1.0, 0.024),
        "wrong_task": (3, 0.0, 0.090),
        "temporal_shuffle": (3, 2 / 3, 0.050),
        "image_frame_shuffle": (3, 2 / 3, 0.055),
        "null": (3, 0.0, 0.120),
        "counterfactual": (3, 0.0, 0.110),
    }
    return [
        {
            "condition": condition,
            "trial_count": trial_count,
            "success_rate": success_rate,
            "object_position_error_mean_m": error_mean,
        }
        for condition, (trial_count, success_rate, error_mean) in values.items()
    ]


def _load(path: Path):
    return load_semantic_direction_plan(
        path,
        expected_suite_id="semantic-prompt-control-test",
        expected_prompt_control_manifest_sha256=HASHES[
            "prompt_control_manifest_sha256"
        ],
        expected_semantic_manifest_sha256=HASHES["semantic_manifest_sha256"],
        expected_training_report_sha256=HASHES["training_report_sha256"],
        expected_checkpoint_sha256=HASHES["checkpoint_sha256"],
        expected_mujoco_config_sha256=HASHES["mujoco_config_sha256"],
    )


def test_semantic_direction_plan_evaluates_fixed_object_state_rules(
    tmp_path: Path,
) -> None:
    plan = _load(_write_plan(tmp_path))

    evaluation = evaluate_semantic_directionality(plan, _conditions())

    assert plan.hypothesis_id == "object-state-direction-v1"
    assert evaluation["result"] == "pass"
    assert evaluation["claim_scope"] == SEMANTIC_DIRECTION_CLAIM_SCOPE
    assert evaluation["rule_count"] == 4
    assert evaluation["passed_rule_count"] == 4
    assert evaluation["statistical_significance_evaluated"] is False
    rules = {rule["rule_id"]: rule for rule in evaluation["rules"]}
    assert rules["same_task_noninferiority"]["result"] == "pass"
    assert rules["negative_success_margin"]["observed_margin"] == 1.0
    assert (
        rules["negative_object_position_error_margin"]["observed_margin_m"]
        == pytest.approx(0.07)
    )
    assert evaluation["descriptive_conditions"] == [
        {
            "condition": "temporal_shuffle",
            "success_rate": pytest.approx(2 / 3),
            "object_position_error_mean_m": 0.050,
        },
        {
            "condition": "image_frame_shuffle",
            "success_rate": pytest.approx(2 / 3),
            "object_position_error_mean_m": 0.055,
        },
    ]


def test_semantic_direction_plan_preserves_preregistered_negative_fail(
    tmp_path: Path,
) -> None:
    conditions = _conditions()
    wrong_task = next(item for item in conditions if item["condition"] == "wrong_task")
    wrong_task["success_rate"] = 2 / 3
    wrong_task["object_position_error_mean_m"] = 0.025

    evaluation = evaluate_semantic_directionality(
        _load(_write_plan(tmp_path)),
        conditions,
    )

    assert evaluation["result"] == "fail"
    assert evaluation["passed_rule_count"] == 2
    rules = {rule["rule_id"]: rule for rule in evaluation["rules"]}
    assert rules["negative_success_margin"]["result"] == "fail"
    assert rules["negative_object_position_error_margin"]["result"] == "fail"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", True, "schema_version"),
        ("schema_version", 1.0, "schema_version"),
        ("suite_id", "different-suite", "suite_id"),
        ("prompt_control_manifest_sha256", "f" * 64, "prompt_control"),
        ("semantic_manifest_sha256", "f" * 64, "semantic_manifest"),
        ("training_report_sha256", "f" * 64, "training_report"),
        ("checkpoint_sha256", "f" * 64, "checkpoint_sha256"),
        ("mujoco_config_sha256", "f" * 64, "mujoco_config"),
        ("primary_endpoint", "all_trials_successful", "primary_endpoint"),
        ("secondary_endpoint", "object_position_error_max_m", "secondary_endpoint"),
        ("minimum_trials_per_condition", 2, "minimum_trials"),
        ("negative_conditions", ["wrong_task", "null"], "partition"),
    ),
)
def test_semantic_direction_plan_rejects_invalid_or_unbound_hypotheses(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(SemanticPromptDirectionError, match=message):
        _load(_write_plan(tmp_path, payload))


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_semantic_direction_plan_requires_exact_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _payload()
    if mutation == "missing":
        del payload["hypothesis_id"]
    else:
        payload["post_hoc_override"] = True

    with pytest.raises(SemanticPromptDirectionError, match="fields"):
        _load(_write_plan(tmp_path, payload))


def test_semantic_direction_plan_rejects_duplicate_json_fields(
    tmp_path: Path,
) -> None:
    payload = json.dumps(_payload()).replace(
        '"negative_success_margin": 0.34,',
        '"negative_success_margin": 0.1, "negative_success_margin": 0.34,',
    )
    path = tmp_path / "duplicate-semantic-direction-plan.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(SemanticPromptDirectionError, match="duplicate field"):
        _load(path)
