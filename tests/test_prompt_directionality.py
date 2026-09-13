from __future__ import annotations

import json
from pathlib import Path

import pytest

from so101_wam.prompt_directionality import (
    DIRECTION_CLAIM_SCOPE,
    DIRECTION_PLAN_SCHEMA,
    PromptDirectionError,
    evaluate_directionality,
    load_direction_plan,
)


HASHES = {
    "prompt_control_manifest_sha256": "a" * 64,
    "benchmark_manifest_sha256": "b" * 64,
    "checkpoint_sha256": "c" * 64,
    "mujoco_config_sha256": "d" * 64,
}


def _payload() -> dict[str, object]:
    return {
        "schema_version": DIRECTION_PLAN_SCHEMA,
        "suite_id": "prompt-control-local-v1",
        "hypothesis_id": "joint-proxy-direction-v1",
        "claim_scope": DIRECTION_CLAIM_SCOPE,
        **HASHES,
        "minimum_trials_per_condition": 3,
        "primary_endpoint": "success_rate",
        "secondary_endpoint": "final_error_mean",
        "same_task_noninferiority_margin": 0.34,
        "negative_success_margin": 0.34,
        "negative_error_margin": 0.2,
        "negative_conditions": ["wrong_task", "null", "counterfactual"],
        "descriptive_conditions": [
            "temporal_shuffle",
            "image_frame_shuffle",
        ],
    }


def _write_plan(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "direction-plan.json"
    path.write_text(
        json.dumps(payload or _payload()),
        encoding="utf-8",
    )
    return path


def _conditions() -> list[dict[str, object]]:
    values = {
        "matched": (3, 1.0, 0.1),
        "same_task_alternate": (3, 1.0, 0.12),
        "wrong_task": (3, 0.0, 0.5),
        "temporal_shuffle": (3, 2 / 3, 0.2),
        "image_frame_shuffle": (3, 2 / 3, 0.22),
        "null": (3, 0.0, 0.6),
        "counterfactual": (3, 0.0, 0.55),
    }
    return [
        {
            "condition": condition,
            "trial_count": trial_count,
            "success_rate": success_rate,
            "final_error_mean": final_error_mean,
        }
        for condition, (
            trial_count,
            success_rate,
            final_error_mean,
        ) in values.items()
    ]


def _load(path: Path):
    return load_direction_plan(
        path,
        expected_suite_id="prompt-control-local-v1",
        expected_prompt_control_manifest_sha256=HASHES[
            "prompt_control_manifest_sha256"
        ],
        expected_benchmark_manifest_sha256=HASHES["benchmark_manifest_sha256"],
        expected_checkpoint_sha256=HASHES["checkpoint_sha256"],
        expected_mujoco_config_sha256=HASHES["mujoco_config_sha256"],
    )


def test_direction_plan_evaluates_fixed_joint_proxy_rules(tmp_path: Path) -> None:
    plan = _load(_write_plan(tmp_path))

    evaluation = evaluate_directionality(plan, _conditions())

    assert plan.hypothesis_id == "joint-proxy-direction-v1"
    assert evaluation["result"] == "pass"
    assert evaluation["claim_scope"] == DIRECTION_CLAIM_SCOPE
    assert evaluation["rule_count"] == 4
    assert evaluation["passed_rule_count"] == 4
    assert evaluation["statistical_significance_evaluated"] is False
    rules = {rule["rule_id"]: rule for rule in evaluation["rules"]}
    assert rules["same_task_noninferiority"]["result"] == "pass"
    assert rules["negative_success_margin"]["observed_margin"] == 1.0
    assert rules["negative_error_margin"]["observed_margin"] == 0.4
    assert evaluation["descriptive_conditions"] == [
        {
            "condition": "temporal_shuffle",
            "success_rate": pytest.approx(2 / 3),
            "final_error_mean": 0.2,
        },
        {
            "condition": "image_frame_shuffle",
            "success_rate": pytest.approx(2 / 3),
            "final_error_mean": 0.22,
        },
    ]


def test_direction_plan_preserves_a_preregistered_negative_result(
    tmp_path: Path,
) -> None:
    conditions = _conditions()
    wrong_task = next(item for item in conditions if item["condition"] == "wrong_task")
    wrong_task["success_rate"] = 2 / 3
    wrong_task["final_error_mean"] = 0.15

    evaluation = evaluate_directionality(
        _load(_write_plan(tmp_path)),
        conditions,
    )

    assert evaluation["result"] == "fail"
    assert evaluation["passed_rule_count"] == 2
    rules = {rule["rule_id"]: rule for rule in evaluation["rules"]}
    assert rules["negative_success_margin"]["result"] == "fail"
    assert rules["negative_error_margin"]["result"] == "fail"


def test_direction_plan_rejects_failed_same_task_noninferiority(
    tmp_path: Path,
) -> None:
    conditions = _conditions()
    matched = next(item for item in conditions if item["condition"] == "matched")
    matched["success_rate"] = 0.0

    evaluation = evaluate_directionality(
        _load(_write_plan(tmp_path)),
        conditions,
    )

    rules = {rule["rule_id"]: rule for rule in evaluation["rules"]}
    assert rules["same_task_noninferiority"]["result"] == "fail"
    assert rules["same_task_noninferiority"]["observed_margin"] == 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("suite", "suite_id"),
        ("binding", "checkpoint_sha256"),
        ("partition", "partition"),
        ("trials", "minimum_trials"),
        ("margin", "negative_success_margin"),
        ("endpoint", "primary_endpoint"),
        ("extra", "unexpected fields"),
    ),
)
def test_direction_plan_rejects_invalid_or_unbound_hypotheses(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    payload = _payload()
    if mutation == "suite":
        payload["suite_id"] = "different-suite"
    elif mutation == "binding":
        payload["checkpoint_sha256"] = "e" * 64
    elif mutation == "partition":
        payload["descriptive_conditions"] = ["temporal_shuffle"]
    elif mutation == "trials":
        payload["minimum_trials_per_condition"] = 2
    elif mutation == "endpoint":
        payload["primary_endpoint"] = "final_error_max"
    elif mutation == "extra":
        payload["post_hoc_override"] = True
    else:
        payload["negative_success_margin"] = 1.1

    with pytest.raises(PromptDirectionError, match=message):
        _load(_write_plan(tmp_path, payload))


def test_direction_plan_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    payload = json.dumps(_payload())
    payload = payload.replace(
        '"negative_success_margin": 0.34,',
        '"negative_success_margin": 0.1, "negative_success_margin": 0.34,',
    )
    path = tmp_path / "duplicate-direction-plan.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(PromptDirectionError, match="duplicate field"):
        _load(path)
