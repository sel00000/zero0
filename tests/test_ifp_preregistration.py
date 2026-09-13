from __future__ import annotations

import json

import pytest

from so101_wam.ifp_preregistration import (
    IFP_PREREGISTRATION_CLAIM_SCOPE,
    IFPPreregistrationError,
    evaluate_ifp_endpoints,
    load_ifp_preregistration_bytes,
)


OPTIMIZER_PROTOCOL = {
    "policy_hz": 10.0,
    "servo_hz": 50.0,
    "latent_dim": 32,
    "transformer_layers": 1,
    "transformer_heads": 4,
    "future_steps": 3,
    "action_horizon": 10,
    "action_history_steps": 4,
    "ifp_stride": 2,
    "ifp_architecture": "fused_modules",
    "ifp_window_steps": 4,
    "max_context_steps": 300,
    "stage1_steps": 100,
    "stage2_steps": 500,
    "sampling_strategy": "task_balanced",
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "future_latent_weight": 1.0,
    "action_weight": 1.0,
    "ifp_weight": 0.25,
    "device": "cpu",
    "optimizer": "AdamW",
}


def _plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "hypothesis_id": "ifp-k4-local-v1",
        "claim_scope": IFP_PREREGISTRATION_CLAIM_SCOPE,
        "seeds": [3, 7, 11],
        "train_split_sha256": "a" * 64,
        "validation_split_sha256": "b" * 64,
        "variant_set": [0, 2, 4],
        "optimizer_protocol": OPTIMIZER_PROTOCOL,
        "closed_loop": {
            "scope": "synthetic_terminal_joint_proxy",
            "minimum_trials_per_seed_variant": 3,
        },
        "endpoints": {
            "primary": "closed_loop.success_rate",
            "secondary": "closed_loop.terminal_error_mean",
        },
        "margins": {
            "primary_min_delta_k4_vs_k0": 0.2,
            "secondary_max_delta_k4_vs_k0": -0.1,
        },
    }


def _variants(*, k4_success: float, k4_error: float) -> list[dict[str, object]]:
    return [
        {
            "ifp_steps": 0,
            "closed_loop": {
                "trial_count": 9,
                "success_rate": 0.4,
                "terminal_error_mean": 1.0,
            },
        },
        {
            "ifp_steps": 2,
            "closed_loop": {
                "trial_count": 9,
                "success_rate": 0.5,
                "terminal_error_mean": 0.8,
            },
        },
        {
            "ifp_steps": 4,
            "closed_loop": {
                "trial_count": 9,
                "success_rate": k4_success,
                "terminal_error_mean": k4_error,
            },
        },
    ]


def test_ifp_preregistration_loads_pre_result_contract() -> None:
    plan = load_ifp_preregistration_bytes(json.dumps(_plan()).encode("utf-8"))

    assert plan.hypothesis_id == "ifp-k4-local-v1"
    assert plan.seeds == (3, 7, 11)
    assert plan.variant_set == (0, 2, 4)
    assert plan.optimizer_protocol == OPTIMIZER_PROTOCOL
    assert plan.minimum_trials_per_seed_variant == 3
    assert plan.primary_min_delta_k4_vs_k0 == 0.2
    assert plan.secondary_max_delta_k4_vs_k0 == -0.1


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("unknown", "unexpected fields"),
        ("result_hash", "unexpected fields"),
        ("seeds", "seeds"),
        ("variants", "variant_set"),
        ("optimizer", "optimizer_protocol"),
        ("endpoint", "primary endpoint"),
        ("scope", "claim_scope"),
    ),
)
def test_ifp_preregistration_rejects_invalid_contracts(
    mutation: str,
    message: str,
) -> None:
    payload = _plan()
    if mutation == "unknown":
        payload["unknown"] = True
    elif mutation == "result_hash":
        payload["source_report_sha256s"] = ["c" * 64]
    elif mutation == "seeds":
        payload["seeds"] = [3, 3, 11]
    elif mutation == "variants":
        payload["variant_set"] = [0, 4]
    elif mutation == "optimizer":
        protocol = dict(OPTIMIZER_PROTOCOL)
        protocol["stage2_final_total_loss"] = 0.1
        payload["optimizer_protocol"] = protocol
    elif mutation == "endpoint":
        payload["endpoints"] = {
            "primary": "validation.action_mse_normalized",
            "secondary": "closed_loop.terminal_error_mean",
        }
    elif mutation == "scope":
        payload["claim_scope"] = "official_ifp_effect"

    with pytest.raises(IFPPreregistrationError, match=message):
        load_ifp_preregistration_bytes(json.dumps(payload).encode("utf-8"))


def test_ifp_preregistration_rejects_duplicate_fields() -> None:
    source = b'{"schema_version":1,"schema_version":1}'

    with pytest.raises(IFPPreregistrationError, match="duplicate field"):
        load_ifp_preregistration_bytes(source)


def test_ifp_endpoint_evaluation_records_pass_and_fail() -> None:
    plan = load_ifp_preregistration_bytes(json.dumps(_plan()).encode("utf-8"))

    passed = evaluate_ifp_endpoints(
        plan,
        _variants(k4_success=0.7, k4_error=0.7),
    )
    failed = evaluate_ifp_endpoints(
        plan,
        _variants(k4_success=0.5, k4_error=1.1),
    )

    assert passed == {
        "passed": True,
        "primary_endpoint": "closed_loop.success_rate",
        "primary_delta_k4_vs_k0": pytest.approx(0.3),
        "primary_required_minimum": 0.2,
        "primary_passed": True,
        "secondary_endpoint": "closed_loop.terminal_error_mean",
        "secondary_endpoint_available": True,
        "secondary_delta_k4_vs_k0": pytest.approx(-0.3),
        "secondary_required_maximum": -0.1,
        "secondary_passed": True,
        "statistical_significance_evaluated": False,
    }
    assert failed["passed"] is False
    assert failed["primary_passed"] is False
    assert failed["secondary_passed"] is False


def test_unavailable_ifp_endpoint_fails() -> None:
    plan = load_ifp_preregistration_bytes(json.dumps(_plan()).encode("utf-8"))
    variants = _variants(k4_success=0.7, k4_error=0.0)
    variants[-1]["closed_loop"]["terminal_error_mean"] = None

    result = evaluate_ifp_endpoints(plan, variants)

    assert result["primary_passed"] is True
    assert result["secondary_endpoint_available"] is False
    assert result["secondary_delta_k4_vs_k0"] is None
    assert result["secondary_passed"] is result["passed"] is False


@pytest.mark.parametrize("invalid", [True, float("nan"), "invalid", -1.0])
def test_null_endpoint_keeps_validation(invalid: object) -> None:
    plan = load_ifp_preregistration_bytes(json.dumps(_plan()).encode("utf-8"))
    variants = _variants(k4_success=0.7, k4_error=0.0)
    variants[0]["closed_loop"]["terminal_error_mean"] = None
    variants[-1]["closed_loop"]["terminal_error_mean"] = invalid

    with pytest.raises(IFPPreregistrationError, match="terminal_error_mean"):
        evaluate_ifp_endpoints(plan, variants)


def test_missing_endpoint_is_not_null() -> None:
    plan = load_ifp_preregistration_bytes(json.dumps(_plan()).encode("utf-8"))
    variants = _variants(k4_success=0.7, k4_error=0.0)
    variants[-1]["closed_loop"].pop("terminal_error_mean")

    with pytest.raises(IFPPreregistrationError, match="terminal_error_mean"):
        evaluate_ifp_endpoints(plan, variants)
