from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from so101_wam.ifp_preregistration import IFP_PREREGISTRATION_CLAIM_SCOPE
from so101_wam.ifp_study import (
    IFPStudyError,
    aggregate_ifp_study,
    main as ifp_study_main,
)
from so101_wam.model import ActionDecoder


COMPARABILITY = {
    "same_closed_loop_protocol": True,
    "same_initial_model_state": True,
    "same_optimizer_budget": True,
    "same_seed": True,
    "same_train_split": True,
    "same_training_schedule": True,
    "same_validation_split": True,
    "same_window_count": True,
}

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
    "stage1_steps": 10,
    "stage2_steps": 20,
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


def _write_report(
    root: Path,
    *,
    seed: int,
    train_split: str = "a" * 64,
    stage2_steps: int = 20,
    metric_extra: bool = False,
    variant_steps: tuple[int, ...] = (0, 2, 4),
    k4_favorable: bool = False,
    action_decoder: str | None = None,
) -> Path:
    variants = []
    for ifp_steps in variant_steps:
        validation = {
            "action_mse_normalized": float(seed) + ifp_steps / 10.0,
            "future_latent_mse": ifp_steps + seed / 100.0,
        }
        if metric_extra and ifp_steps == 4:
            validation["unexpected_metric"] = 1.0
        if k4_favorable:
            success_count = 2 * int(ifp_steps == 4)
            terminal_error = 0.5 if ifp_steps == 4 else 1.0
        else:
            success_count = int(ifp_steps == 4)
            terminal_error = float(seed + ifp_steps)
        optimization = dict(OPTIMIZER_PROTOCOL)
        optimization.update(
            {
                "seed": seed,
                "ifp_steps": ifp_steps,
                "stage2_steps": stage2_steps,
                "stage2_final_total_loss": seed + ifp_steps,
            }
        )
        if action_decoder is not None:
            optimization["action_decoder"] = action_decoder
        variants.append(
            {
                "ifp_steps": ifp_steps,
                "data": {
                    "train_split_sha256": train_split,
                    "validation_split_sha256": "b" * 64,
                    "train_window_count": 12,
                    "validation_window_count": 6,
                    "train_task_count": 2,
                    "validation_task_count": 1,
                    "sampling_audit": {
                        "seed": seed,
                        "schedule_sha256": f"{seed:064x}",
                    },
                },
                "optimization": optimization,
                "validation": validation,
                "inference": {
                    "checkpoint_ifp_steps": 0,
                    "ifp_module_present": False,
                },
                "closed_loop": {
                    "scope": "synthetic_terminal_joint_proxy",
                    "protocol_sha256": "d" * 64,
                    "trial_count": 2,
                    "scored_trial_count": 2,
                    "execution_failure_count": 0,
                    "success_count": success_count,
                    "success_rate": success_count / 2.0,
                    "terminal_error_mean": terminal_error,
                    "terminal_error_basis": "max_abs_joint_position_error",
                    "failure_counts": (
                        {}
                        if success_count == 2
                        else {
                            "terminal_joint_target_tolerance": 2 - success_count,
                        }
                    ),
                    "artifact": None,
                    "artifact_sha256": None,
                },
            }
        )

    payload = {
        "schema_version": "so101_wam.ifp_ablation.v4",
        "result": "complete",
        "evidence_level": "offline_and_simulation_diagnostic",
        "robot_used": False,
        "scientific_claim": "not_evaluated",
        "comparability": COMPARABILITY,
        "variants": variants,
    }
    path = root / f"seed-{seed}" / "ifp_ablation.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_plan(
    root: Path,
    *,
    seeds: tuple[int, ...] = (3, 7, 11),
    train_split: str = "a" * 64,
    stage2_steps: int = 20,
    scope: str = "synthetic_terminal_joint_proxy",
    minimum_trials: int = 2,
    primary_margin: float = 0.5,
    secondary_margin: float = -0.25,
) -> Path:
    optimizer_protocol = dict(OPTIMIZER_PROTOCOL)
    optimizer_protocol["stage2_steps"] = stage2_steps
    payload = {
        "schema_version": 1,
        "hypothesis_id": "ifp-k4-local-v1",
        "claim_scope": IFP_PREREGISTRATION_CLAIM_SCOPE,
        "seeds": list(seeds),
        "train_split_sha256": train_split,
        "validation_split_sha256": "b" * 64,
        "variant_set": [0, 2, 4],
        "optimizer_protocol": optimizer_protocol,
        "closed_loop": {
            "scope": scope,
            "minimum_trials_per_seed_variant": minimum_trials,
        },
        "endpoints": {
            "primary": "closed_loop.success_rate",
            "secondary": "closed_loop.terminal_error_mean",
        },
        "margins": {
            "primary_min_delta_k4_vs_k0": primary_margin,
            "secondary_max_delta_k4_vs_k0": secondary_margin,
        },
    }
    path = root / "ifp_study_preregistration.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_ifp_study_aggregates_three_comparable_seeds(tmp_path: Path) -> None:
    reports = tuple(
        _write_report(tmp_path, seed=seed)
        for seed in (11, 3, 7)
    )
    output = tmp_path / "ifp_study.json"

    study = aggregate_ifp_study(reports, report_path=output)

    assert study["schema_version"] == "so101_wam.ifp_study.v4"
    assert study["result"] == "complete"
    assert study["scientific_claim"] == "not_evaluated"
    assert "preregistration" not in study
    assert "endpoint_evaluation" not in study
    assert set(study) == {
        "schema_version",
        "result",
        "evidence_level",
        "robot_used",
        "scientific_claim",
        "result_semantics",
        "seed_count",
        "seeds",
        "comparability",
        "sources",
        "variants",
        "limitations",
    }
    assert study["seed_count"] == 3
    assert study["seeds"] == [3, 7, 11]
    assert study["comparability"] == {
        "minimum_seed_count": True,
        "same_closed_loop_protocol": True,
        "same_data_split": True,
        "same_metric_schema": True,
        "same_optimizer_protocol": True,
        "same_variant_set": True,
        "unique_seeds": True,
    }
    assert [variant["ifp_steps"] for variant in study["variants"]] == [0, 2, 4]
    k0 = study["variants"][0]
    assert k0["validation"]["action_mse_normalized"] == {
        "count": 3,
        "max": 11.0,
        "mean": 7.0,
        "min": 3.0,
        "sample_std": 4.0,
    }
    k4 = study["variants"][2]
    assert k4["closed_loop"]["trial_count"] == 6
    assert k4["closed_loop"]["scored_trial_count"] == 6
    assert k4["closed_loop"]["execution_failure_count"] == 0
    assert k4["closed_loop"]["success_count"] == 3
    assert k4["closed_loop"]["success_rate"] == 0.5
    assert k4["closed_loop"]["terminal_error_basis"] == (
        "max_abs_joint_position_error"
    )
    assert k4["closed_loop"]["failure_counts"] == {
        "terminal_joint_target_tolerance": 3,
    }
    assert "winner" not in study
    assert json.loads(output.read_text(encoding="utf-8")) == study

    sources = study["sources"]
    assert [source["seed"] for source in sources] == [3, 7, 11]
    for source in sources:
        source_path = tmp_path / source["report"]
        assert source["report_sha256"] == sha256(source_path.read_bytes()).hexdigest()

    with pytest.raises(IFPStudyError, match="already exists"):
        aggregate_ifp_study(reports, report_path=output)


def test_ifp_legacy_field(tmp_path: Path) -> None:
    reports = tuple(
        _write_report(
            tmp_path,
            seed=seed,
            action_decoder=ActionDecoder.LEGACY_MEAN.value,
            k4_favorable=True,
        )
        for seed in (3, 7, 11)
    )
    plan = _write_plan(tmp_path)

    study = aggregate_ifp_study(
        reports,
        report_path=tmp_path / "explicit-legacy.json",
        preregistration_path=plan,
    )

    assert study["scientific_claim"] == "local_ifp_diagnostic_evaluated"
    assert study["endpoint_evaluation"]["passed"] is True


@pytest.mark.parametrize(
    "mode",
    [
        ActionDecoder.MEAN_REPEAT_CONTROL.value,
        ActionDecoder.ORDERED_CONCAT.value,
        "unknown_decoder",
    ],
)
def test_ifp_rejects_decoder(tmp_path: Path, mode: str) -> None:
    reports = tuple(
        _write_report(tmp_path, seed=seed, action_decoder=mode)
        for seed in (3, 7, 11)
    )

    with pytest.raises(IFPStudyError, match="action_decoder"):
        aggregate_ifp_study(
            reports,
            report_path=tmp_path / f"{mode}.json",
        )


def test_ifp_study_requires_three_unique_seeds(tmp_path: Path) -> None:
    first = _write_report(tmp_path, seed=3)
    second = _write_report(tmp_path, seed=7)

    with pytest.raises(IFPStudyError, match="at least 3"):
        aggregate_ifp_study(
            (first, second),
            report_path=tmp_path / "too-small.json",
        )

    duplicate = tmp_path / "duplicate" / "ifp_ablation.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(IFPStudyError, match="unique seeds"):
        aggregate_ifp_study(
            (first, second, duplicate),
            report_path=tmp_path / "duplicate-seed.json",
        )


def test_ifp_study_weights_terminal_error_by_scored_trials(tmp_path: Path) -> None:
    reports = [
        _write_report(tmp_path, seed=seed)
        for seed in (3, 7, 11)
    ]
    for report in reports[1:]:
        payload = json.loads(report.read_text(encoding="utf-8"))
        for variant in payload["variants"]:
            variant["closed_loop"]["terminal_error_mean"] = 1.0
        report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    first = json.loads(reports[0].read_text(encoding="utf-8"))
    for variant in first["variants"]:
        closed = variant["closed_loop"]
        closed["scored_trial_count"] = 1
        closed["execution_failure_count"] = 1
        closed["terminal_error_mean"] = 10.0
        if closed["success_count"] == 0:
            closed["failure_counts"] = {
                "stale_action": 1,
                "terminal_joint_target_tolerance": 1,
            }
        else:
            closed["failure_counts"] = {"stale_action": 1}
    reports[0].write_text(json.dumps(first, sort_keys=True), encoding="utf-8")

    study = aggregate_ifp_study(
        reports,
        report_path=tmp_path / "weighted.json",
    )

    k4 = study["variants"][2]["closed_loop"]
    assert k4["trial_count"] == 6
    assert k4["scored_trial_count"] == 5
    assert k4["execution_failure_count"] == 1
    assert k4["terminal_error_mean"] == pytest.approx(14.0 / 5.0)


@pytest.mark.parametrize("failed_seeds", [(3,), (3, 7, 11)])
def test_study_keeps_unscored_seeds(
    tmp_path: Path,
    failed_seeds: tuple[int, ...],
) -> None:
    reports = [_write_report(tmp_path, seed=seed) for seed in (3, 7, 11)]
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["variants"][0]["optimization"]["seed"] not in failed_seeds:
            continue
        for variant in payload["variants"]:
            variant["closed_loop"].update(
                scored_trial_count=0,
                execution_failure_count=2,
                success_count=0,
                success_rate=0.0,
                terminal_error_mean=None,
                failure_counts={"stale_action": 2},
            )
        path.write_text(json.dumps(payload), encoding="utf-8")

    study = aggregate_ifp_study(
        reports,
        report_path=tmp_path / "failures.json",
        preregistration_path=_write_plan(tmp_path),
    )

    assert study["result"] == "complete"
    k4 = study["variants"][2]["closed_loop"]
    assert k4["trial_count"] == 6
    assert k4["execution_failure_count"] == 2 * len(failed_seeds)
    assert k4["success_rate"] == (3 - len(failed_seeds)) / 6
    assert k4["terminal_error_mean"] == (
        13.0 if len(failed_seeds) == 1 else None
    )
    assert k4["failure_counts"]["stale_action"] == 2 * len(failed_seeds)
    if len(failed_seeds) == 3:
        endpoint = study["endpoint_evaluation"]
        assert endpoint["secondary_endpoint_available"] is False
        assert endpoint["secondary_delta_k4_vs_k0"] is None
        assert endpoint["secondary_passed"] is endpoint["passed"] is False


def test_study_requires_protocol_hash(tmp_path: Path) -> None:
    reports = [_write_report(tmp_path, seed=seed) for seed in (3, 7, 11)]
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    for variant in payload["variants"]:
        variant["closed_loop"].pop("protocol_sha256", None)
    reports[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IFPStudyError, match="protocol"):
        aggregate_ifp_study(reports, report_path=tmp_path / "missing-protocol.json")


@pytest.mark.parametrize("changed_steps", [(0,), (0, 2, 4)])
def test_study_rejects_protocol_drift(
    tmp_path: Path,
    changed_steps: tuple[int, ...],
) -> None:
    reports = [_write_report(tmp_path, seed=seed) for seed in (3, 7, 11)]
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    for variant in payload["variants"]:
        if variant["ifp_steps"] in changed_steps:
            variant["closed_loop"]["protocol_sha256"] = "e" * 64
    reports[0].write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "different-protocol.json"

    with pytest.raises(IFPStudyError, match="closed-loop protocol"):
        aggregate_ifp_study(reports, report_path=output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("split", "data split"),
        ("budget", "optimizer protocol"),
        ("metric", "metric schema"),
        ("variant", "variant set"),
    ),
)
def test_ifp_study_rejects_incomparable_reports(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    reports = [
        _write_report(tmp_path, seed=3),
        _write_report(tmp_path, seed=7),
        _write_report(
            tmp_path,
            seed=11,
            train_split="c" * 64 if mutation == "split" else "a" * 64,
            stage2_steps=21 if mutation == "budget" else 20,
            metric_extra=mutation == "metric",
            variant_steps=(0, 4) if mutation == "variant" else (0, 2, 4),
        ),
    ]

    output = tmp_path / "ifp_study.json"
    with pytest.raises(IFPStudyError, match=message):
        aggregate_ifp_study(reports, report_path=output)
    assert not output.exists()


def test_ifp_study_evaluates_matching_preregistration(tmp_path: Path) -> None:
    reports = tuple(
        _write_report(tmp_path, seed=seed, k4_favorable=True)
        for seed in (11, 3, 7)
    )
    plan = _write_plan(tmp_path)
    output = tmp_path / "ifp_study.json"

    study = aggregate_ifp_study(
        reports,
        report_path=output,
        preregistration_path=plan,
    )

    assert study["result"] == "complete"
    assert study["scientific_claim"] == "local_ifp_diagnostic_evaluated"
    assert study["preregistration"] == {
        "plan_sha256": sha256(plan.read_bytes()).hexdigest(),
        "hypothesis_id": "ifp-k4-local-v1",
        "claim_scope": IFP_PREREGISTRATION_CLAIM_SCOPE,
        "plan_matched": True,
        "external_preregistration_timing_verified": False,
    }
    assert study["endpoint_evaluation"]["passed"] is True
    assert study["endpoint_evaluation"]["primary_delta_k4_vs_k0"] == 1.0
    assert study["endpoint_evaluation"]["secondary_delta_k4_vs_k0"] == -0.5
    assert "winner" not in study
    assert json.loads(output.read_text(encoding="utf-8")) == study


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("seeds", "seeds"),
        ("split", "train split"),
        ("optimizer", "optimizer protocol"),
        ("scope", "closed-loop scope"),
        ("trials", "trial minimum"),
    ),
)
def test_ifp_study_rejects_preregistration_protocol_mismatch(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    reports = tuple(
        _write_report(tmp_path, seed=seed)
        for seed in (3, 7, 11)
    )
    plan = _write_plan(
        tmp_path,
        seeds=(3, 7, 13) if mutation == "seeds" else (3, 7, 11),
        train_split="c" * 64 if mutation == "split" else "a" * 64,
        stage2_steps=21 if mutation == "optimizer" else 20,
        scope="different_proxy" if mutation == "scope" else (
            "synthetic_terminal_joint_proxy"
        ),
        minimum_trials=3 if mutation == "trials" else 2,
    )
    output = tmp_path / "ifp_study.json"

    with pytest.raises(IFPStudyError, match=message):
        aggregate_ifp_study(
            reports,
            report_path=output,
            preregistration_path=plan,
        )

    assert not output.exists()


def test_ifp_study_records_negative_preregistered_result(tmp_path: Path) -> None:
    reports = tuple(
        _write_report(tmp_path, seed=seed)
        for seed in (3, 7, 11)
    )
    plan = _write_plan(tmp_path)
    output = tmp_path / "ifp_study.json"

    study = aggregate_ifp_study(
        reports,
        report_path=output,
        preregistration_path=plan,
    )

    assert study["result"] == "complete"
    assert study["endpoint_evaluation"]["passed"] is False
    assert study["endpoint_evaluation"]["secondary_passed"] is False
    assert "winner" not in study


def test_ifp_study_cli_accepts_preregistration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports = tuple(
        _write_report(tmp_path, seed=seed, k4_favorable=True)
        for seed in (3, 7, 11)
    )
    plan = _write_plan(tmp_path)
    output = tmp_path / "ifp_study.json"

    result = ifp_study_main(
        [
            "--reports",
            *(str(report) for report in reports),
            "--preregistration",
            str(plan),
            "--report",
            str(output),
        ]
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["endpoint_evaluation"]["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == printed
