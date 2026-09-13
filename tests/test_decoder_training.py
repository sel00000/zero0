from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from so101_wam.checkpoint import load_compact_wam_bundle
from so101_wam.decoder_evaluation import (
    DECODER_AUDIT_SCHEMA,
    FutureSource,
    named_tensor_hashes,
    order_schedule,
    reduce_decoder_rows,
    evaluate_decoder,
)
from so101_wam.deployment import canonical_json_sha256
from so101_wam.model import ActionDecoder, CompactWAM
from so101_wam.training import (
    CompactWAMTrainingConfig,
    TrainingError,
    train_offline_candidate,
)
from so101_wam.training_data import EpisodeRecord, build_training_windows


def _row(key: str, task: str, mse: float, native: float) -> dict[str, Any]:
    return {
        "key": key,
        "task": task,
        "mse": mse,
        "mae_native": [native] * 12,
        "reverse": {"output_mse": 0.0, "error_delta": 0.0},
        "permute": {"output_mse": 0.0, "error_delta": 0.0},
    }


def _windows(
    records: tuple[EpisodeRecord, ...],
    config: CompactWAMTrainingConfig,
):
    return build_training_windows(
        records,
        policy_hz=config.policy_hz,
        servo_hz=config.servo_hz,
        action_history_steps=config.action_history_steps,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        ifp_steps=config.ifp_steps,
    )


def _model(config: CompactWAMTrainingConfig, mode: ActionDecoder) -> CompactWAM:
    return CompactWAM(
        latent_dim=config.latent_dim,
        transformer_heads=config.transformer_heads,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        action_history_steps=config.action_history_steps,
        ifp_steps=config.ifp_steps,
        action_decoder=mode,
    )


@contextmanager
def _deterministic():
    threads = torch.get_num_threads()
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def _train_audit(
    tmp_path: Path,
    records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    config: CompactWAMTrainingConfig,
    mode: ActionDecoder,
    stem: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    checkpoint_path = tmp_path / f"{stem}.pt"
    report_path = tmp_path / f"{stem}.json"
    train_offline_candidate(
        records[0],
        records[1],
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        checkpoint_id=stem,
        config=replace(config, action_decoder=mode),
        study_sha256="a" * 64,
        device="cpu",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = load_compact_wam_bundle(checkpoint_path)
    return report, named_tensor_hashes(bundle.model), dict(bundle.metadata)


def _non_head(items: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in items.items()
        if not key.startswith("action_head.")
    }


def test_macro_and_axis_means() -> None:
    rows = [
        _row("a", "1:one", 1.0, 2.0),
        _row("b", "1:one", 3.0, 4.0),
        _row("c", "2:two", 10.0, 9.0),
    ]

    summary = reduce_decoder_rows(rows)

    assert summary["window_count"] == 3
    assert summary["task_count"] == 2
    assert summary["task_macro_mse"] == 6.0
    assert summary["native_mae_per_axis"] == [6.0] * 12
    assert summary["per_task"]["1:one"]["windows"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mse", math.nan),
        ("mae_native", [math.inf] * 12),
        ("reverse", {"output_mse": math.nan, "error_delta": 0.0}),
    ],
)
def test_reduce_rejects_nonfinite(field: str, value: Any) -> None:
    row = _row("a", "1:one", 1.0, 2.0)
    row[field] = value

    with pytest.raises(ValueError):
        reduce_decoder_rows([row])


@pytest.mark.parametrize("field", ["mse", "mae_native", "reverse"])
def test_rejects_numpy_numbers(field: str) -> None:
    row = _row("a", "1:one", 1.0, 2.0)
    if field == "mse":
        row["mse"] = np.float64(1.0)
    elif field == "mae_native":
        row["mae_native"] = [np.float64(1.0), *([1.0] * 11)]
    else:
        row["reverse"] = {"output_mse": np.float64(0.0), "error_delta": 0.0}

    with pytest.raises(ValueError):
        reduce_decoder_rows([row])


def test_duplicate_window() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        reduce_decoder_rows([_row("a", "1:one", 1.0, 2.0), _row("a", "2:two", 3.0, 4.0)])


def test_axis_width() -> None:
    row = _row("a", "1:one", 1.0, 2.0)
    row["mae_native"] = [1.0] * 11

    with pytest.raises(ValueError, match="12"):
        reduce_decoder_rows([row])


def test_large_finite_mean() -> None:
    summary = reduce_decoder_rows(
        [_row("a", "1:one", 1e308, 1.0), _row("b", "1:one", 1e308, 1.0)]
    )

    assert summary["task_macro_mse"] == 1e308


def test_single_future_rejected(
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    windows = _windows(decoder_records[1], decoder_config)
    single = windows[0].__class__(
        pair=windows[0].pair,
        anchor_policy_position=windows[0].anchor_policy_position,
        action_history_steps=windows[0].action_history_steps,
        future_steps=1,
        action_horizon=windows[0].action_horizon,
        servo_hz=windows[0].servo_hz,
        ifp_steps=windows[0].ifp_steps,
        ifp_stride=windows[0].ifp_stride,
    )

    with pytest.raises(ValueError, match="future"):
        order_schedule([single])


def test_order_nonidentity(
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    windows = _windows(decoder_records[1], decoder_config)

    schedule = order_schedule(windows[:4])

    assert order_schedule(windows[:4]) == schedule
    for spec, item in zip(windows[:4], schedule, strict=True):
        assert item["permutation"] != list(range(spec.future_steps))


@pytest.mark.parametrize("mode", list(ActionDecoder))
def test_eval_restores_state(
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    mode: ActionDecoder,
) -> None:
    torch.manual_seed(13)
    model = _model(decoder_config, mode)
    windows = _windows(decoder_records[1], decoder_config)[:2]
    model.train()
    before_hashes = named_tensor_hashes(model)
    before_torch = torch.get_rng_state().clone()
    before_numpy = np.random.get_state()

    first = evaluate_decoder(model, windows, source=FutureSource.ORACLE)
    second = evaluate_decoder(model, windows, source=FutureSource.ORACLE)

    assert first == second
    assert model.training
    assert named_tensor_hashes(model) == before_hashes
    assert all(parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]


def test_eval_source_rejected(
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    model = _model(decoder_config, ActionDecoder.ORDERED_CONCAT)
    model.train()
    model.encoder.eval()
    modes = {name: module.training for name, module in model.named_modules()}

    with pytest.raises(ValueError, match="source"):
        evaluate_decoder(model, _windows(decoder_records[1], decoder_config)[:1], source="oracle")

    assert {name: module.training for name, module in model.named_modules()} == modes


def test_eval_failure_restores(
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    model = _model(decoder_config, ActionDecoder.ORDERED_CONCAT)
    model.train()
    model.encoder.eval()
    modes = {name: module.training for name, module in model.named_modules()}
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.fill_(float("nan"))

    with pytest.raises(ValueError):
        evaluate_decoder(
            model,
            _windows(decoder_records[1], decoder_config)[:1],
            source=FutureSource.ORACLE,
        )

    assert {name: module.training for name, module in model.named_modules()} == modes


@pytest.mark.parametrize("mode", list(ActionDecoder))
def test_audit_replays_same_seed(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    mode: ActionDecoder,
) -> None:
    with _deterministic():
        left, left_hashes, left_meta = _train_audit(
            tmp_path,
            decoder_records,
            decoder_config,
            mode,
            f"{mode.value}-left",
        )
        right, right_hashes, right_meta = _train_audit(
            tmp_path,
            decoder_records,
            decoder_config,
            mode,
            f"{mode.value}-right",
        )

    assert left_hashes == right_hashes
    assert left["training_schedule_sha256"] == right["training_schedule_sha256"]
    assert left["validation"] == right["validation"]
    assert left["training_evidence_sha256"] == canonical_json_sha256(
        {
            key: value
            for key, value in left.items()
            if key not in {"training_evidence_sha256", "artifacts"}
        }
    )

    audit = left["decoder_audit"]
    right_audit = right["decoder_audit"]
    assert audit["schema_version"] == DECODER_AUDIT_SCHEMA
    assert audit["study_sha256"] == "a" * 64
    assert audit["stage1"]["source"] == FutureSource.ORACLE.value
    assert audit["stage2"]["source"] == FutureSource.PREDICTED.value
    assert audit["initial_tensors"] == right_audit["initial_tensors"]
    assert audit["stage1_tensors"] == right_audit["stage1_tensors"]
    assert audit["final_tensors"] == right_audit["final_tensors"]
    assert audit["head_parameters"] == right_audit["head_parameters"]
    assert audit["model_parameters"] == right_audit["model_parameters"]
    assert audit["stage1"] == right_audit["stage1"]
    assert audit["stage2"] == right_audit["stage2"]
    assert left_meta["action_decoder"] == mode.value
    assert left_meta["study_sha256"] == "a" * 64
    assert left_meta["offline_trained"] is True
    assert left_meta["trained"] is False
    assert left_meta["deployment_ready"] is False
    assert right_meta["action_decoder"] == mode.value


def test_audit_fair_init(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    reports: dict[ActionDecoder, dict[str, Any]] = {}
    with _deterministic():
        for mode in ActionDecoder:
            report, _, _ = _train_audit(
                tmp_path,
                decoder_records,
                decoder_config,
                mode,
                f"{mode.value}-single",
            )
            reports[mode] = report

    legacy = reports[ActionDecoder.LEGACY_MEAN]
    repeat = reports[ActionDecoder.MEAN_REPEAT_CONTROL]
    ordered = reports[ActionDecoder.ORDERED_CONCAT]
    for report in reports.values():
        audit = report["decoder_audit"]
        assert _non_head(audit["initial_tensors"]) == _non_head(
            legacy["decoder_audit"]["initial_tensors"]
        )
        assert _non_head(audit["stage1_tensors"]) == _non_head(
            audit["initial_tensors"]
        )
        assert report["normalization"] == legacy["normalization"]
        assert report["data"] == legacy["data"]
        assert (
            report["training_schedule_sha256"]
            == legacy["training_schedule_sha256"]
        )
        assert (
            audit["stage1"]["permutation_sha256"]
            == legacy["decoder_audit"]["stage1"]["permutation_sha256"]
        )

    assert repeat["decoder_audit"]["head_parameters"] == ordered[
        "decoder_audit"
    ]["head_parameters"]
    repeat_head = {
        key: value
        for key, value in repeat["decoder_audit"]["initial_tensors"].items()
        if key.startswith("action_head.")
    }
    ordered_head = {
        key: value
        for key, value in ordered["decoder_audit"]["initial_tensors"].items()
        if key.startswith("action_head.")
    }
    assert repeat_head == ordered_head


def test_audit_rejects_bad_inputs(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    with pytest.raises(TrainingError, match="action_decoder"):
        CompactWAMTrainingConfig(action_decoder="legacy_mean")  # type: ignore[arg-type]

    base = replace(decoder_config, action_decoder=ActionDecoder.ORDERED_CONCAT)
    cases = (
        ("bad", base, "cpu", "study_sha256"),
        ("a" * 64, replace(base, future_steps=1), "cpu", "future_steps"),
        ("a" * 64, replace(base, stage1_steps=0), "cpu", "stage1_steps"),
        ("a" * 64, base, "meta", "cpu"),
    )
    for study_sha256, config, device, message in cases:
        checkpoint_path = tmp_path / f"{message}.pt"
        report_path = tmp_path / f"{message}.json"
        with pytest.raises(TrainingError, match=message):
            train_offline_candidate(
                decoder_records[0],
                decoder_records[1],
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                checkpoint_id=message,
                config=config,
                study_sha256=study_sha256,
                device=device,
            )
        assert not checkpoint_path.exists()
        assert not report_path.exists()


def test_nonaudit_keeps_f1(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    checkpoint_path = tmp_path / "plain.pt"
    report_path = tmp_path / "plain.json"
    config = replace(decoder_config, future_steps=1)

    train_offline_candidate(
        decoder_records[0],
        decoder_records[1],
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        checkpoint_id="plain",
        config=config,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "decoder_audit" not in report
    assert report["model"]["action_decoder"] == ActionDecoder.LEGACY_MEAN.value
