from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_COUNT
from so101_wam.model import ActionRangeConstraint, CompactWAM, ModelContractError
from so101_wam.paired_training import (
    _PairedBatch,
    _train_paired_inverse_dynamics,
    paired_training_batch,
)
from so101_wam.training import (
    COMPACT_IMAGE_MAX_SIDE,
    CompactWAMTrainingConfig,
    _evaluate,
    _train_inverse_dynamics_stage,
)
from so101_wam.training_data import build_training_windows, materialize_training_window


def _model(**kwargs: Any) -> CompactWAM:
    options: dict[str, Any] = {
        "latent_dim": 8,
        "transformer_heads": 2,
        "future_steps": 1,
        "action_horizon": 2,
        "action_history_steps": 4,
        "ifp_steps": 0,
    }
    options.update(kwargs)
    return CompactWAM(**options)


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        torch.randint(0, 256, (1, 2, PRIMARY_CAMERA_COUNT, 3, 8, 8), dtype=torch.uint8),
        torch.randn(1, 2, ACTION_DIM),
        torch.randn(1, 2, ACTION_DIM),
        torch.randint(0, 256, (1, 4, PRIMARY_CAMERA_COUNT, 3, 8, 8), dtype=torch.uint8),
        torch.randn(1, 4, ACTION_DIM),
        torch.randn(1, 4, ACTION_DIM),
        torch.ones(1, 2, dtype=torch.bool),
    )


def _bounds() -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.linspace(-1.25, -0.15, ACTION_DIM, dtype=torch.float64)
    upper = torch.linspace(0.2, 1.3, ACTION_DIM, dtype=torch.float64)
    return lower, upper


def test_normalized_clamp_enum() -> None:
    assert ActionRangeConstraint.NORMALIZED_CLAMP.value == "normalized_clamp"


def test_legacy_decode_unchanged() -> None:
    torch.manual_seed(10)
    legacy = _model()
    torch.manual_seed(10)
    explicit = _model(action_range_constraint=ActionRangeConstraint.UNBOUNDED)
    raw = torch.randn(2, 3, ACTION_DIM)

    assert legacy.action_range_constraint is ActionRangeConstraint.UNBOUNDED
    assert tuple(legacy.state_dict()) == tuple(explicit.state_dict())
    torch.testing.assert_close(legacy.decode_actions(raw), legacy.denormalize_axes(raw), rtol=0, atol=0)
    for left, right in zip(legacy.parameters(), explicit.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_bounded_decode_in_bounds() -> None:
    lower, upper = _bounds()
    model = _model(
        action_range_constraint=ActionRangeConstraint.AFFINE_TANH,
        action_lower=lower,
        action_upper=upper,
    )
    raw = torch.tensor([-1000.0, -3.0, 0.0, 3.0, 1000.0], dtype=torch.float32).view(1, 5, 1)
    raw = raw.expand(1, 5, ACTION_DIM).clone().requires_grad_(True)

    decoded = model.decode_actions(raw)

    assert torch.all(decoded.double() >= lower.view(1, 1, -1))
    assert torch.all(decoded.double() <= upper.view(1, 1, -1))
    assert torch.any(decoded > lower.to(decoded).view(1, 1, -1))
    assert torch.any(decoded < upper.to(decoded).view(1, 1, -1))
    decoded[:, 1:4].sum().backward()
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())
    assert float(raw.grad[:, 2, :].abs().sum()) > 0.0


def test_clamp_inrange_grad() -> None:
    model = _model(
        action_range_constraint=ActionRangeConstraint.NORMALIZED_CLAMP,
        action_lower=torch.full((ACTION_DIM,), -100.0, dtype=torch.float64),
        action_upper=torch.full((ACTION_DIM,), 100.0, dtype=torch.float64),
    )
    mean = torch.linspace(-6.0, 5.0, ACTION_DIM)
    scale = torch.linspace(0.5, 6.0, ACTION_DIM)
    model.set_axis_normalization(mean, scale)
    raw = torch.linspace(-0.25, 0.25, ACTION_DIM).view(1, 1, -1).requires_grad_(True)

    decoded = model.decode_actions(raw)

    torch.testing.assert_close(decoded, model.denormalize_axes(raw), rtol=0, atol=0)
    decoded.sum().backward()
    assert raw.grad is not None
    torch.testing.assert_close(raw.grad, scale.view(1, 1, -1), rtol=0, atol=0)


def test_clamp_endpoint_grad() -> None:
    lower = torch.zeros(ACTION_DIM, dtype=torch.float64)
    upper = torch.full((ACTION_DIM,), 100.0, dtype=torch.float64)
    model = _model(
        action_range_constraint=ActionRangeConstraint.NORMALIZED_CLAMP,
        action_lower=lower,
        action_upper=upper,
    )
    model.set_axis_normalization(
        torch.full((ACTION_DIM,), 50.0),
        torch.full((ACTION_DIM,), 50.0),
    )
    raw = torch.tensor([[-1.0, -2.0, 1.0, 2.0]], dtype=torch.float32).view(1, 4, 1)
    raw = raw.expand(1, 4, ACTION_DIM).clone().requires_grad_(True)

    decoded = model.decode_actions(raw)

    assert torch.all(decoded[:, 0, :] == 0.0)
    assert torch.all(decoded[:, 1, :] == 0.0)
    assert torch.all(decoded[:, 2, :] == 100.0)
    assert torch.all(decoded[:, 3, :] == 100.0)
    decoded.sum().backward()
    assert raw.grad is not None
    assert torch.all(raw.grad[:, 0, :] == 50.0)
    assert torch.all(raw.grad[:, 1, :] == 0.0)
    assert torch.all(raw.grad[:, 2, :] == 50.0)
    assert torch.all(raw.grad[:, 3, :] == 0.0)


def test_clamp_uses_rounded_bounds() -> None:
    lower = torch.full((ACTION_DIM,), 0.1, dtype=torch.float64)
    upper = torch.full((ACTION_DIM,), 1.0, dtype=torch.float64)
    model = _model(
        action_range_constraint=ActionRangeConstraint.NORMALIZED_CLAMP,
        action_lower=lower,
        action_upper=upper,
    )
    raw = torch.full((1, 1, ACTION_DIM), -100.0)

    decoded = model.decode_actions(raw)

    assert torch.all(decoded.double() >= lower.view(1, 1, -1))
    assert torch.all(decoded.double() <= upper.view(1, 1, -1))


@pytest.mark.parametrize(
    ("lower", "upper", "match"),
    [
        ([0.0] * (ACTION_DIM - 1), [1.0] * ACTION_DIM, "shape"),
        ([0.0] * ACTION_DIM, [1.0] * (ACTION_DIM - 1), "shape"),
        ([0.0] * ACTION_DIM, [float("inf")] * ACTION_DIM, "NaN or infinity"),
        ([1.0] * ACTION_DIM, [1.0] * ACTION_DIM, "lower < upper"),
    ],
)
def test_rejects_bad_bounds(
    lower: list[float],
    upper: list[float],
    match: str,
) -> None:
    with pytest.raises(ModelContractError, match=match):
        _model(
            action_range_constraint=ActionRangeConstraint.AFFINE_TANH,
            action_lower=lower,
            action_upper=upper,
        )


@pytest.mark.parametrize(
    "mode",
    [ActionRangeConstraint.AFFINE_TANH, ActionRangeConstraint.NORMALIZED_CLAMP],
)
def test_requires_both_bounds(mode: ActionRangeConstraint) -> None:
    lower, _ = _bounds()

    with pytest.raises(ModelContractError, match="requires action_lower and action_upper"):
        _model(
            action_range_constraint=mode,
            action_lower=lower,
        )


@pytest.mark.parametrize(
    "mode",
    [ActionRangeConstraint.AFFINE_TANH, ActionRangeConstraint.NORMALIZED_CLAMP],
)
def test_rejects_nonfinite_raw(mode: ActionRangeConstraint) -> None:
    lower, upper = _bounds()
    model = _model(
        action_range_constraint=mode,
        action_lower=lower,
        action_upper=upper,
    )
    raw = torch.zeros(1, 2, ACTION_DIM)
    raw[0, 0, 0] = torch.nan

    with pytest.raises(ModelContractError, match="raw action output"):
        model.decode_actions(raw)


@pytest.mark.parametrize(
    "mode",
    [ActionRangeConstraint.AFFINE_TANH, ActionRangeConstraint.NORMALIZED_CLAMP],
)
def test_rejects_huge_raw(mode: ActionRangeConstraint) -> None:
    lower, upper = _bounds()
    model = _model(
        action_range_constraint=mode,
        action_lower=lower,
        action_upper=upper,
    )
    raw = torch.zeros(1, 2, ACTION_DIM, dtype=torch.float64)
    raw[0, 0, 0] = torch.finfo(torch.float32).max * 2.0

    with pytest.raises(ModelContractError, match="float32 range"):
        model.decode_actions(raw)


def test_clamp_rejects_overflow() -> None:
    model = _model(
        action_range_constraint=ActionRangeConstraint.NORMALIZED_CLAMP,
        action_lower=torch.full((ACTION_DIM,), -1.0, dtype=torch.float64),
        action_upper=torch.full((ACTION_DIM,), 1.0, dtype=torch.float64),
    )
    model.set_axis_normalization(
        torch.zeros(ACTION_DIM),
        torch.full((ACTION_DIM,), torch.finfo(torch.float32).max),
    )
    raw = torch.full((1, 1, ACTION_DIM), 2.0)

    with pytest.raises(ModelContractError, match="native"):
        model.decode_actions(raw)


def test_rejects_tight_bounds() -> None:
    lower = torch.ones(ACTION_DIM, dtype=torch.float64)
    upper = lower + 1e-8

    with pytest.raises(ModelContractError, match="float32"):
        _model(
            action_range_constraint=ActionRangeConstraint.AFFINE_TANH,
            action_lower=lower,
            action_upper=upper,
        )


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_bounded_cached_parity(mode: str) -> None:
    torch.manual_seed(11)
    lower, upper = _bounds()
    model = _model(
        action_range_constraint=ActionRangeConstraint(mode),
        action_lower=lower,
        action_upper=upper,
    )
    inputs = _inputs()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask = inputs

    prompt_encoding = model.encode_prompt(
        prompt,
        prompt_proprio,
        prompt_actions,
        prompt_mask=prompt_mask,
    )
    cached = model.infer_action_cached(
        prompt_encoding,
        live,
        live_proprio,
        live_actions,
    )
    uncached = model.infer_action(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=prompt_mask,
    )

    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode",
    [ActionRangeConstraint.AFFINE_TANH, ActionRangeConstraint.NORMALIZED_CLAMP],
)
def test_load_refreshes_bounds(mode: ActionRangeConstraint) -> None:
    low_source = torch.full((ACTION_DIM,), 10.0, dtype=torch.float64)
    high_source = torch.full((ACTION_DIM,), 20.0, dtype=torch.float64)
    source = _model(
        action_range_constraint=mode,
        action_lower=low_source,
        action_upper=high_source,
    )
    target = _model(
        action_range_constraint=mode,
        action_lower=torch.full((ACTION_DIM,), -1.0, dtype=torch.float64),
        action_upper=torch.full((ACTION_DIM,), 1.0, dtype=torch.float64),
    )

    target.load_state_dict(source.state_dict())
    raw = torch.zeros(1, 1, ACTION_DIM)
    if mode is ActionRangeConstraint.NORMALIZED_CLAMP:
        raw = torch.full_like(raw, 15.0)
    decoded = target.decode_actions(raw)

    assert torch.all(decoded >= 10.0)
    assert torch.all(decoded <= 20.0)


def test_stage1_legacy_loss_path(
    decoder_records,
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    config = replace(decoder_config, future_steps=1, action_horizon=2, ifp_steps=0)
    model = _model(action_history_steps=config.action_history_steps)
    windows = tuple(
        window
        for window in build_training_windows(
            decoder_records[0],
            policy_hz=config.policy_hz,
            servo_hz=config.servo_hz,
            action_history_steps=config.action_history_steps,
            future_steps=config.future_steps,
            action_horizon=config.action_horizon,
            ifp_steps=config.ifp_steps,
        )
        if window.action_history_steps == model.action_history_steps
    )
    batch = materialize_training_window(windows[0], image_max_side=COMPACT_IMAGE_MAX_SIDE)
    with torch.no_grad():
        target_future = model.encoder(batch.target_future_images, segment_id=1)
        current = model.normalize_axes(batch.live_proprio[:, -1, :])
        history = model.normalize_axes(batch.live_actions[:, -model.action_history_steps :, :])
        expected = torch.nn.functional.mse_loss(
            model.action_head(target_future, current, history),
            model.normalize_axes(batch.target_actions),
        )

    loss = _train_inverse_dynamics_stage(model, windows[:1], config=config, device=torch.device("cpu"))

    assert loss == pytest.approx(float(expected))


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_stage1_bounded_loss(
    mode: str,
    decoder_records,
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    lower = torch.full((ACTION_DIM,), -100.0, dtype=torch.float64)
    upper = torch.full((ACTION_DIM,), 100.0, dtype=torch.float64)
    config = replace(decoder_config, future_steps=1, action_horizon=2, ifp_steps=0)
    model = _model(
        action_history_steps=config.action_history_steps,
        action_range_constraint=ActionRangeConstraint(mode),
        action_lower=lower,
        action_upper=upper,
    )
    windows = build_training_windows(
        decoder_records[0],
        policy_hz=config.policy_hz,
        servo_hz=config.servo_hz,
        action_history_steps=config.action_history_steps,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        ifp_steps=config.ifp_steps,
    )
    batch = materialize_training_window(windows[0], image_max_side=COMPACT_IMAGE_MAX_SIDE)
    with torch.no_grad():
        target_future = model.encoder(batch.target_future_images, segment_id=1)
        current = model.normalize_axes(batch.live_proprio[:, -1, :])
        history = model.normalize_axes(batch.live_actions[:, -model.action_history_steps :, :])
        raw = model.action_head(target_future, current, history)
        expected = torch.nn.functional.mse_loss(
            model.normalize_axes(model.decode_actions(raw)),
            model.normalize_axes(batch.target_actions),
        )

    loss = _train_inverse_dynamics_stage(model, windows[:1], config=config, device=torch.device("cpu"))

    assert loss == pytest.approx(float(expected))


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_bounded_eval_metrics(
    mode: str,
    decoder_records,
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    lower = torch.full((ACTION_DIM,), -100.0, dtype=torch.float64)
    upper = torch.full((ACTION_DIM,), 100.0, dtype=torch.float64)
    lower[-1] = 0.0
    config = replace(decoder_config, future_steps=1, action_horizon=2, ifp_steps=0)
    model = _model(
        action_history_steps=config.action_history_steps,
        action_range_constraint=ActionRangeConstraint(mode),
        action_lower=lower,
        action_upper=upper,
    )
    windows = build_training_windows(
        decoder_records[0],
        policy_hz=config.policy_hz,
        servo_hz=config.servo_hz,
        action_history_steps=config.action_history_steps,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        ifp_steps=config.ifp_steps,
    )

    metrics = _evaluate(model, windows[:1], config=config, device=torch.device("cpu"))

    for axis in range(ACTION_DIM):
        assert f"action_mae_native_axis_{axis}" in metrics
    assert "action_out_of_bounds_count" in metrics
    assert "action_endpoint_target_count" in metrics
    assert "action_endpoint_mae_native" in metrics


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_paired_stage1_loss(
    mode: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_paired_training import _pair, _training_config

    pair = _pair(tmp_path)
    item = paired_training_batch(
        pair,
        neutral_axes=[0.0] * ACTION_DIM,
        policy_hz=10.0,
        servo_hz=50.0,
        action_history_steps=4,
        future_steps=1,
        action_horizon=2,
    )
    batch_item = _PairedBatch(
        pair=pair,
        batch=item,
        anchor_policy_position=0,
        anchor_time_s=0.0,
    )
    lower = torch.full((ACTION_DIM,), -100.0, dtype=torch.float64)
    upper = torch.full((ACTION_DIM,), 100.0, dtype=torch.float64)
    model = _model(
        action_range_constraint=ActionRangeConstraint(mode),
        action_lower=lower,
        action_upper=upper,
    )
    config = _training_config()
    calls: list[torch.Tensor] = []
    original = model.decode_actions

    def spy(raw: torch.Tensor) -> torch.Tensor:
        calls.append(raw.detach().clone())
        return original(raw)

    monkeypatch.setattr(model, "decode_actions", spy)

    _train_paired_inverse_dynamics(model, (batch_item,), config=config, device=torch.device("cpu"))

    assert calls
