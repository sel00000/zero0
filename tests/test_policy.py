from __future__ import annotations

import numpy as np
import pytest
import torch

from so101_wam.constants import ACTION_DIM
from so101_wam.adapters.fake import FakeBimanualRobot
from so101_wam.config import ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.context import Gen15Context
from so101_wam.contracts import PhysicalPrompt, SensorimotorFrame
from so101_wam.model import CompactWAM, ModelContractError
from so101_wam.policy import (
    CompactWAMPolicy,
    FutureLatentTelemetryMode,
    PolicyError,
)
from so101_wam.runtime import RuntimeState, SO101WAMRuntime


def make_frame(timestamp_s: float, *, value: int) -> SensorimotorFrame:
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={
            "left_wrist": np.full((8, 8, 3), value, dtype=np.uint8),
            "right_wrist": np.full((8, 8, 3), value + 1, dtype=np.uint8),
        },
        joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
        executed_action=np.full(ACTION_DIM, value + 0.5, dtype=np.float32),
    )


def make_snapshot(*, live_count: int = 4, prompt_offset: int = 0):
    prompt = PhysicalPrompt(
        (
            make_frame(0.0, value=1 + prompt_offset),
            make_frame(3.0, value=2 + prompt_offset),
        )
    )
    context = Gen15Context(prompt, policy_hz=1.0)
    for index in range(live_count):
        context.append_live(make_frame(4.0 + index, value=10 + index))
    return context.snapshot()


def test_compact_wam_policy_predicts_action_chunk_without_training_update() -> None:
    torch.manual_seed(5)
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=3,
        action_history_steps=4,
    )
    model.train()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    chunk = CompactWAMPolicy(model, servo_hz=50.0).predict(make_snapshot(), now_s=12.0)

    assert chunk.target_joint_position.shape == (3, ACTION_DIM)
    assert chunk.dt_s == pytest.approx(0.02)
    assert chunk.created_at_s == 12.0
    assert chunk.target_joint_position.dtype == np.float32
    assert not chunk.target_joint_position.flags.writeable
    assert model.training is True
    for previous, current in zip(before, model.parameters(), strict=True):
        assert torch.equal(previous, current)


def test_policy_records_future_latent_telemetry_only_when_opted_in() -> None:
    torch.manual_seed(5)
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=3,
        action_history_steps=4,
    )
    snapshot = make_snapshot()
    disabled = CompactWAMPolicy(model, servo_hz=50.0)
    disabled.predict(snapshot, now_s=12.0)

    assert disabled.take_future_latent_telemetry() is None

    enabled = CompactWAMPolicy(
        model,
        servo_hz=50.0,
        telemetry_mode=FutureLatentTelemetryMode.CAPTURE,
    )
    chunk = enabled.predict(snapshot, now_s=12.0)
    telemetry = enabled.take_future_latent_telemetry()

    assert chunk.created_at_s == 12.0
    assert telemetry is not None
    assert telemetry.observation_timestamp_s == snapshot.live_frames[-1].timestamp_s
    assert telemetry.future_latents.shape == (2, 2, 16)
    assert telemetry.observed_latent.shape == (2, 16)
    assert telemetry.future_latents.dtype == np.float32
    assert telemetry.observed_latent.dtype == np.float32
    assert not telemetry.future_latents.flags.writeable
    assert not telemetry.observed_latent.flags.writeable
    assert enabled.take_future_latent_telemetry() is None


def test_policy_reuses_and_clears_prompt_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=3,
        action_history_steps=4,
    )
    policy = CompactWAMPolicy(model, servo_hz=50.0)
    snapshot = make_snapshot()
    encode_calls = 0
    encode_prompt = model.encode_prompt

    def counted_encode(*args: torch.Tensor, **kwargs: torch.Tensor):
        nonlocal encode_calls
        encode_calls += 1
        return encode_prompt(*args, **kwargs)

    monkeypatch.setattr(model, "encode_prompt", counted_encode)

    first = policy.predict(snapshot, now_s=12.0)
    second = policy.predict(snapshot, now_s=13.0)
    policy.predict(make_snapshot(prompt_offset=5), now_s=13.5)
    policy.clear_prompt_cache()
    third = policy.predict(snapshot, now_s=14.0)

    assert encode_calls == 3
    np.testing.assert_array_equal(
        first.target_joint_position,
        second.target_joint_position,
    )
    np.testing.assert_array_equal(
        first.target_joint_position,
        third.target_joint_position,
    )


def test_compact_wam_policy_rejects_short_live_history_before_inference() -> None:
    model = CompactWAM(latent_dim=16, transformer_heads=4, action_history_steps=4)
    policy = CompactWAMPolicy(model, servo_hz=50.0)

    with pytest.raises(ModelContractError, match="live history"):
        policy.predict(make_snapshot(live_count=3), now_s=12.0)


class BadOutputModel(torch.nn.Module):
    action_horizon = 2
    action_history_steps = 1

    def infer_action(self, **kwargs: torch.Tensor) -> torch.Tensor:
        del kwargs
        return self.output


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (torch.zeros(2, 2, ACTION_DIM), "shape"),
        (torch.zeros(1, 2, ACTION_DIM, dtype=torch.float64), "float32"),
        (torch.full((1, 2, ACTION_DIM), float("nan")), "NaN"),
    ],
)
def test_compact_wam_policy_rejects_bad_model_outputs(output: torch.Tensor, message: str) -> None:
    model = BadOutputModel()
    model.output = output

    with pytest.raises(PolicyError, match=message):
        CompactWAMPolicy(model, servo_hz=50.0).predict(make_snapshot(live_count=1), now_s=12.0)  # type: ignore[arg-type]


def test_compact_wam_policy_validates_time_and_servo_rate() -> None:
    model = CompactWAM(latent_dim=16, transformer_heads=4)

    with pytest.raises(PolicyError, match="servo_hz"):
        CompactWAMPolicy(model, servo_hz=0.0)
    with pytest.raises(PolicyError, match="now_s"):
        CompactWAMPolicy(model, servo_hz=50.0).predict(make_snapshot(), now_s=-1.0)


def test_compact_wam_policy_rejects_device_mismatch_without_implicit_move() -> None:
    model = CompactWAM(latent_dim=16, transformer_heads=4)

    with pytest.raises(PolicyError, match="move the model"):
        CompactWAMPolicy(model, servo_hz=50.0, device="meta", move_model=False)


def test_compact_model_closes_runtime_shadow_loop_after_measured_history() -> None:
    torch.manual_seed(7)
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=2,
        action_history_steps=4,
    )
    runtime = SO101WAMRuntime(
        config=ProjectConfig(
            runtime=RuntimeConfig(action_horizon=2, actuation_enabled=False),
            safety=SafetyConfig(
                joint_lower=(-100.0,) * ACTION_DIM,
                joint_upper=(100.0,) * ACTION_DIM,
                max_delta_per_servo_tick=(2.0,) * ACTION_DIM,
            ),
        ),
        prompt=PhysicalPrompt((make_frame(0.0, value=1), make_frame(3.0, value=2))),
        robot=FakeBimanualRobot(),
        policy=CompactWAMPolicy(model, servo_hz=50.0),
    )
    for timestamp in (4.0, 4.1, 4.2, 4.3):
        runtime.prime_live(now_s=timestamp)

    policy_step = runtime.policy_step(now_s=4.4)
    servo_step = runtime.servo_step(now_s=4.4)

    assert policy_step.action.horizon == 2
    assert policy_step.batch.live_images.shape[1] == 5
    assert servo_step.safety.accepted is True
    assert servo_step.sent is False
    assert runtime.state is RuntimeState.ROLLING
