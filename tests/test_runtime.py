from __future__ import annotations

import numpy as np
import pytest

from so101_wam.adapters.fake import FakeBimanualRobot
from so101_wam.config import ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.constants import ACTION_DIM
from so101_wam.context import ContextSnapshot
from so101_wam.contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from so101_wam.runtime import RuntimeErrorState, RuntimeState, SO101WAMRuntime, ServoExecutor


def make_frame(timestamp_s: float, *, value: float = 0.0) -> SensorimotorFrame:
    pixel = int(value * 10) % 255
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={
            "left_wrist": np.full((8, 8, 3), pixel, dtype=np.uint8),
            "right_wrist": np.full((8, 8, 3), pixel + 1, dtype=np.uint8),
        },
        joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
        executed_action=np.full(ACTION_DIM, value, dtype=np.float32),
    )


def make_prompt() -> PhysicalPrompt:
    return PhysicalPrompt((make_frame(0.0), make_frame(3.0, value=0.1)))


def config(*, actuation: bool, horizon: int = 2, safety: SafetyConfig | None = None) -> ProjectConfig:
    return ProjectConfig(
        runtime=RuntimeConfig(actuation_enabled=actuation, action_horizon=horizon),
        safety=safety or SafetyConfig.fake_normalized(),
    )


class SequencePolicy:
    def __init__(self, *, horizon: int, dt_s: float = 0.02) -> None:
        self.horizon = horizon
        self.dt_s = dt_s
        self.calls = 0
        self.snapshots: list[ContextSnapshot] = []

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        self.calls += 1
        self.snapshots.append(snapshot)
        base = float(self.calls)
        rows = np.stack(
            [np.full(ACTION_DIM, base + index, dtype=np.float32) for index in range(self.horizon)]
        )
        return ActionChunk(target_joint_position=rows, dt_s=self.dt_s, created_at_s=now_s)


def test_servo_executor_splits_chunk_and_validates_servo_period() -> None:
    executor = ServoExecutor(servo_hz=50.0)
    executor.load(
        ActionChunk(
            target_joint_position=np.array([[0.1] * ACTION_DIM, [0.2] * ACTION_DIM], dtype=np.float32),
            dt_s=0.02,
            created_at_s=10.0,
        )
    )

    first = executor.next_tick(now_s=11.0)
    second = executor.next_tick(now_s=11.02)

    assert first.horizon == second.horizon == 1
    assert first.created_at_s == second.created_at_s == 10.0
    np.testing.assert_allclose(first.target_joint_position[0], np.full(ACTION_DIM, 0.1))
    np.testing.assert_allclose(second.target_joint_position[0], np.full(ACTION_DIM, 0.2))
    assert executor.pending == 0
    with pytest.raises(RuntimeErrorState, match="no pending"):
        executor.next_tick(now_s=11.04)

    with pytest.raises(RuntimeErrorState, match="servo period"):
        executor.load(
            ActionChunk(
                target_joint_position=np.zeros((1, ACTION_DIM), dtype=np.float32),
                dt_s=0.1,
                created_at_s=12.0,
            )
        )


def test_shadow_rollout_never_records_unsent_policy_target_as_executed() -> None:
    robot = FakeBimanualRobot()
    runtime = SO101WAMRuntime(
        config=config(actuation=False),
        prompt=make_prompt(),
        robot=robot,
        policy=SequencePolicy(horizon=2),
    )

    policy_step = runtime.policy_step(now_s=4.0)
    first = runtime.servo_step(now_s=4.0)
    second = runtime.servo_step(now_s=4.02)

    assert policy_step.batch.live_images.shape == (1, 1, 2, 3, 8, 8)
    assert first.shadow is second.shadow is True
    assert first.sent is second.sent is False
    assert robot.sent_actions == []
    np.testing.assert_allclose(first.executed_action, np.zeros(ACTION_DIM))
    np.testing.assert_allclose(second.executed_action, np.zeros(ACTION_DIM))
    np.testing.assert_allclose(runtime.last_executed_action, np.zeros(ACTION_DIM))
    assert runtime.context.live_size == 1
    assert runtime.state is RuntimeState.ROLLOUT_READY


def test_enabled_fake_rollout_sends_one_row_per_servo_tick() -> None:
    robot = FakeBimanualRobot()
    policy = SequencePolicy(horizon=2)
    runtime = SO101WAMRuntime(
        config=config(actuation=True),
        prompt=make_prompt(),
        robot=robot,
        policy=policy,
    )
    runtime.policy_step(now_s=4.0)

    first = runtime.servo_step(now_s=4.0)
    second = runtime.servo_step(now_s=4.02)

    assert first.sent is second.sent is True
    assert len(robot.sent_actions) == 2
    np.testing.assert_allclose(robot.sent_actions[0], np.full(ACTION_DIM, 1.0))
    np.testing.assert_allclose(robot.sent_actions[1], np.full(ACTION_DIM, 2.0))
    assert runtime.state is RuntimeState.ROLLOUT_READY


def test_policy_clock_only_adds_context_frames_and_replaces_pending_horizon() -> None:
    robot = FakeBimanualRobot()
    policy = SequencePolicy(horizon=3)
    runtime = SO101WAMRuntime(
        config=config(actuation=True, horizon=3),
        prompt=make_prompt(),
        robot=robot,
        policy=policy,
    )

    runtime.policy_step(now_s=4.0)
    first = runtime.servo_step(now_s=4.0)
    assert first.pending == 2
    assert runtime.context.live_size == 1

    runtime.policy_step(now_s=4.1)
    replacement = runtime.servo_step(now_s=4.1)

    assert policy.calls == 2
    assert runtime.context.live_size == 2
    np.testing.assert_allclose(replacement.executed_action, np.full(ACTION_DIM, 2.0))
    assert replacement.pending == 2


def test_prime_live_builds_measured_history_without_policy_or_motion() -> None:
    robot = FakeBimanualRobot(joint_position=np.full(ACTION_DIM, 4.0, dtype=np.float32))
    policy = SequencePolicy(horizon=1)
    runtime = SO101WAMRuntime(
        config=config(actuation=False, horizon=1),
        prompt=make_prompt(),
        robot=robot,
        policy=policy,
    )

    for timestamp in (4.0, 4.1, 4.2, 4.3):
        snapshot = runtime.prime_live(now_s=timestamp)

    assert len(snapshot.live_frames) == 4
    assert policy.calls == 0
    assert robot.sent_actions == []
    for frame in snapshot.live_frames:
        np.testing.assert_allclose(frame.executed_action, np.full(ACTION_DIM, 4.0))


def test_safety_rejection_flushes_chunk_and_requires_explicit_recovery() -> None:
    tight = SafetyConfig(
        joint_lower=(-1.0,) * ACTION_DIM,
        joint_upper=(1.0,) * ACTION_DIM,
        max_delta_per_servo_tick=(2.0,) * ACTION_DIM,
    )
    robot = FakeBimanualRobot()
    runtime = SO101WAMRuntime(
        config=config(actuation=True, horizon=2, safety=tight),
        prompt=make_prompt(),
        robot=robot,
        policy=SequencePolicy(horizon=2),
    )
    runtime.policy_step(now_s=4.0)

    first = runtime.servo_step(now_s=4.0)
    rejected = runtime.servo_step(now_s=4.02)

    assert first.safety.accepted is True  # first row equals the upper boundary
    assert rejected.safety.accepted is False
    assert runtime.state is RuntimeState.RECOVERY
    with pytest.raises(RuntimeErrorState, match="recovery"):
        runtime.servo_step(now_s=4.04)


def test_out_of_limit_action_enters_recovery_without_send() -> None:
    class UnsafePolicy:
        def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
            del snapshot
            return ActionChunk(
                target_joint_position=np.full((2, ACTION_DIM), 10.0, dtype=np.float32),
                dt_s=0.02,
                created_at_s=now_s,
            )

    tight = SafetyConfig(
        joint_lower=(-1.0,) * ACTION_DIM,
        joint_upper=(1.0,) * ACTION_DIM,
        max_delta_per_servo_tick=(2.0,) * ACTION_DIM,
    )
    robot = FakeBimanualRobot()
    runtime = SO101WAMRuntime(
        config=config(actuation=True, horizon=2, safety=tight),
        prompt=make_prompt(),
        robot=robot,
        policy=UnsafePolicy(),
    )
    runtime.policy_step(now_s=4.0)

    result = runtime.servo_step(now_s=4.0)

    assert result.safety.accepted is False
    assert result.sent is False
    assert result.pending == 0
    assert robot.sent_actions == []
    assert runtime.state is RuntimeState.RECOVERY
    with pytest.raises(RuntimeErrorState, match="explicit recovery"):
        runtime.policy_step(now_s=4.1)
    runtime.recover()
    assert runtime.state is RuntimeState.ROLLOUT_READY


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (SequencePolicy(horizon=1), "policy horizon"),
        (SequencePolicy(horizon=2, dt_s=0.1), "servo period"),
    ],
)
def test_bad_policy_contract_fails_closed(policy: SequencePolicy, message: str) -> None:
    runtime = SO101WAMRuntime(
        config=config(actuation=False, horizon=2),
        prompt=make_prompt(),
        robot=FakeBimanualRobot(),
        policy=policy,
    )

    with pytest.raises(RuntimeErrorState, match=message):
        runtime.policy_step(now_s=4.0)

    assert runtime.state is RuntimeState.RECOVERY
    assert runtime.executor.pending == 0


def test_halt_latches_runtime_fault_and_is_terminal() -> None:
    runtime = SO101WAMRuntime(
        config=config(actuation=False, horizon=1),
        prompt=make_prompt(),
        robot=FakeBimanualRobot(),
        policy=SequencePolicy(horizon=1),
    )

    runtime.halt("operator")

    assert runtime.state is RuntimeState.HALT
    assert runtime.safety is not None and runtime.safety.fault_latch.latched
    with pytest.raises(RuntimeErrorState, match="halted"):
        runtime.policy_step(now_s=4.0)


def test_stalled_policy_chunk_keeps_original_age_and_hits_watchdog() -> None:
    robot = FakeBimanualRobot()
    runtime = SO101WAMRuntime(
        config=config(actuation=True, horizon=10),
        prompt=make_prompt(),
        robot=robot,
        policy=SequencePolicy(horizon=10),
    )
    runtime.policy_step(now_s=4.0)

    stale = runtime.servo_step(now_s=4.101)

    assert stale.safety.accepted is False
    assert "stale_action" in stale.safety.reasons
    assert stale.sent is False
    assert runtime.state is RuntimeState.RECOVERY


def test_runtime_rejects_backend_mismatch_before_rollout() -> None:
    with pytest.raises(RuntimeErrorState, match="backend"):
        SO101WAMRuntime(
            config=ProjectConfig(runtime=RuntimeConfig(backend="lerobot", actuation_enabled=False)),
            prompt=make_prompt(),
            robot=FakeBimanualRobot(),
            policy=SequencePolicy(horizon=10),
        )


def test_runtime_rejects_adapter_actuation_mismatch() -> None:
    class ActuationTaggedFake(FakeBimanualRobot):
        actuation_enabled = True

    with pytest.raises(RuntimeErrorState, match="actuation flag"):
        SO101WAMRuntime(
            config=config(actuation=False),
            prompt=make_prompt(),
            robot=ActuationTaggedFake(),
            policy=SequencePolicy(horizon=2),
        )
