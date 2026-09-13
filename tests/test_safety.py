from __future__ import annotations

import numpy as np
import pytest

from so101_wam.config import SafetyConfig
from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import ActionChunk, ContractError, SensorimotorFrame
from so101_wam.safety import FaultLatch, SafetyError, SafetySupervisor


def make_frame(
    timestamp_s: float,
    joints: np.ndarray | None = None,
    *,
    camera_skew_s: float | None = None,
) -> SensorimotorFrame:
    image_timestamps_s = (
        None
        if camera_skew_s is None
        else {"left_wrist": 10.0, "right_wrist": 10.0 + camera_skew_s}
    )
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={
            "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_wrist": np.ones((8, 8, 3), dtype=np.uint8),
        },
        joint_position=np.zeros(ACTION_DIM, dtype=np.float32)
        if joints is None
        else joints,
        image_timestamps_s=image_timestamps_s,
    )


def make_chunk(created_at_s: float, targets: np.ndarray) -> ActionChunk:
    return ActionChunk(
        target_joint_position=targets,
        dt_s=0.02,
        created_at_s=created_at_s,
    )


def test_runtime_fault_is_latched_until_explicit_reset() -> None:
    fault = FaultLatch()

    fault.trigger("operator")
    assert fault.latched is True
    assert fault.reason == "operator"

    fault.reset()
    assert fault.latched is False
    assert fault.reason is None
    assert fault.reset_count == 1


def test_safety_supervisor_rejects_latched_fault_with_hold_action() -> None:
    supervisor = SafetySupervisor(SafetyConfig.fake_normalized())
    supervisor.trigger_fault("test")

    decision = supervisor.evaluate(
        make_frame(1.0),
        make_chunk(1.0, np.ones((2, ACTION_DIM), dtype=np.float32)),
        now_s=1.01,
    )

    assert decision.accepted is False
    assert decision.hold is True
    assert decision.fault_latched is True
    assert decision.reasons[:2] == (
        "runtime_fault_latched",
        "runtime_fault_reason:test",
    )
    np.testing.assert_allclose(
        decision.action.target_joint_position[0], np.zeros(ACTION_DIM)
    )

    supervisor.reset_fault()
    cleared = supervisor.evaluate(
        make_frame(1.0),
        make_chunk(1.0, np.ones((1, ACTION_DIM), dtype=np.float32)),
        now_s=1.01,
    )
    assert cleared.accepted is True


def test_safety_supervisor_rejects_stale_observation_and_action() -> None:
    supervisor = SafetySupervisor(SafetyConfig.fake_normalized())

    decision = supervisor.evaluate(
        make_frame(0.0),
        make_chunk(0.0, np.zeros((1, ACTION_DIM), dtype=np.float32)),
        now_s=0.2,
    )

    assert decision.accepted is False
    assert decision.hold is True
    assert "stale_observation" in decision.reasons
    assert "stale_action" in decision.reasons


def test_safety_supervisor_rejects_excessive_wrist_camera_skew() -> None:
    supervisor = SafetySupervisor(SafetyConfig.fake_normalized())

    decision = supervisor.evaluate(
        make_frame(1.0, camera_skew_s=0.018),
        make_chunk(1.0, np.zeros((1, ACTION_DIM), dtype=np.float32)),
        now_s=1.01,
    )

    assert decision.accepted is False
    assert decision.hold is True
    assert "camera_skew" in decision.reasons


def test_safety_supervisor_rejects_materially_future_observation_and_action() -> None:
    supervisor = SafetySupervisor(SafetyConfig.fake_normalized())

    decision = supervisor.evaluate(
        make_frame(1.018),
        make_chunk(1.031, np.zeros((1, ACTION_DIM), dtype=np.float32)),
        now_s=1.0,
    )

    assert decision.accepted is False
    assert decision.hold is True
    assert "future_observation" in decision.reasons
    assert "future_action" in decision.reasons


def test_safety_supervisor_allows_tiny_future_timestamp_tolerance() -> None:
    supervisor = SafetySupervisor(SafetyConfig.fake_normalized())

    decision = supervisor.evaluate(
        make_frame(1.017),
        make_chunk(1.02, np.zeros((1, ACTION_DIM), dtype=np.float32)),
        now_s=1.0,
    )

    assert decision.accepted is True
    assert decision.hold is False


def test_safety_supervisor_rejects_hard_joint_limit_violations() -> None:
    supervisor = SafetySupervisor(SafetyConfig.fake_normalized())
    target = np.zeros((1, ACTION_DIM), dtype=np.float32)
    target[0, 0] = 101.0

    decision = supervisor.evaluate(make_frame(1.0), make_chunk(1.0, target), now_s=1.01)

    assert decision.accepted is False
    assert decision.hold is True
    assert decision.reasons == ("action[0]_joint_limit:left_shoulder_pan.pos:above",)


def test_safety_supervisor_clips_per_servo_tick_and_audits_reason() -> None:
    supervisor = SafetySupervisor(SafetyConfig.fake_normalized())
    targets = np.array([[10.0] * ACTION_DIM, [10.0] * ACTION_DIM], dtype=np.float32)

    decision = supervisor.evaluate(
        make_frame(1.0), make_chunk(1.0, targets), now_s=1.01
    )

    assert decision.accepted is True
    assert decision.clipped is True
    assert decision.reasons == ("max_delta_clipped",)
    np.testing.assert_allclose(
        decision.action.target_joint_position[0, :5], np.full(5, 2.0)
    )
    np.testing.assert_allclose(
        decision.action.target_joint_position[1, :5], np.full(5, 4.0)
    )
    assert decision.action.target_joint_position[0, -1] == 4.0


def test_safety_uses_contracts_to_reject_nan_and_bad_config() -> None:
    with pytest.raises(ContractError, match="NaN"):
        make_frame(1.0, np.full(ACTION_DIM, np.nan, dtype=np.float32))

    config = SafetyConfig(
        joint_lower=(0.0,) * ACTION_DIM,
        joint_upper=(1.0,) * ACTION_DIM,
        max_delta_per_servo_tick=(1.0,) * (ACTION_DIM - 1) + (float("inf"),),
    )
    with pytest.raises(SafetyError, match="NaN or infinity"):
        SafetySupervisor(config)
