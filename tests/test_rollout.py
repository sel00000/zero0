from __future__ import annotations

from dataclasses import asdict
import json
import numpy as np
import pytest
from time import perf_counter
from types import SimpleNamespace

import so101_wam.rollout as rollout_mod
from so101_wam.adapters.fake import FakeBimanualRobot
from so101_wam.adapters.lerobot import LeRobotBiSOAdapter
from so101_wam.config import LeRobotConfig, ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.constants import ACTION_DIM, ARM_JOINT_NAMES
from so101_wam.context import ContextSnapshot
from so101_wam.contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from so101_wam.rollout import (
    ExecutabilityTrace,
    RolloutError,
    SafetyRejectedError,
    run_managed_rollout,
    validate_executability_trace,
    validate_mujoco_scored_trace,
    validate_scored_trace,
)
from so101_wam.runtime import RuntimeState, SO101WAMRuntime


class FakeClock:
    def __init__(self, start_s: float = 100.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s


class AliveThread:
    def is_alive(self) -> bool:
        return True


class PinnedBus:
    def __init__(self, fake: FakeBimanualRobot, *, offset: int) -> None:
        self.fake = fake
        self.offset = offset

    def sync_read(self, data_name: str, *, num_retry: int = 0) -> dict[str, float]:
        assert data_name == "Present_Position"
        assert num_retry == 0
        values = self.fake.joint_position[self.offset : self.offset + 6]
        return {
            joint: float(value)
            for joint, value in zip(ARM_JOINT_NAMES, values, strict=True)
        }


class PinnedFrameLock:
    def __init__(self, camera) -> None:
        self.camera = camera

    def __enter__(self):
        self.camera.refresh()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


class PinnedCamera:
    def __init__(
        self,
        fake: FakeBimanualRobot,
        *,
        key: str,
        timestamp_offset_s: float,
    ) -> None:
        self.fake = fake
        self.key = key
        self.timestamp_offset_s = timestamp_offset_s
        self.latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.latest_timestamp = perf_counter()
        self.thread = AliveThread()
        self.frame_lock = PinnedFrameLock(self)
        self.is_connected = True

    def refresh(self) -> None:
        self.latest_frame = self.fake.get_lerobot_observation()[self.key]
        self.latest_timestamp = perf_counter() + self.timestamp_offset_s

    def disconnect(self) -> None:
        self.is_connected = False
        self.thread = None


class DummyBiSOFollower:
    def __init__(self) -> None:
        self.fake = FakeBimanualRobot()
        self.is_connected = False
        self.is_calibrated = True
        self.connect_calibrate: list[bool] = []
        self.disconnect_calls = 0
        self.left_arm = SimpleNamespace(
            bus=PinnedBus(self.fake, offset=0),
            cameras={
                "wrist": PinnedCamera(
                    self.fake,
                    key="left_wrist",
                    timestamp_offset_s=-0.005,
                )
            },
            config=SimpleNamespace(num_read_retries=0),
        )
        self.right_arm = SimpleNamespace(
            bus=PinnedBus(self.fake, offset=6),
            cameras={
                "wrist": PinnedCamera(
                    self.fake,
                    key="right_wrist",
                    timestamp_offset_s=0.0,
                )
            },
            config=SimpleNamespace(num_read_retries=0),
        )

    def connect(self, *, calibrate: bool = True) -> None:
        self.connect_calibrate.append(calibrate)
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    def get_observation(self) -> dict[str, object]:
        return self.fake.get_lerobot_observation()

    def get_atomic_observation(
        self,
    ) -> tuple[dict[str, object], dict[str, float]]:
        now_s = perf_counter()
        return self.fake.get_lerobot_observation(), {
            "left_wrist": now_s - 0.005,
            "right_wrist": now_s,
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self.fake.send_action(action)


class RampPolicy:
    required_history_steps = 4

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        current = snapshot.live_frames[-1].joint_position
        targets = np.stack(
            [current + 0.1 * (index + 1) for index in range(self.horizon)]
        ).astype(np.float32)
        return ActionChunk(target_joint_position=targets, dt_s=0.02, created_at_s=now_s)


def _frame(timestamp_s: float, value: float) -> SensorimotorFrame:
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={
            "left_wrist": np.full((8, 8, 3), int(value), dtype=np.uint8),
            "right_wrist": np.full((8, 8, 3), int(value) + 1, dtype=np.uint8),
        },
        joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
        executed_action=np.full(ACTION_DIM, value, dtype=np.float32),
    )


def _lerobot_config() -> LeRobotConfig:
    return LeRobotConfig(
        left_port="left",
        right_port="right",
        left_wrist_camera=0,
        right_wrist_camera=1,
        calibration_dir="calibration",
        hardware_id="bench-a",
        calibration_id="cal-a",
        home_joint_position=(0.0,) * ACTION_DIM,
        home_joint_tolerance=(0.1,) * ACTION_DIM,
    )


def _runtime(
    *, actuation: bool
) -> tuple[SO101WAMRuntime, DummyBiSOFollower, LeRobotBiSOAdapter]:
    raw = DummyBiSOFollower()
    lerobot_config = _lerobot_config()
    adapter = LeRobotBiSOAdapter(
        raw, config=lerobot_config, actuation_enabled=actuation
    )
    runtime = SO101WAMRuntime(
        config=ProjectConfig(
            runtime=RuntimeConfig(backend="lerobot", actuation_enabled=actuation),
            safety=SafetyConfig(
                joint_lower=(-10.0,) * ACTION_DIM,
                joint_upper=(10.0,) * ACTION_DIM,
                max_delta_per_servo_tick=(1.0,) * ACTION_DIM,
                calibrated=actuation,
            ),
            lerobot=lerobot_config,
        ),
        prompt=PhysicalPrompt((_frame(0.0, 0.0), _frame(3.0, 0.1))),
        robot=adapter,
        policy=RampPolicy(horizon=10),
    )
    return runtime, raw, adapter


def test_managed_shadow_rollout_connects_primes_runs_and_disconnects() -> None:
    runtime, raw, adapter = _runtime(actuation=False)
    clock = FakeClock()

    summary = run_managed_rollout(
        runtime,
        adapter,
        policy_steps=2,
        clock=clock,
        sleeper=clock.sleep,
    )

    assert raw.connect_calibrate == [False]
    assert raw.disconnect_calls == 1
    assert adapter.is_connected is False
    assert summary.policy_steps == 2
    assert summary.servo_steps == 10
    assert summary.sent_actions == 0
    assert summary.shadow_steps == 10
    assert summary.final_state == RuntimeState.ROLLOUT_READY.value
    assert set(asdict(summary)) == {
        "prompt_fingerprint",
        "policy_steps",
        "servo_steps",
        "sent_actions",
        "shadow_steps",
        "elapsed_s",
        "final_state",
    }


def test_managed_rollout_captures_bounded_executability_trace() -> None:
    runtime, raw, adapter = _runtime(actuation=False)
    clock = FakeClock()
    trace = ExecutabilityTrace("heldout:seed:7", action_limit=1)

    summary = run_managed_rollout(
        runtime,
        adapter,
        policy_steps=2,
        clock=clock,
        sleeper=clock.sleep,
        rollout_observer=trace,
    )
    payload = trace.snapshot()

    assert summary.policy_steps == 2
    assert raw.fake.sent_actions == []
    assert payload["rollout_id"] == "heldout:seed:7"
    assert payload["policy_steps"] == 2
    assert payload["servo_steps"] == 10
    assert payload["sent_actions"] == 0
    assert payload["shadow_steps"] == 10
    assert payload["safety_accepted_steps"] == 10
    assert payload["safety_rejected_steps"] == 0
    assert payload["safety_clipped_steps"] == 0
    assert payload["safety_reason_counts"] == {}
    assert payload["decoded_action_count"] == 2
    assert payload["decoded_actions_truncated"] is True
    assert len(payload["decoded_actions"]) == 1
    assert payload["decoded_actions"][0]["action_shape"] == [10, ACTION_DIM]
    assert len(payload["decoded_actions"][0]["action_sha256"]) == 64
    assert len(payload["policy_action_stream_sha256"]) == 64
    assert len(payload["servo_target_stream_sha256"]) == 64
    assert len(payload["executed_action_stream_sha256"]) == 64
    assert json.loads(json.dumps(payload)) == payload
    assert (
        validate_executability_trace(
            payload,
            rollout_id="heldout:seed:7",
        )
        == payload
    )
    validate_scored_trace(payload, policy_steps=2)

    with pytest.raises(RolloutError, match="policy step count mismatch"):
        validate_scored_trace(payload, policy_steps=1)

    no_servo = {
        **payload,
        "servo_steps": 0,
        "shadow_steps": 0,
        "safety_accepted_steps": 0,
    }
    validate_executability_trace(no_servo, rollout_id="heldout:seed:7")
    with pytest.raises(RolloutError, match="no servo steps"):
        validate_scored_trace(no_servo, policy_steps=2)

    rejected = {
        **payload,
        "safety_accepted_steps": 9,
        "safety_rejected_steps": 1,
    }
    validate_executability_trace(rejected, rollout_id="heldout:seed:7")
    with pytest.raises(RolloutError, match="safety rejection"):
        validate_scored_trace(rejected, policy_steps=2)

    uncovered = {**payload, "shadow_steps": 9}
    validate_executability_trace(uncovered, rollout_id="heldout:seed:7")
    with pytest.raises(RolloutError, match="execution coverage mismatch"):
        validate_scored_trace(uncovered, policy_steps=2)

    invalid = {**payload, "safety_rejected_steps": 1}
    with pytest.raises(RolloutError, match="safety step counts mismatch"):
        validate_executability_trace(invalid, rollout_id="heldout:seed:7")


def test_trace_records_joint_margins_and_contacts() -> None:
    trace = ExecutabilityTrace(
        "diagnostic:seed:7",
        joint_lower=(-10.0,) * ACTION_DIM,
        joint_upper=(10.0,) * ACTION_DIM,
        contact_limit=1,
    )
    policy_target = np.zeros((1, ACTION_DIM), dtype=np.float32)
    policy_target[0, 0] = 8.0
    policy_action = ActionChunk(
        target_joint_position=policy_target,
        dt_s=0.02,
        created_at_s=1.0,
    )
    trace.on_policy_step(
        policy_index=0,
        step=SimpleNamespace(action=policy_action),
    )

    observed = np.zeros(ACTION_DIM, dtype=np.float32)
    observed[0] = -5.0
    servo_target = np.zeros((1, ACTION_DIM), dtype=np.float32)
    servo_target[0, 0] = 6.0
    executed = np.zeros(ACTION_DIM, dtype=np.float32)
    executed[0] = 4.0
    trace.on_servo_step(
        policy_index=0,
        servo_index=0,
        step=SimpleNamespace(
            action=ActionChunk(
                target_joint_position=servo_target,
                dt_s=0.02,
                created_at_s=1.0,
            ),
            observation=SimpleNamespace(joint_position=observed),
            executed_action=executed,
            sent=True,
            shadow=False,
            safety=SimpleNamespace(accepted=True, clipped=False, reasons=()),
        ),
    )
    trace.on_contact_sample(
        policy_index=0,
        servo_index=0,
        phase="servo",
        contacts=(
            {
                "geom1": "right_wrist",
                "geom2": "torso",
                "body1": "right_wrist",
                "body2": "torso",
                "category1": "right_arm",
                "category2": "torso",
                "distance": -0.001,
                "forbidden": True,
            },
        ),
    )
    trace.on_contact_sample(
        policy_index=0,
        servo_index=1,
        phase="servo",
        contacts=(),
    )

    payload = trace.snapshot()

    assert payload["schema_version"] == 3
    assert payload["joint_limit_margin_contract"] == (
        "nearest_bound_margin_over_span_v1"
    )
    assert payload["joint_limit_margins"] == {
        "decoded_policy_target_min": pytest.approx(0.1),
        "servo_safe_target_min": pytest.approx(0.2),
        "measured_joint_min": pytest.approx(0.25),
        "executed_action_min": pytest.approx(0.3),
    }
    assert payload["contact_progression_count"] == 2
    assert payload["contact_progression_truncated"] is True
    assert len(payload["contact_progression"]) == 1
    sample = payload["contact_progression"][0]
    assert sample["category_pairs"] == [["right_arm", "torso"]]
    assert sample["forbidden_contact_count"] == 1
    assert sample["minimum_distance_m"] == pytest.approx(-0.001)
    assert payload["runtime_future_latent_proxy_available"] is False
    assert payload["runtime_future_latent_success_claimed"] is False
    assert validate_executability_trace(
        payload,
        rollout_id="diagnostic:seed:7",
    ) == payload


def test_trace_records_time_aligned_future_latent_mse() -> None:
    trace = ExecutabilityTrace("future-latent:seed:7", future_latent_limit=2)
    observed = np.zeros((2, 3), dtype=np.float32)
    action = ActionChunk(
        target_joint_position=np.zeros((1, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=10.0,
    )

    for policy_index in range(3):
        trace.on_policy_step(
            policy_index=policy_index,
            step=SimpleNamespace(action=action),
        )
        future = np.stack(
            (
                np.full((2, 3), 1.0, dtype=np.float32),
                np.full((2, 3), 2.0, dtype=np.float32),
            )
        )
        trace.on_future_latent_step(
            policy_index=policy_index,
            observation_timestamp_s=10.0 + policy_index * 0.1,
            future_latents=future,
            observed_latent=observed,
        )

    payload = trace.snapshot()

    assert payload["schema_version"] == 3
    assert payload["future_latent_alignment_contract"] == (
        "next_policy_observation_compact_live_encoder_mse_v1"
    )
    assert payload["future_latent_prediction_count"] == 3
    assert payload["future_latent_observation_count"] == 3
    assert payload["future_latent_aligned_pair_count"] == 3
    assert payload["future_latent_censored_pair_count"] == 3
    assert payload["future_latent_element_count"] == 18
    assert payload["future_latent_squared_error_sum"] == pytest.approx(36.0)
    assert payload["future_latent_mse_mean"] == pytest.approx(2.0)
    assert payload["future_latent_mse_by_offset"] == {
        "1": {
            "aligned_pair_count": 2,
            "element_count": 12,
            "squared_error_sum": pytest.approx(12.0),
            "mse_mean": pytest.approx(1.0),
        },
        "2": {
            "aligned_pair_count": 1,
            "element_count": 6,
            "squared_error_sum": pytest.approx(24.0),
            "mse_mean": pytest.approx(4.0),
        },
    }
    assert payload["future_latent_progression_truncated"] is True
    assert len(payload["future_latent_progression"]) == 2
    assert payload["runtime_future_latent_proxy_available"] is True
    assert payload["runtime_future_latent_success_claimed"] is False
    assert validate_executability_trace(
        payload,
        rollout_id="future-latent:seed:7",
    ) == payload

    tampered = {**payload, "future_latent_aligned_pair_count": 2}
    with pytest.raises(RolloutError, match="future latent"):
        validate_executability_trace(
            tampered,
            rollout_id="future-latent:seed:7",
        )

    tampered_index = {
        **payload,
        "future_latent_index_stream_sha256": "0" * 64,
    }
    with pytest.raises(RolloutError, match="index hash"):
        validate_executability_trace(
            tampered_index,
            rollout_id="future-latent:seed:7",
        )


def test_trace_future_latent_rejects_invalid_time_atomically() -> None:
    trace = ExecutabilityTrace("future-latent-invalid-time:seed:7")
    action = ActionChunk(
        target_joint_position=np.zeros((1, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=10.0,
    )
    trace.on_policy_step(
        policy_index=0,
        step=SimpleNamespace(action=action),
    )

    with pytest.raises(RolloutError, match="timestamp"):
        trace.on_future_latent_step(
            policy_index=0,
            observation_timestamp_s=float("nan"),
            future_latents=np.zeros((1, 2, 3), dtype=np.float32),
            observed_latent=np.zeros((2, 3), dtype=np.float32),
        )

    payload = trace.snapshot()

    assert payload["future_latent_prediction_count"] == 0
    assert payload["future_latent_observation_count"] == 0
    assert payload["future_latent_future_steps"] is None
    assert payload["future_latent_latent_dim"] is None
    assert validate_executability_trace(
        payload,
        rollout_id="future-latent-invalid-time:seed:7",
    ) == payload


def test_trace_future_latent_rejects_nonmonotonic_time_atomically() -> None:
    trace = ExecutabilityTrace("future-latent-nonmonotonic-time:seed:7")
    action = ActionChunk(
        target_joint_position=np.zeros((1, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=10.0,
    )
    future = np.zeros((2, 2, 3), dtype=np.float32)
    observed = np.zeros((2, 3), dtype=np.float32)

    for policy_index in range(2):
        trace.on_policy_step(
            policy_index=policy_index,
            step=SimpleNamespace(action=action),
        )

    trace.on_future_latent_step(
        policy_index=0,
        observation_timestamp_s=10.0,
        future_latents=future,
        observed_latent=observed,
    )
    before = trace.snapshot()

    for rejected_timestamp_s in (10.0, 9.0):
        with pytest.raises(RolloutError, match="timestamp"):
            trace.on_future_latent_step(
                policy_index=1,
                observation_timestamp_s=rejected_timestamp_s,
                future_latents=future,
                observed_latent=observed,
            )
        assert trace.snapshot() == before

    trace.on_future_latent_step(
        policy_index=1,
        observation_timestamp_s=10.1,
        future_latents=future,
        observed_latent=observed,
    )
    after = trace.snapshot()

    assert after["future_latent_prediction_count"] == 2
    assert after["future_latent_aligned_pair_count"] == 1
    assert validate_executability_trace(
        after,
        rollout_id="future-latent-nonmonotonic-time:seed:7",
    ) == after


def test_trace_future_latent_skips_saturated_progression_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = ExecutabilityTrace(
        "future-latent-saturated-hashing:seed:7",
        future_latent_limit=1,
    )
    action = ActionChunk(
        target_joint_position=np.zeros((1, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=10.0,
    )
    future = np.zeros((2, 2, 3), dtype=np.float32)
    observed = np.zeros((2, 3), dtype=np.float32)
    original_array_source = rollout_mod._array_source
    call_shapes: list[tuple[int, ...]] = []

    def counted_array_source(value: object) -> bytes:
        call_shapes.append(tuple(np.asarray(value).shape))
        return original_array_source(value)

    monkeypatch.setattr(rollout_mod, "_array_source", counted_array_source)

    for policy_index in range(3):
        trace.on_policy_step(
            policy_index=policy_index,
            step=SimpleNamespace(action=action),
        )

    trace.on_future_latent_step(
        policy_index=0,
        observation_timestamp_s=10.0,
        future_latents=future,
        observed_latent=observed,
    )
    trace.on_future_latent_step(
        policy_index=1,
        observation_timestamp_s=10.1,
        future_latents=future,
        observed_latent=observed,
    )
    assert len(trace.snapshot()["future_latent_progression"]) == 1

    call_shapes.clear()
    trace.on_future_latent_step(
        policy_index=2,
        observation_timestamp_s=10.2,
        future_latents=future,
        observed_latent=observed,
    )
    payload = trace.snapshot()

    assert call_shapes == [(2, 2, 3), (2, 3)]
    assert payload["future_latent_aligned_pair_count"] == 3
    assert payload["future_latent_mse_by_offset"]["1"]["aligned_pair_count"] == 2
    assert payload["future_latent_mse_by_offset"]["2"]["aligned_pair_count"] == 1
    assert validate_executability_trace(
        payload,
        rollout_id="future-latent-saturated-hashing:seed:7",
    ) == payload


def test_mujoco_scored_trace_rejects_duplicate_contact_indexes() -> None:
    trace = ExecutabilityTrace(
        "duplicate-contact:seed:7",
        joint_lower=(-10.0,) * ACTION_DIM,
        joint_upper=(10.0,) * ACTION_DIM,
    )
    target = np.zeros((1, ACTION_DIM), dtype=np.float32)
    action = ActionChunk(
        target_joint_position=target,
        dt_s=0.02,
        created_at_s=1.0,
    )
    trace.on_policy_step(
        policy_index=0,
        step=SimpleNamespace(action=action),
    )
    for servo_index in range(2):
        trace.on_servo_step(
            policy_index=0,
            servo_index=servo_index,
            step=SimpleNamespace(
                action=action,
                observation=SimpleNamespace(joint_position=target[0]),
                executed_action=target[0],
                sent=True,
                shadow=False,
                safety=SimpleNamespace(accepted=True, clipped=False, reasons=()),
            ),
        )
        trace.on_contact_sample(
            policy_index=0,
            servo_index=0,
            phase="servo",
            contacts=(),
        )
    payload = trace.snapshot()

    validate_executability_trace(
        payload,
        rollout_id="duplicate-contact:seed:7",
    )
    with pytest.raises(RolloutError, match="contact index stream mismatch"):
        validate_mujoco_scored_trace(payload, policy_steps=1)


def test_managed_actuated_rollout_sends_one_row_per_servo_tick() -> None:
    runtime, raw, adapter = _runtime(actuation=True)
    clock = FakeClock()

    summary = run_managed_rollout(
        runtime,
        adapter,
        policy_steps=1,
        clock=clock,
        sleeper=clock.sleep,
    )

    assert summary.servo_steps == summary.sent_actions == 5
    assert len(raw.fake.sent_actions) == 5
    assert raw.disconnect_calls == 1


def test_managed_rollout_reports_terminal_observation_before_disconnect() -> None:
    runtime, _, adapter = _runtime(actuation=True)
    clock = FakeClock()
    captured: list[tuple[float, bool]] = []

    def capture(timestamp_s: float) -> None:
        captured.append((timestamp_s, adapter.is_connected))

    run_managed_rollout(
        runtime,
        adapter,
        policy_steps=1,
        clock=clock,
        sleeper=clock.sleep,
        terminal_observer=capture,
    )

    assert len(captured) == 1
    terminal_timestamp_s, connected_during_capture = captured[0]
    assert connected_during_capture is True
    assert terminal_timestamp_s == clock.now_s
    assert adapter.is_connected is False


def test_managed_actuated_rollout_blocks_pose_outside_home_tolerance() -> None:
    runtime, raw, adapter = _runtime(actuation=True)
    raw.fake.joint_position[0] = 0.2
    clock = FakeClock()

    with pytest.raises(RolloutError, match="outside home tolerance"):
        run_managed_rollout(
            runtime, adapter, policy_steps=1, clock=clock, sleeper=clock.sleep
        )

    assert raw.fake.sent_actions == []
    assert raw.disconnect_calls == 1
    assert adapter.is_connected is False
    assert runtime.state is RuntimeState.HALT


def test_managed_rollout_disconnects_and_halts_on_policy_failure() -> None:
    runtime, raw, adapter = _runtime(actuation=False)
    clock = FakeClock()

    class BrokenPolicy:
        required_history_steps = 1

        def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
            del snapshot, now_s
            raise RuntimeError("boom")

    runtime.policy = BrokenPolicy()
    with pytest.raises(RuntimeError, match="boom"):
        run_managed_rollout(
            runtime, adapter, policy_steps=1, clock=clock, sleeper=clock.sleep
        )

    assert raw.disconnect_calls == 1
    assert adapter.is_connected is False
    assert runtime.state is RuntimeState.HALT
    assert runtime.safety is not None and runtime.safety.fault_latch.latched


def test_managed_rollout_measures_capture_latency_and_rejects_stale_frames() -> None:
    runtime, raw, adapter = _runtime(actuation=False)
    clock = FakeClock()
    original_get_observation = raw.get_atomic_observation

    def slow_observation() -> tuple[dict[str, object], dict[str, float]]:
        observation = original_get_observation()
        clock.now_s += 0.2
        return observation

    raw.get_atomic_observation = slow_observation  # type: ignore[method-assign]
    trace = ExecutabilityTrace("watchdog:seed:7")

    with pytest.raises(SafetyRejectedError, match="safety rejected") as raised:
        run_managed_rollout(
            runtime,
            adapter,
            policy_steps=1,
            clock=clock,
            sleeper=clock.sleep,
            rollout_observer=trace,
        )

    assert raised.value.reasons == ("stale_observation", "stale_action")
    assert trace.snapshot()["safety_rejected_steps"] == 1
    assert trace.snapshot()["safety_reason_counts"] == {
        "stale_action": 1,
        "stale_observation": 1,
    }
    assert raw.disconnect_calls == 1
    assert runtime.state is RuntimeState.HALT
