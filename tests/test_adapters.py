from __future__ import annotations

import importlib
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

from so101_wam.adapters import (
    ActuationDisabledError,
    FakeBimanualRobot,
    FakeWristCameras,
    LeRobotBiSOAdapter,
    action_to_lerobot,
    observation_from_lerobot,
    read_lerobot_observation_atomic,
)
from so101_wam.config import LeRobotConfig
from so101_wam.constants import (
    ACTION_DIM,
    ARM_JOINT_NAMES,
    JOINT_KEYS,
    PRIMARY_CAMERA_KEYS,
)
from so101_wam.contracts import ActionChunk, ContractError


class _AliveThread:
    def is_alive(self) -> bool:
        return True


class _FrameLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


class _PinnedCamera:
    def __init__(self, value: int, timestamp_s: float) -> None:
        self.latest_frame = np.full((8, 8, 3), value, dtype=np.uint8)
        self.latest_timestamp = timestamp_s
        self.is_connected = True
        self.thread = _AliveThread()
        self.frame_lock = _FrameLock()

    def disconnect(self) -> None:
        self.is_connected = False
        self.thread = None


class _PinnedBus:
    def sync_read(self, data_name: str, *, num_retry: int = 0) -> dict[str, float]:
        assert data_name == "Present_Position"
        assert num_retry == 0
        return {joint: 0.0 for joint in ARM_JOINT_NAMES}


def test_fake_wrist_cameras_are_deterministic_and_distinct() -> None:
    cameras = FakeWristCameras(height=4, width=5)

    first = cameras.capture(frame_index=3)
    second = cameras.capture(frame_index=3)

    assert tuple(first) == PRIMARY_CAMERA_KEYS
    np.testing.assert_array_equal(first["left_wrist"], second["left_wrist"])
    assert first["left_wrist"].shape == (4, 5, 3)
    assert not np.array_equal(first["left_wrist"], first["right_wrist"])


def test_fake_bimanual_robot_round_trips_lerobot_actions() -> None:
    robot = FakeBimanualRobot()
    action = {key: float(index) for index, key in enumerate(JOINT_KEYS)}

    returned = robot.send_action(action)
    observation = robot.get_observation(timestamp_s=1.0)

    np.testing.assert_allclose(
        observation.joint_position, np.arange(ACTION_DIM, dtype=np.float32)
    )
    assert returned == action
    assert len(robot.sent_actions) == 1


def test_fake_bimanual_robot_rejects_multi_step_action_chunks() -> None:
    robot = FakeBimanualRobot()
    chunk = ActionChunk(
        target_joint_position=np.ones((2, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=1.0,
    )

    with pytest.raises(ValueError, match="horizon must be 1"):
        robot.send_action(chunk)

    assert robot.sent_actions == []


def test_lerobot_module_does_not_import_lerobot_at_module_import() -> None:
    sys.modules.pop("so101_wam.adapters.lerobot", None)
    sys.modules.pop("lerobot", None)

    importlib.import_module("so101_wam.adapters.lerobot")

    assert "lerobot" not in sys.modules


def test_lerobot_observation_mapping_requires_exact_joint_and_camera_keys() -> None:
    robot = FakeBimanualRobot()
    raw = robot.get_lerobot_observation()

    frame = observation_from_lerobot(raw, timestamp_s=2.0)

    assert frame.timestamp_s == 2.0
    assert tuple(frame.images) == PRIMARY_CAMERA_KEYS
    np.testing.assert_allclose(frame.joint_position, np.zeros(ACTION_DIM))

    bad = dict(raw)
    bad["head_optional"] = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ContractError, match="extra"):
        observation_from_lerobot(bad, timestamp_s=2.0)


def test_atomic_lerobot_read_keeps_each_frame_tied_to_its_locked_timestamp() -> None:
    class AliveThread:
        def is_alive(self) -> bool:
            return True

    class MutateOnExitLock:
        def __init__(self, camera) -> None:
            self.camera = camera

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            self.camera.latest_frame = np.full((8, 8, 3), 99, dtype=np.uint8)
            self.camera.latest_timestamp += 1.0

    class Camera:
        def __init__(self, value: int, timestamp_s: float) -> None:
            self.latest_frame = np.full((8, 8, 3), value, dtype=np.uint8)
            self.latest_timestamp = timestamp_s
            self.thread = AliveThread()
            self.frame_lock = MutateOnExitLock(self)

    class Bus:
        def sync_read(
            self, data_name: str, *, num_retry: int = 0
        ) -> dict[str, float]:
            assert data_name == "Present_Position"
            assert num_retry == 0
            return {joint: 0.0 for joint in ARM_JOINT_NAMES}

    left_camera = Camera(10, 100.000)
    right_camera = Camera(20, 100.005)
    robot = SimpleNamespace(
        left_arm=SimpleNamespace(
            bus=Bus(),
            cameras={"wrist": left_camera},
            config=SimpleNamespace(num_read_retries=0),
        ),
        right_arm=SimpleNamespace(
            bus=Bus(),
            cameras={"wrist": right_camera},
            config=SimpleNamespace(num_read_retries=0),
        ),
    )

    frame = read_lerobot_observation_atomic(
        robot,
        timestamp_s=7.0,
        camera_clock=lambda: 100.010,
        max_camera_age_s=0.1,
        allow_external_provider=False,
    )

    assert np.all(frame.images["left_wrist"] == 10)
    assert np.all(frame.images["right_wrist"] == 20)
    assert frame.image_timestamps_s == {
        "left_wrist": 100.000,
        "right_wrist": 100.005,
    }
    assert np.all(left_camera.latest_frame == 99)
    assert left_camera.latest_timestamp == pytest.approx(101.000)


def test_atomic_lerobot_read_rejects_stale_camera_buffers() -> None:
    fake = FakeBimanualRobot()

    class AtomicProvider:
        def get_atomic_observation(
            self,
        ) -> tuple[dict[str, object], dict[str, float]]:
            return fake.get_lerobot_observation(), {
                "left_wrist": 9.7,
                "right_wrist": 9.8,
            }

    with pytest.raises(ContractError, match="camera frame is stale"):
        read_lerobot_observation_atomic(
            AtomicProvider(),
            timestamp_s=10.0,
            camera_clock=lambda: 10.0,
            max_camera_age_s=0.1,
        )


def test_lerobot_action_mapping_uses_exact_12_joint_keys() -> None:
    chunk = ActionChunk(
        target_joint_position=np.arange(ACTION_DIM, dtype=np.float32).reshape(
            1, ACTION_DIM
        ),
        dt_s=0.02,
        created_at_s=1.0,
    )

    mapped = action_to_lerobot(chunk)

    assert tuple(mapped) == JOINT_KEYS
    assert mapped["left_shoulder_pan.pos"] == 0.0
    assert mapped["right_gripper.pos"] == 11.0


def test_lerobot_action_mapping_rejects_multi_step_action_chunks() -> None:
    chunk = ActionChunk(
        target_joint_position=np.ones((2, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=1.0,
    )

    with pytest.raises(ContractError, match="horizon must be 1"):
        action_to_lerobot(chunk)


def test_lerobot_adapter_fails_closed_until_actuation_is_enabled() -> None:
    robot = FakeBimanualRobot()
    adapter = LeRobotBiSOAdapter(robot)
    action = np.ones(ACTION_DIM, dtype=np.float32)

    with pytest.raises(ActuationDisabledError, match="disabled"):
        adapter.send_action(action)

    assert robot.sent_actions == []


def test_lerobot_adapter_maps_observations_actions_and_send_returns() -> None:
    class DummyBiSOFollower:
        def __init__(self) -> None:
            self.robot = FakeBimanualRobot()
            self.is_connected = False
            self.is_calibrated = True
            now_s = perf_counter()
            self.left_arm = SimpleNamespace(
                bus=_PinnedBus(),
                cameras={"wrist": _PinnedCamera(10, now_s - 0.005)},
                config=SimpleNamespace(num_read_retries=0),
            )
            self.right_arm = SimpleNamespace(
                bus=_PinnedBus(),
                cameras={"wrist": _PinnedCamera(20, now_s)},
                config=SimpleNamespace(num_read_retries=0),
            )

        def connect(self, *, calibrate: bool = True) -> None:
            assert calibrate is False
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

        def get_observation(self) -> dict[str, object]:
            return self.robot.get_lerobot_observation()

        def get_atomic_observation(
            self,
        ) -> tuple[dict[str, object], dict[str, float]]:
            raise AssertionError("real actuation must use the pinned arm path")

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            return self.robot.send_action(action)

    robot = DummyBiSOFollower()
    adapter = LeRobotBiSOAdapter(
        robot,
        config=LeRobotConfig(
            left_port="fake-left",
            right_port="fake-right",
            left_wrist_camera=0,
            right_wrist_camera=1,
            calibration_dir="fake-calibration-dir",
            hardware_id="test-bench",
            calibration_id="test-calibration",
            home_joint_position=(0.0,) * ACTION_DIM,
            home_joint_tolerance=(0.1,) * ACTION_DIM,
        ),
        actuation_enabled=True,
    )

    with pytest.raises(RuntimeError, match="connected"):
        adapter.get_observation(timestamp_s=1.5)
    adapter.connect(calibrate=False)
    frame = adapter.get_observation(timestamp_s=1.5)
    executed = adapter.send_action(np.arange(ACTION_DIM, dtype=np.float32))

    assert adapter.joint_keys == JOINT_KEYS
    assert adapter.camera_keys == PRIMARY_CAMERA_KEYS
    np.testing.assert_allclose(frame.joint_position, np.zeros(ACTION_DIM))
    assert frame.primary_image_skew_s == pytest.approx(0.005)
    assert executed is not None
    np.testing.assert_allclose(executed, np.arange(ACTION_DIM, dtype=np.float32))
    robot.is_calibrated = False
    with pytest.raises(RuntimeError, match="is_calibrated=true"):
        adapter.send_action(np.zeros(ACTION_DIM, dtype=np.float32))
    adapter.disconnect()
    assert adapter.is_connected is False


def test_lerobot_adapter_real_actuation_rejects_external_atomic_provider() -> None:
    class ExternalOnlyRobot:
        def __init__(self) -> None:
            self.fake = FakeBimanualRobot()
            self.is_connected = True
            self.is_calibrated = True
            self.provider_calls = 0

        def connect(self, *, calibrate: bool = True) -> None:
            del calibrate

        def disconnect(self) -> None:
            self.is_connected = False

        def get_observation(self) -> dict[str, object]:
            return self.fake.get_lerobot_observation()

        def get_atomic_observation(
            self,
        ) -> tuple[dict[str, object], dict[str, float]]:
            self.provider_calls += 1
            return self.fake.get_lerobot_observation(), {
                "left_wrist": perf_counter(),
                "right_wrist": perf_counter(),
            }

    robot = ExternalOnlyRobot()
    config = LeRobotConfig(
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
    adapter = LeRobotBiSOAdapter(robot, config=config, actuation_enabled=True)

    with pytest.raises(ContractError, match="real actuation requires atomic LeRobot wrist"):
        adapter.get_observation(timestamp_s=1.0)

    assert robot.provider_calls == 0


def test_lerobot_adapter_requires_camera_timestamps_before_real_actuation() -> None:
    class MissingTimestampRobot:
        def __init__(self) -> None:
            self.fake = FakeBimanualRobot()
            self.is_connected = False
            self.is_calibrated = True

        def connect(self, *, calibrate: bool = True) -> None:
            del calibrate
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

        def get_observation(self) -> dict[str, object]:
            return self.fake.get_lerobot_observation()

    config = LeRobotConfig(
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
    robot = MissingTimestampRobot()
    adapter = LeRobotBiSOAdapter(robot, config=config, actuation_enabled=True)
    adapter.connect(calibrate=False)
    try:
        with pytest.raises(ContractError, match="atomic LeRobot wrist"):
            adapter.get_observation(timestamp_s=1.0)
    finally:
        adapter.disconnect()


def test_lerobot_adapter_blocks_uncalibrated_real_output() -> None:
    class UncalibratedRobot:
        def __init__(self) -> None:
            self.is_connected = False
            self.is_calibrated = False
            self.disconnect_calls = 0
            self.sent_actions: list[dict[str, float]] = []

        def connect(self, *, calibrate: bool = True) -> None:
            del calibrate
            self.is_connected = True

        def disconnect(self) -> None:
            self.disconnect_calls += 1
            self.is_connected = False

        def send_action(self, action: dict[str, float]) -> None:
            self.sent_actions.append(action)

    robot = UncalibratedRobot()
    config = LeRobotConfig(
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
    adapter = LeRobotBiSOAdapter(robot, config=config, actuation_enabled=True)

    with pytest.raises(RuntimeError, match="is_calibrated=true"):
        adapter.connect(calibrate=False)

    assert adapter.is_connected is False
    assert robot.disconnect_calls == 1
    assert robot.sent_actions == []


def test_lerobot_adapter_cleans_up_when_connect_state_is_not_reported() -> None:
    class IncompleteConnectRobot:
        is_connected = False

        def __init__(self) -> None:
            self.disconnect_calls = 0

        def connect(self, *, calibrate: bool = True) -> None:
            del calibrate

        def disconnect(self) -> None:
            self.disconnect_calls += 1

    robot = IncompleteConnectRobot()
    adapter = LeRobotBiSOAdapter(robot)

    with pytest.raises(RuntimeError, match="did not report a connected state"):
        adapter.connect()

    assert robot.disconnect_calls == 1


def test_lerobot_adapter_cleans_left_arm_after_partial_bimanual_connect() -> None:
    class Arm:
        def __init__(self) -> None:
            self.is_connected = False
            self.disconnect_calls = 0

        def disconnect(self) -> None:
            self.disconnect_calls += 1
            self.is_connected = False

    class PartialRobot:
        def __init__(self) -> None:
            self.left_arm = Arm()
            self.right_arm = Arm()
            self.top_disconnect_calls = 0

        @property
        def is_connected(self) -> bool:
            return self.left_arm.is_connected and self.right_arm.is_connected

        def connect(self, *, calibrate: bool = True) -> None:
            del calibrate
            self.left_arm.is_connected = True
            raise RuntimeError("right arm connection failed")

        def disconnect(self) -> None:
            self.top_disconnect_calls += 1
            if not self.is_connected:
                raise RuntimeError("bimanual robot is not fully connected")

    robot = PartialRobot()
    adapter = LeRobotBiSOAdapter(robot)

    with pytest.raises(RuntimeError, match="right arm connection failed"):
        adapter.connect(calibrate=False)

    assert robot.top_disconnect_calls == 1
    assert robot.left_arm.disconnect_calls == 1
    assert robot.left_arm.is_connected is False
    assert robot.right_arm.is_connected is False


def test_lerobot_adapter_disconnect_visits_right_arm_after_left_failure() -> None:
    class Arm:
        def __init__(self, *, fail_first: bool = False) -> None:
            self.is_connected = True
            self.fail_first = fail_first
            self.disconnect_calls = 0

        def disconnect(self) -> None:
            self.disconnect_calls += 1
            if self.fail_first and self.disconnect_calls == 1:
                raise RuntimeError("left disconnect failed")
            self.is_connected = False

    class BimanualRobot:
        def __init__(self) -> None:
            self.left_arm = Arm(fail_first=True)
            self.right_arm = Arm()

        @property
        def is_connected(self) -> bool:
            return self.left_arm.is_connected and self.right_arm.is_connected

        def disconnect(self) -> None:
            self.left_arm.disconnect()
            self.right_arm.disconnect()

    robot = BimanualRobot()
    adapter = LeRobotBiSOAdapter(robot)

    with pytest.raises(RuntimeError, match="independent arm cleanup completed"):
        adapter.disconnect()

    assert robot.left_arm.disconnect_calls == 2
    assert robot.right_arm.disconnect_calls == 1
    assert robot.left_arm.is_connected is False
    assert robot.right_arm.is_connected is False
