from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_wam.config import (
    LeRobotConfig,
    ProjectConfig,
    RuntimeConfig,
    SafetyConfig,
)
from so101_wam.constants import ACTION_DIM, ARM_JOINT_NAMES
from so101_wam.contracts import ContractError
from so101_wam.dataset import (
    DatasetArtifactExistsError,
    load_episode,
    physical_prompt_from_episode,
)
from so101_wam.hardware_preflight import HardwarePreflightError
from so101_wam.prompt_recorder import (
    ACTION_SOURCE,
    CAPTURE_MODE,
    PromptRecorderError,
    record_physical_prompt,
)


class FakeClock:
    def __init__(self, start_s: float = 100.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s


class AlwaysAliveThread:
    def is_alive(self) -> bool:
        return True


class RefreshingLock:
    def __init__(self, refresh) -> None:
        self.refresh = refresh

    def __enter__(self):
        self.refresh()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


class DummyCamera:
    def __init__(
        self,
        *,
        clock: FakeClock,
        timestamp_offset_s: float,
        image_value: int,
        resolution: tuple[int, int] = (8, 8),
        timestamp_jitter_s: tuple[float, ...] = (),
    ) -> None:
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.clock = clock
        self.timestamp_offset_s = timestamp_offset_s
        self.timestamp_jitter_s = timestamp_jitter_s
        self.refresh_count = 0
        self.latest_frame = np.full(
            (*resolution, 3),
            image_value,
            dtype=np.uint8,
        )
        self.latest_timestamp: float | None = None
        self.frame_lock = RefreshingLock(self._refresh)
        self.thread = AlwaysAliveThread()

    def _refresh(self) -> None:
        jitter_s = (
            self.timestamp_jitter_s[
                self.refresh_count % len(self.timestamp_jitter_s)
            ]
            if self.timestamp_jitter_s
            else 0.0
        )
        self.latest_timestamp = self.clock() + self.timestamp_offset_s + jitter_s
        self.refresh_count += 1

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class DummyBus:
    def __init__(self, *, joint_offset: float, on_read=None) -> None:
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_args: list[bool] = []
        self.disable_torque_calls = 0
        self.joint_offset = joint_offset
        self.on_read = on_read

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_args.append(disable_torque)
        self.is_connected = False

    def disable_torque(self) -> None:
        self.disable_torque_calls += 1

    def write(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            f"prompt recorder must not write motor registers: {args}, {kwargs}"
        )

    def sync_write(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            f"prompt recorder must not send Goal_Position: {args}, {kwargs}"
        )

    def sync_read(
        self,
        data_name: str,
        *,
        num_retry: int = 0,
    ) -> dict[str, float]:
        assert data_name == "Present_Position"
        assert num_retry == 0
        if self.on_read is not None:
            self.on_read()
        return {
            joint: self.joint_offset + index
            for index, joint in enumerate(ARM_JOINT_NAMES)
        }


class DummyArm:
    def __init__(
        self,
        *,
        clock: FakeClock,
        timestamp_offset_s: float,
        image_value: int,
        joint_offset: float,
        on_read=None,
        timestamp_jitter_s: tuple[float, ...] = (),
    ) -> None:
        self.bus = DummyBus(joint_offset=joint_offset, on_read=on_read)
        self.cameras = {
            "wrist": DummyCamera(
                clock=clock,
                timestamp_offset_s=timestamp_offset_s,
                image_value=image_value,
                timestamp_jitter_s=timestamp_jitter_s,
            )
        }
        self.config = SimpleNamespace(
            disable_torque_on_disconnect=True,
            num_read_retries=0,
        )

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and self.cameras["wrist"].is_connected

    @property
    def is_calibrated(self) -> bool:
        return True


class DummyBiSOFollower:
    def __init__(
        self,
        clock: FakeClock,
        *,
        capture_latency_s: float = 0.005,
        camera_skew_s: float = 0.005,
        camera_timestamp_jitter_s: tuple[float, ...] = (),
    ) -> None:
        self.clock = clock
        self.capture_latency_s = capture_latency_s
        self.observation_calls = 0
        self.connect_calls = 0
        self.send_calls = 0

        def right_read() -> None:
            self.observation_calls += 1
            self.clock.now_s += self.capture_latency_s

        self.left_arm = DummyArm(
            clock=clock,
            timestamp_offset_s=-camera_skew_s,
            image_value=10,
            joint_offset=0.0,
            timestamp_jitter_s=camera_timestamp_jitter_s,
        )
        self.right_arm = DummyArm(
            clock=clock,
            timestamp_offset_s=0.0,
            image_value=20,
            joint_offset=6.0,
            on_read=right_read,
            timestamp_jitter_s=camera_timestamp_jitter_s,
        )

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, *, calibrate: bool = True) -> None:
        del calibrate
        self.connect_calls += 1
        raise AssertionError("prompt recorder must bypass BiSOFollower.connect()")

    def send_action(self, action: object) -> None:
        del action
        self.send_calls += 1
        raise AssertionError("prompt recorder must never call send_action()")


def _config(
    *,
    actuation: bool = True,
    calibrated: bool = True,
    include_home: bool = True,
) -> ProjectConfig:
    return ProjectConfig(
        runtime=RuntimeConfig(
            backend="lerobot",
            camera_hz=30.0,
            policy_hz=10.0,
            actuation_enabled=actuation,
        ),
        safety=SafetyConfig(
            joint_lower=(-180.0,) * ACTION_DIM,
            joint_upper=(180.0,) * ACTION_DIM,
            max_delta_per_servo_tick=(1.0,) * ACTION_DIM,
            calibrated=calibrated,
        ),
        lerobot=LeRobotConfig(
            left_port="left",
            right_port="right",
            left_wrist_camera=0,
            right_wrist_camera=1,
            camera_width=8,
            camera_height=8,
            camera_fps=30,
            calibration_dir="calibration",
            hardware_id="bench-a",
            calibration_id="cal-a",
            home_joint_position=(
                tuple(float(index) for index in range(ACTION_DIM))
                if include_home
                else None
            ),
            home_joint_tolerance=(1.0,) * ACTION_DIM if include_home else None,
        ),
    )


def test_recorder_creates_no_command_checksum_verified_prompt(tmp_path) -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock)

    result = record_physical_prompt(
        _config(actuation=True),
        output_dir=tmp_path,
        task="place block",
        duration_s=3.0,
        episode_index=7,
        task_index=2,
        robot_factory=lambda config: robot,
        clock=clock,
        camera_clock=clock,
        sleeper=clock.sleep,
    )

    npz_path = tmp_path / "prompt_000007.npz"
    manifest_path = tmp_path / "prompt_000007.json"
    loaded = load_episode(npz_path, manifest_path)
    prompt = physical_prompt_from_episode(loaded, policy_hz=10.0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["result"] == "recorded"
    assert result["round_trip_verified"] is True
    assert result["action_source"] == ACTION_SOURCE
    assert result["capture_mode"] == CAPTURE_MODE
    assert result["goal_position_commands_sent"] == 0
    assert result["torque_disabled_for_capture"] is True
    assert result["recorded_frame_count"] == 91
    assert result["policy_prompt_frame_count"] == 31
    assert result["duration_s"] == pytest.approx(3.0)
    assert result["camera_fps_estimate_hz"] == {
        "left_wrist": pytest.approx(30.0),
        "right_wrist": pytest.approx(30.0),
    }
    assert loaded.frame_count == 91
    assert loaded.metadata["action_source"] == ACTION_SOURCE
    assert loaded.metadata["effective_actuation_enabled"] is False
    assert loaded.metadata["configured_actuation_enabled"] is True
    assert loaded.metadata["head_camera_included"] is False
    assert loaded.metadata["capture_interval_max_relative_error"] < 1e-9
    assert manifest["npz_sha256"] == result["npz_sha256"]
    assert prompt.fingerprint == result["prompt_fingerprint"]
    np.testing.assert_array_equal(loaded.action, loaded.joint_state)
    np.testing.assert_array_equal(
        loaded.joint_state[0],
        np.arange(ACTION_DIM, dtype=np.float32),
    )
    assert np.all(loaded.wrist_rgb[:, 0] == 10)
    assert np.all(loaded.wrist_rgb[:, 1] == 20)

    assert robot.observation_calls == 92
    assert robot.connect_calls == 0
    assert robot.send_calls == 0
    assert robot.is_connected is False
    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.disconnect_args == [True]
    assert robot.left_arm.bus.disable_torque_calls == 1
    assert robot.right_arm.bus.disable_torque_calls == 1
    assert robot.left_arm.cameras["wrist"].disconnect_calls == 1
    assert robot.right_arm.cameras["wrist"].disconnect_calls == 1


def test_recorder_rejects_camera_skew_and_cleans_every_resource(tmp_path) -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock, camera_skew_s=0.03)

    with pytest.raises(PromptRecorderError, match="camera skew"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.is_connected is False
    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.disconnect_args == [True]
    assert not tuple(tmp_path.glob("prompt_000000.*"))


def test_recorder_rejects_capture_drift_and_leaves_no_artifact(tmp_path) -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock, capture_latency_s=0.04)

    with pytest.raises(PromptRecorderError, match="timestamp step"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.observation_calls == 2
    assert robot.is_connected is False
    assert not tuple(tmp_path.glob("prompt_000000.*"))


def test_recorder_rejects_irregular_camera_cadence_even_if_average_is_30_hz(
    tmp_path,
) -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(
        clock,
        camera_timestamp_jitter_s=(0.0, -0.010),
    )

    with pytest.raises(PromptRecorderError, match="camera timestamp step"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.observation_calls == 3
    assert robot.is_connected is False
    assert not tuple(tmp_path.glob("prompt_000000.*"))


def test_recorder_rejects_missing_atomic_camera_timestamp_and_cleans_up(
    tmp_path,
) -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock)
    left_camera = robot.left_arm.cameras["wrist"]
    left_camera.frame_lock = RefreshingLock(lambda: None)
    del left_camera.latest_timestamp

    with pytest.raises(ContractError, match="camera timestamp is invalid"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.observation_calls == 2
    assert robot.is_connected is False
    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.disconnect_args == [True]
    assert not tuple(tmp_path.glob("prompt_000000.*"))


def test_recorder_requires_torque_off_capability_and_cleans_up(tmp_path) -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock)
    robot.left_arm.bus.disable_torque = None

    with pytest.raises(HardwarePreflightError, match="torque-off"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.observation_calls == 0
    assert robot.is_connected is False
    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.connect_calls == 1
    assert robot.left_arm.bus.disable_torque_calls == 0
    assert robot.right_arm.bus.disable_torque_calls == 0
    assert not tuple(tmp_path.glob("prompt_000000.*"))


def test_recorder_rejects_missing_measured_home_before_hardware_construction(
    tmp_path,
) -> None:
    factory_calls = 0

    def factory(config):
        nonlocal factory_calls
        del config
        factory_calls += 1
        raise AssertionError("hardware must not be constructed")

    with pytest.raises(PromptRecorderError, match="measured home"):
        record_physical_prompt(
            _config(actuation=False, include_home=False),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=factory,
        )

    assert factory_calls == 0


def test_recorder_rejects_uncalibrated_safety_before_hardware_construction(
    tmp_path,
) -> None:
    factory_calls = 0

    def factory(config):
        nonlocal factory_calls
        del config
        factory_calls += 1
        raise AssertionError("hardware must not be constructed")

    with pytest.raises(PromptRecorderError, match="measured calibrated safety"):
        record_physical_prompt(
            _config(actuation=False, calibrated=False),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=factory,
        )

    assert factory_calls == 0


def test_recorder_checks_home_pose_before_disabling_torque(tmp_path) -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock)
    robot.right_arm.bus.joint_offset = 30.0

    with pytest.raises(HardwarePreflightError, match="home tolerance"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.left_arm.bus.connect_calls == 1
    assert robot.right_arm.bus.connect_calls == 1
    assert robot.left_arm.bus.disable_torque_calls == 0
    assert robot.right_arm.bus.disable_torque_calls == 0
    assert robot.left_arm.cameras["wrist"].connect_calls == 0
    assert robot.right_arm.cameras["wrist"].connect_calls == 0
    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.disconnect_args == [True]
    assert not tuple(tmp_path.glob("prompt_000000.*"))


def test_recorder_rejects_existing_artifact_before_hardware_construction(
    tmp_path,
) -> None:
    (tmp_path / "prompt_000000.npz").write_bytes(b"existing")
    factory_calls = 0

    def factory(config):
        nonlocal factory_calls
        del config
        factory_calls += 1
        raise AssertionError("hardware must not be constructed")

    with pytest.raises(DatasetArtifactExistsError, match="already exists"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.0,
            robot_factory=factory,
        )

    assert factory_calls == 0


def test_recorder_rejects_off_grid_duration_before_hardware_construction(
    tmp_path,
) -> None:
    factory_calls = 0

    def factory(config):
        nonlocal factory_calls
        del config
        factory_calls += 1
        raise AssertionError("hardware must not be constructed")

    with pytest.raises(PromptRecorderError, match="capture grid"):
        record_physical_prompt(
            _config(),
            output_dir=tmp_path,
            task="place block",
            duration_s=3.01,
            robot_factory=factory,
        )

    assert factory_calls == 0
