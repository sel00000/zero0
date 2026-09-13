from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from so101_wam.config import (
    LeRobotConfig,
    LeRobotLeaderConfig,
    ProjectConfig,
    RuntimeConfig,
    SafetyConfig,
)
from so101_wam.constants import ACTION_DIM, ARM_JOINT_NAMES, JOINT_KEYS
from so101_wam.dataset import load_episode
from so101_wam.demo_recorder import record_teleop_demo
from so101_wam.safety import SafetySupervisor


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Camera:
    def __init__(self, clock, value):
        self.clock = clock
        self.is_connected = False
        self.latest_frame = np.full((4, 4, 3), value, dtype=np.uint8)
        self.latest_timestamp = None
        self.thread = SimpleNamespace(is_alive=lambda: True)
        self.frame_lock = self
        self.age = 0.001
        self.frozen = False

    def __enter__(self):
        if not self.frozen or self.latest_timestamp is None:
            self.latest_timestamp = self.clock() - self.age

    def __exit__(self, *args):
        pass

    def disconnect(self):
        self.is_connected = False
        self.thread = None


class Arm:
    def __init__(self, clock, value):
        self.position = 0.0
        self.bus = SimpleNamespace(
            is_connected=False,
            sync_read=lambda *args, **kwargs: {
                key: self.position for key in ARM_JOINT_NAMES
            },
        )
        self.cameras = {"wrist": Camera(clock, value)}
        self.config = SimpleNamespace(num_read_retries=0)

    @property
    def is_connected(self):
        return self.bus.is_connected

    def disconnect(self):
        self.bus.is_connected = False
        self.cameras["wrist"].disconnect()


class Follower:
    def __init__(self, clock):
        self.clock = clock
        self.left_arm = Arm(clock, 10)
        self.right_arm = Arm(clock, 20)
        self.is_calibrated = True
        self.sent = []
        self.receipts = []
        self.failure = None
        self.send_latency = 0.0

    @property
    def is_connected(self):
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, *, calibrate):
        assert calibrate is False
        for arm in (self.left_arm, self.right_arm):
            arm.bus.is_connected = True
            arm.cameras["wrist"].is_connected = True

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def send_action(self, action):
        self.sent.append(action)
        self.clock.sleep(self.send_latency)
        if self.failure == "partial" and len(self.sent) == 2:
            raise RuntimeError("right arm write failed after left arm write")
        if self.failure == "interrupt":
            raise KeyboardInterrupt
        if self.failure == "no_receipt":
            return None
        if self.failure == "bad_receipt":
            return {"left_gripper.pos": 0.0}
        if self.failure == "unsafe_receipt":
            return {key: 1.0 for key in JOINT_KEYS}
        receipt = {key: float(np.clip(value, -0.25, 0.25)) for key, value in action.items()}
        self.receipts.append(receipt)
        return receipt


class Leader:
    def __init__(self, clock):
        self.clock = clock
        self.is_connected = False
        self.is_calibrated = True
        self.calls = 0
        self.initial = 0.0
        self.target = 0.8
        self.read_latency = 0.0

    def connect(self, *, calibrate):
        assert calibrate is False
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_action(self):
        self.clock.sleep(self.read_latency)
        value = self.initial if self.calls == 0 else self.target
        self.calls += 1
        return {key: value for key in JOINT_KEYS}


@pytest.fixture
def session(tmp_path):
    clock = Clock()
    follower, leader = Follower(clock), Leader(clock)
    config = ProjectConfig(
        runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
        safety=SafetyConfig(
            joint_lower=(-2.0,) * ACTION_DIM,
            joint_upper=(2.0,) * ACTION_DIM,
            max_delta_per_servo_tick=(0.5,) * ACTION_DIM,
            calibrated=True,
        ),
        lerobot=LeRobotConfig(
            left_port="/dev/follower-left", right_port="/dev/follower-right",
            left_wrist_camera=0, right_wrist_camera=1,
            camera_width=4, camera_height=4,
            calibration_dir="/fake/calibration", hardware_id="fake-follower",
            calibration_id="fake-follower-calibration",
            home_joint_position=(0.0,) * ACTION_DIM, home_joint_tolerance=(0.1,) * ACTION_DIM,
        ),
        leader=LeRobotLeaderConfig(
            left_port="/dev/leader-left", right_port="/dev/leader-right",
            calibration_dir="/fake/leader-calibration",
            calibration_id="fake-leader-calibration",
        ),
    )
    kwargs = dict(
        output_dir=tmp_path, task="stack cups", duration_s=0.1,
        follower_factory=lambda _: follower, leader_factory=lambda _: leader,
        clock=clock, camera_clock=clock, sleeper=clock.sleep,
    )
    return config, kwargs, follower, leader


def test_sent_labels(session):
    config, kwargs, follower, leader = session
    result = record_teleop_demo(config, **kwargs)
    data = load_episode(result["npz_path"])

    assert data.frame_count == len(follower.sent) == 4
    np.testing.assert_allclose(data.timestamps_s, np.arange(4) / 30)
    np.testing.assert_array_equal(data.joint_state, 0.0)
    np.testing.assert_array_equal(data.action[1:], 0.25)
    expected = [[receipt[key] for key in JOINT_KEYS] for receipt in follower.receipts]
    np.testing.assert_array_equal(data.action, expected)
    np.testing.assert_allclose(data.metadata["requested_action"][1:], 0.8)
    np.testing.assert_allclose(data.metadata["commanded_action"][1:], 0.5)
    assert data.metadata["action_source"] == "lerobot_send_action_return"
    assert data.metadata["task_success"] is None
    assert data.metadata["evidence_level"] == "injected_devices_unverified"
    times = data.metadata["timing"]
    assert times["send_started_s"][-1] - times["origin_s"] == pytest.approx(0.1)
    for start, end in zip(times["send_started_s"], times["send_finished_s"]):
        assert end >= start
    assert times["camera_timestamp_s"]["left_wrist"] == times["camera_timestamp_s"]["right_wrist"]
    assert result["round_trip_verified"] is True
    assert not follower.is_connected and not leader.is_connected


@pytest.mark.parametrize("failure", ["no_receipt", "bad_receipt", "unsafe_receipt", "partial"])
def test_missing_receipt(session, failure):
    config, kwargs, follower, leader = session
    follower.failure = failure
    with pytest.raises((ValueError, RuntimeError)):
        record_teleop_demo(config, **kwargs)
    assert not list(kwargs["output_dir"].glob("*.npz"))
    assert not list(kwargs["output_dir"].glob("*.json"))
    assert not follower.is_connected and not leader.is_connected


@pytest.mark.parametrize("fault", ["stale", "skew", "frozen"])
def test_camera_fault(session, fault):
    config, kwargs, follower, leader = session
    camera = follower.left_arm.cameras["wrist"]
    if fault == "frozen":
        camera.frozen = True
    else:
        camera.age = 1.0 if fault == "stale" else 0.03
    with pytest.raises((ValueError, RuntimeError)):
        record_teleop_demo(config, **kwargs)
    assert len(follower.sent) == (1 if fault == "frozen" else 0)
    assert not list(kwargs["output_dir"].glob("*.npz"))
    assert not follower.is_connected and not leader.is_connected


@pytest.mark.parametrize("device", ["follower", "leader"])
def test_initial_alignment(session, device):
    config, kwargs, follower, leader = session
    if device == "follower":
        follower.left_arm.position = 0.2
    else:
        leader.initial = 0.2
    with pytest.raises((ValueError, RuntimeError), match="home|aligned"):
        record_teleop_demo(config, **kwargs)
    assert follower.sent == []
    assert not follower.is_connected and not leader.is_connected


@pytest.mark.parametrize("device", ["follower", "leader"])
def test_device_latency(session, device):
    config, kwargs, follower, leader = session
    if device == "follower":
        follower.send_latency = 0.2
    else:
        leader.read_latency = 0.2
    with pytest.raises((ValueError, RuntimeError)):
        record_teleop_demo(config, **kwargs)
    assert not list(kwargs["output_dir"].glob("*.npz"))
    assert len(follower.sent) == (1 if device == "follower" else 0)
    assert not follower.is_connected and not leader.is_connected


def test_interrupt_cleanup(session):
    config, kwargs, follower, leader = session
    follower.failure = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        record_teleop_demo(config, **kwargs)
    assert not follower.is_connected and not leader.is_connected
    assert not list(kwargs["output_dir"].glob("*.npz"))


def test_clock_regression(session):
    config, kwargs, follower, leader = session
    leader.read_latency = -0.001
    with pytest.raises((ValueError, RuntimeError), match="clock"):
        record_teleop_demo(config, **kwargs)
    assert follower.sent == []
    assert not follower.is_connected and not leader.is_connected


def test_camera_expiry(session, monkeypatch):
    config, kwargs, follower, _ = session
    for arm in (follower.left_arm, follower.right_arm):
        arm.cameras["wrist"].age = 0.095
    evaluate = SafetySupervisor.evaluate

    def slow_check(self, *args, **options):
        result = evaluate(self, *args, **options)
        follower.clock.sleep(0.01)
        return result

    monkeypatch.setattr(SafetySupervisor, "evaluate", slow_check)
    with pytest.raises((ValueError, RuntimeError), match="stale|expired"):
        record_teleop_demo(config, **kwargs)
    assert follower.sent == []


def test_joint_limit(session):
    config, kwargs, follower, leader = session
    leader.target = 3.0
    with pytest.raises((ValueError, RuntimeError), match="safety"):
        record_teleop_demo(config, **kwargs)
    assert len(follower.sent) == 1
    assert not list(kwargs["output_dir"].glob("*.npz"))


@pytest.mark.parametrize("device", ["follower", "leader"])
def test_disconnect_failure(session, monkeypatch, device):
    config, kwargs, follower, leader = session
    selected = follower if device == "follower" else leader
    disconnect = selected.disconnect

    def failed_disconnect():
        disconnect()
        raise RuntimeError("disconnect failure")

    monkeypatch.setattr(selected, "disconnect", failed_disconnect)
    with pytest.raises(RuntimeError, match="disconnect"):
        record_teleop_demo(config, **kwargs)
    assert not follower.is_connected and not leader.is_connected
    assert not list(kwargs["output_dir"].glob("*.npz"))


def test_long_episode(session):
    config, kwargs, _, _ = session
    kwargs["duration_s"] = 13.0
    result = record_teleop_demo(config, **kwargs)
    data = load_episode(result["npz_path"])
    assert data.timestamps_s[-1] == pytest.approx(13.0)
    assert data.frame_count == 391


@pytest.mark.parametrize("fault", ["disabled", "shared_port", "duration", "overwrite"])
def test_request_preflight(session, fault):
    config, kwargs, _, _ = session
    if fault == "disabled":
        config = replace(config, runtime=replace(config.runtime, actuation_enabled=False))
    elif fault == "shared_port":
        config = replace(config, leader=replace(config.leader, left_port=config.lerobot.left_port))
    elif fault == "duration":
        kwargs["duration_s"] = float("nan")
    else:
        (kwargs["output_dir"] / "demo_000000.json").write_text("existing")

    def unexpected(_):
        pytest.fail("hardware constructed before validating request")

    kwargs.update(follower_factory=unexpected, leader_factory=unexpected)
    with pytest.raises((ValueError, RuntimeError, PermissionError)):
        record_teleop_demo(config, **kwargs)
