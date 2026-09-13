from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from so101_wam.config import ConfigError, LeRobotConfig, LeRobotLeaderConfig
from so101_wam import lerobot_factory
from so101_wam.lerobot_factory import LeRobotFactoryError, create_bi_so_follower


class _Record:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _Robot:
    def __init__(self, config: _Record) -> None:
        self.config = config


def _config() -> LeRobotConfig:
    return LeRobotConfig(
        left_port="/dev/ttyACM0",
        right_port="/dev/ttyACM1",
        left_wrist_camera="/dev/video0",
        right_wrist_camera=2,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        calibration_dir="/var/lib/so101/calibration",
    )


def test_factory_builds_per_arm_wrist_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    api = lerobot_factory._LeRobotAPI(
        color_mode_rgb="rgb",
        backend_v4l2="v4l2",
        no_rotation="none",
        camera_config=_Record,
        arm_config=_Record,
        bimanual_config=_Record,
        bimanual_robot=_Robot,
    )
    monkeypatch.setattr(lerobot_factory, "_load_lerobot_api", lambda: api)

    robot = create_bi_so_follower(_config())

    left = robot.config.kwargs["left_arm_config"]
    right = robot.config.kwargs["right_arm_config"]
    left_camera = left.kwargs["cameras"]["wrist"]
    right_camera = right.kwargs["cameras"]["wrist"]
    assert left.kwargs["port"] == "/dev/ttyACM0"
    assert right.kwargs["port"] == "/dev/ttyACM1"
    assert left.kwargs["use_degrees"] is True
    assert right.kwargs["use_degrees"] is True
    assert left_camera.kwargs["index_or_path"] == Path("/dev/video0")
    assert right_camera.kwargs["index_or_path"] == 2
    assert left_camera.kwargs["color_mode"] == "rgb"
    assert left_camera.kwargs["backend"] == "v4l2"
    assert robot.config.kwargs["calibration_dir"] == Path("/var/lib/so101/calibration")


def test_factory_rejects_incomplete_hardware_config_before_import() -> None:
    with pytest.raises(ConfigError, match="hardware session"):
        create_bi_so_follower(LeRobotConfig())


def test_factory_reports_missing_optional_lerobot_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> lerobot_factory._LeRobotAPI:
        raise LeRobotFactoryError("missing")

    monkeypatch.setattr(lerobot_factory, "_load_lerobot_api", unavailable)
    with pytest.raises(LeRobotFactoryError, match="missing"):
        create_bi_so_follower(_config())


def test_loads_leader_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    def module(name: str, **attrs: object) -> ModuleType:
        result = ModuleType(name)
        for key, value in attrs.items():
            setattr(result, key, value)
        return result

    modules = {
        "lerobot": module("lerobot"),
        "lerobot.cameras": module(
            "lerobot.cameras",
            ColorMode=SimpleNamespace(RGB="rgb"),
            Cv2Backends=SimpleNamespace(V4L2="v4l2"),
            Cv2Rotation=SimpleNamespace(NO_ROTATION="none"),
        ),
        "lerobot.cameras.opencv": module(
            "lerobot.cameras.opencv",
            OpenCVCameraConfig=_Record,
        ),
        "lerobot.robots": module("lerobot.robots"),
        "lerobot.robots.bi_so_follower": module(
            "lerobot.robots.bi_so_follower",
            BiSOFollower=_Robot,
            BiSOFollowerConfig=_Record,
        ),
        "lerobot.robots.so_follower": module(
            "lerobot.robots.so_follower",
            SOFollowerConfig=_Record,
        ),
        "lerobot.teleoperators": module("lerobot.teleoperators"),
        "lerobot.teleoperators.bi_so_leader": module(
            "lerobot.teleoperators.bi_so_leader",
            BiSOLeader=_Robot,
            BiSOLeaderConfig=_Record,
        ),
        "lerobot.teleoperators.so_leader": module(
            "lerobot.teleoperators.so_leader",
            SOLeaderConfig=_Record,
        ),
    }
    for name, loaded in modules.items():
        monkeypatch.setitem(__import__("sys").modules, name, loaded)
    monkeypatch.setattr(lerobot_factory, "version", lambda name: "0.6.1")

    api = lerobot_factory._load_lerobot_api()

    assert api.leader_arm_config is _Record
    assert api.leader_config is _Record
    assert api.bimanual_leader is _Robot


def test_factory_builds_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    api = lerobot_factory._LeRobotAPI(
        color_mode_rgb="rgb",
        backend_v4l2="v4l2",
        no_rotation="none",
        camera_config=_Record,
        arm_config=_Record,
        leader_arm_config=_Record,
        bimanual_config=_Record,
        leader_config=_Record,
        bimanual_robot=_Robot,
        bimanual_leader=_Robot,
    )
    monkeypatch.setattr(lerobot_factory, "_load_lerobot_api", lambda: api)

    robot = lerobot_factory.create_bi_so_leader(
        LeRobotLeaderConfig(
            robot_id="leader-a",
            left_port="/dev/ttyUSB0",
            right_port="/dev/ttyUSB1",
            calibration_dir="/var/lib/leader/calibration",
            calibration_id="cal-leader-a",
        )
    )

    left = robot.config.kwargs["left_arm_config"]
    right = robot.config.kwargs["right_arm_config"]
    assert left.kwargs["port"] == "/dev/ttyUSB0"
    assert right.kwargs["port"] == "/dev/ttyUSB1"
    assert left.kwargs["use_degrees"] is True
    assert right.kwargs["use_degrees"] is True
    assert robot.config.kwargs["id"] == "leader-a"
    assert robot.config.kwargs["calibration_dir"] == Path("/var/lib/leader/calibration")
