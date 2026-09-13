from __future__ import annotations

from pathlib import Path

import pytest

from so101_wam.config import (
    DEFAULT_FAKE_CONFIG_PATH,
    DEFAULT_MUJOCO_CONFIG_PATH,
    DEFAULT_ROBOT_FREE_CONFIG_PATH,
    ConfigError,
    LeRobotConfig,
    ProjectConfig,
    RuntimeConfig,
    SafetyConfig,
)
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("bundled_path", "repository_name"),
    [
        (DEFAULT_FAKE_CONFIG_PATH, "fake.toml"),
        (DEFAULT_MUJOCO_CONFIG_PATH, "mujoco.toml"),
        (DEFAULT_ROBOT_FREE_CONFIG_PATH, "mujoco_robot_free.toml"),
    ],
)
def test_bundled_default_config_matches_repository_template(
    bundled_path: Path,
    repository_name: str,
) -> None:
    repository_path = ROOT / "configs" / repository_name

    assert bundled_path.is_absolute()
    assert bundled_path.is_file()
    assert bundled_path.read_bytes() == repository_path.read_bytes()
    ProjectConfig.load(bundled_path)


def test_fake_config_is_fail_closed_and_wrist_only() -> None:
    config = ProjectConfig.load(ROOT / "configs" / "fake.toml")

    assert config.runtime.primary_cameras == PRIMARY_CAMERA_KEYS
    assert config.runtime.backend == "fake"
    assert config.runtime.use_head_camera is False
    assert config.runtime.actuation_enabled is False
    assert config.safety.calibrated is False
    assert len(config.safety.joint_lower) == ACTION_DIM
    assert config.lerobot.version == "0.6.1"


def test_hardware_template_loads_but_cannot_actuate() -> None:
    config = ProjectConfig.load(ROOT / "configs" / "lerobot_hardware.template.toml")

    assert config.runtime.backend == "lerobot"
    assert config.runtime.actuation_enabled is False
    assert config.safety.calibrated is False
    assert config.lerobot.left_port == config.lerobot.right_port == ""
    assert config.lerobot.left_wrist_camera == config.lerobot.right_wrist_camera == ""


def test_core_benchmark_rejects_a_head_camera_as_required_input() -> None:
    with pytest.raises(ConfigError, match="exactly"):
        RuntimeConfig(primary_cameras=("left_wrist", "head_optional"))


def test_real_lerobot_actuation_requires_measured_safety_and_identity() -> None:
    with pytest.raises(ConfigError, match="measured calibrated"):
        ProjectConfig(runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True))

    measured = SafetyConfig(
        joint_lower=(-1.0,) * ACTION_DIM,
        joint_upper=(1.0,) * ACTION_DIM,
        max_delta_per_servo_tick=(0.1,) * ACTION_DIM,
        calibrated=True,
    )
    with pytest.raises(ConfigError, match="hardware session"):
        ProjectConfig(
            runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
            safety=measured,
        )

    with pytest.raises(ConfigError, match="home_joint_position"):
        ProjectConfig(
            runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
            safety=measured,
            lerobot=LeRobotConfig(
                left_port="/dev/ttyACM0",
                right_port="/dev/ttyACM1",
                left_wrist_camera="/dev/video0",
                right_wrist_camera="/dev/video2",
                calibration_dir="/tmp/calibration",
                hardware_id="bench-a",
                calibration_id="cal-2026-08-31",
            ),
        )

    with pytest.raises(ConfigError, match="home_joint_tolerance"):
        ProjectConfig(
            runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
            safety=measured,
            lerobot=LeRobotConfig(
                left_port="/dev/ttyACM0",
                right_port="/dev/ttyACM1",
                left_wrist_camera="/dev/video0",
                right_wrist_camera="/dev/video2",
                calibration_dir="/tmp/calibration",
                hardware_id="bench-a",
                calibration_id="cal-2026-08-31",
                home_joint_position=(0.0,) * ACTION_DIM,
            ),
        )

    ready = ProjectConfig(
        runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
        safety=measured,
        lerobot=LeRobotConfig(
            left_port="/dev/ttyACM0",
            right_port="/dev/ttyACM1",
            left_wrist_camera="/dev/video0",
            right_wrist_camera="/dev/video2",
            calibration_dir="/tmp/calibration",
            hardware_id="bench-a",
            calibration_id="cal-2026-08-31",
            home_joint_position=(0.0,) * ACTION_DIM,
            home_joint_tolerance=(0.1,) * ACTION_DIM,
        ),
    )
    assert ready.runtime.actuation_enabled is True


def test_real_lerobot_actuation_rejects_invalid_home_gate() -> None:
    with pytest.raises(ConfigError, match="positive"):
        LeRobotConfig(home_joint_tolerance=(0.0,) * ACTION_DIM)

    measured = SafetyConfig(
        joint_lower=(-1.0,) * ACTION_DIM,
        joint_upper=(1.0,) * ACTION_DIM,
        max_delta_per_servo_tick=(0.1,) * ACTION_DIM,
        calibrated=True,
    )
    with pytest.raises(ConfigError, match="outside safety limits"):
        ProjectConfig(
            runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
            safety=measured,
            lerobot=LeRobotConfig(
                left_port="left",
                right_port="right",
                left_wrist_camera=0,
                right_wrist_camera=1,
                calibration_dir="calibration",
                hardware_id="bench-a",
                calibration_id="cal-a",
                home_joint_position=(2.0,) + (0.0,) * (ACTION_DIM - 1),
                home_joint_tolerance=(0.1,) * ACTION_DIM,
            ),
        )

    with pytest.raises(ConfigError, match="tolerance band"):
        ProjectConfig(
            runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
            safety=measured,
            lerobot=LeRobotConfig(
                left_port="left",
                right_port="right",
                left_wrist_camera=0,
                right_wrist_camera=1,
                calibration_dir="calibration",
                hardware_id="bench-a",
                calibration_id="cal-a",
                home_joint_position=(0.95,) + (0.0,) * (ACTION_DIM - 1),
                home_joint_tolerance=(0.1,) * ACTION_DIM,
            ),
        )


def test_lerobot_hardware_session_requires_distinct_ports_and_cameras() -> None:
    with pytest.raises(ConfigError, match="wrist cameras"):
        LeRobotConfig(left_port="a", right_port="b", calibration_dir="cal").require_hardware_session()

    with pytest.raises(ConfigError, match="must be distinct"):
        LeRobotConfig(
            left_port="a",
            right_port="b",
            left_wrist_camera=0,
            right_wrist_camera=0,
            calibration_dir="cal",
        ).require_hardware_session()

    with pytest.raises(ConfigError, match="must match"):
        ProjectConfig(
            runtime=RuntimeConfig(backend="lerobot", camera_hz=30.0),
            lerobot=LeRobotConfig(camera_fps=25),
        )


def test_positional_config() -> None:
    mujoco = ProjectConfig().mujoco
    config = ProjectConfig(RuntimeConfig(), SafetyConfig.fake_normalized(), LeRobotConfig(), mujoco)

    assert config.mujoco is mujoco
    assert config.leader.robot_id == "zero01_leader"


def test_leader_defaults() -> None:
    config = ProjectConfig.from_mapping({})

    assert config.leader.robot_id == "zero01_leader"
    assert config.leader.left_port == ""
    assert config.leader.right_port == ""
    assert config.leader.calibration_dir == ""
    assert config.leader.calibration_id == ""


def test_leader_requires_ports() -> None:
    with pytest.raises(ConfigError, match="leader hardware session"):
        ProjectConfig.from_mapping(
            {"leader": {"left_port": "left"}}
        ).leader.require_hardware_session()

    with pytest.raises(ConfigError, match="distinct"):
        ProjectConfig.from_mapping(
            {
                "leader": {
                    "left_port": "same",
                    "right_port": "same",
                    "calibration_dir": "calibration",
                    "calibration_id": "cal-a",
                }
            }
        ).leader.require_hardware_session()

    ProjectConfig.from_mapping(
        {
            "leader": {
                "robot_id": "leader-a",
                "left_port": "left",
                "right_port": "right",
                "calibration_dir": "calibration",
                "calibration_id": "cal-a",
            }
        }
    ).leader.require_hardware_session()
