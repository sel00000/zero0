"""Version-pinned lazy construction of a LeRobot v0.6.1 BiSOFollower."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from .config import LeRobotConfig, LeRobotLeaderConfig
from .constants import LEROBOT_VERSION_PIN


class LeRobotFactoryError(RuntimeError):
    """Raised when the optional LeRobot hardware stack cannot be constructed."""


@dataclass(frozen=True, slots=True)
class _LeRobotAPI:
    color_mode_rgb: Any
    backend_v4l2: Any
    no_rotation: Any
    camera_config: Callable[..., Any]
    arm_config: Callable[..., Any]
    bimanual_config: Callable[..., Any]
    bimanual_robot: Callable[..., Any]
    leader_arm_config: Callable[..., Any] | None = None
    leader_config: Callable[..., Any] | None = None
    bimanual_leader: Callable[..., Any] | None = None


def _load_lerobot_api() -> _LeRobotAPI:
    try:
        installed_version = version("lerobot")
    except PackageNotFoundError as error:
        raise LeRobotFactoryError(
            f"LeRobot {LEROBOT_VERSION_PIN} is not installed in the robot environment"
        ) from error
    if installed_version != LEROBOT_VERSION_PIN:
        raise LeRobotFactoryError(
            f"LeRobot version must be {LEROBOT_VERSION_PIN}, got {installed_version}"
        )
    try:
        from lerobot.cameras import ColorMode, Cv2Backends, Cv2Rotation  # type: ignore[import-untyped]
        from lerobot.cameras.opencv import OpenCVCameraConfig  # type: ignore[import-untyped]
        from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig  # type: ignore[import-untyped]
        from lerobot.robots.so_follower import SOFollowerConfig  # type: ignore[import-untyped]
        from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig  # type: ignore[import-untyped]
        from lerobot.teleoperators.so_leader import SOLeaderConfig  # type: ignore[import-untyped]
    except (ImportError, ModuleNotFoundError) as error:
        raise LeRobotFactoryError(
            f"LeRobot {LEROBOT_VERSION_PIN} hardware modules are unavailable; "
            "install the official v0.6.1 hardware/feetech extras in the robot environment"
        ) from error

    return _LeRobotAPI(
        color_mode_rgb=ColorMode.RGB,
        backend_v4l2=Cv2Backends.V4L2,
        no_rotation=Cv2Rotation.NO_ROTATION,
        camera_config=OpenCVCameraConfig,
        arm_config=SOFollowerConfig,
        leader_arm_config=SOLeaderConfig,
        bimanual_config=BiSOFollowerConfig,
        leader_config=BiSOLeaderConfig,
        bimanual_robot=BiSOFollower,
        bimanual_leader=BiSOLeader,
    )


def _camera_source(value: str | int) -> Path | int:
    return Path(value) if isinstance(value, str) else value


def create_bi_so_follower(config: LeRobotConfig) -> Any:
    """Create, but do not connect, the configured dual SO-101 follower.

    Wrist cameras deliberately live inside each arm config under the local key
    ``wrist``. LeRobot's bimanual wrapper then emits the canonical
    ``left_wrist`` and ``right_wrist`` observation keys.
    """

    config.require_hardware_session()
    api = _load_lerobot_api()

    def camera(source: str | int) -> Any:
        return api.camera_config(
            index_or_path=_camera_source(source),
            fps=config.camera_fps,
            width=config.camera_width,
            height=config.camera_height,
            color_mode=api.color_mode_rgb,
            rotation=api.no_rotation,
            fourcc="MJPG",
            backend=api.backend_v4l2,
        )

    left_arm = api.arm_config(
        port=config.left_port,
        cameras={"wrist": camera(config.left_wrist_camera)},
        use_degrees=True,
    )
    right_arm = api.arm_config(
        port=config.right_port,
        cameras={"wrist": camera(config.right_wrist_camera)},
        use_degrees=True,
    )
    robot_config = api.bimanual_config(
        id=config.robot_id,
        calibration_dir=Path(config.calibration_dir),
        left_arm_config=left_arm,
        right_arm_config=right_arm,
    )
    return api.bimanual_robot(robot_config)


def create_bi_so_leader(config: LeRobotLeaderConfig) -> Any:
    """Create, but do not connect, the configured dual SO-101 leader."""

    config.require_hardware_session()
    api = _load_lerobot_api()
    if (
        api.leader_arm_config is None
        or api.leader_config is None
        or api.bimanual_leader is None
    ):
        raise LeRobotFactoryError("LeRobot leader API is unavailable")

    left_arm = api.leader_arm_config(port=config.left_port, use_degrees=True)
    right_arm = api.leader_arm_config(port=config.right_port, use_degrees=True)
    leader_config = api.leader_config(
        id=config.robot_id,
        calibration_dir=Path(config.calibration_dir),
        left_arm_config=left_arm,
        right_arm_config=right_arm,
    )
    return api.bimanual_leader(leader_config)


__all__ = [
    "LeRobotFactoryError",
    "create_bi_so_follower",
    "create_bi_so_leader",
]
