"""Dependency-free TOML configuration with fail-closed hardware defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Mapping
import tomllib

from .constants import ACTION_DIM, LEROBOT_VERSION_PIN, PRIMARY_CAMERA_KEYS


BUNDLED_CONFIG_DIR = Path(__file__).resolve().parent / "assets" / "configs"
DEFAULT_FAKE_CONFIG_PATH = BUNDLED_CONFIG_DIR / "fake.toml"
DEFAULT_MUJOCO_CONFIG_PATH = BUNDLED_CONFIG_DIR / "mujoco.toml"
DEFAULT_ROBOT_FREE_CONFIG_PATH = BUNDLED_CONFIG_DIR / "mujoco_robot_free.toml"


class ConfigError(ValueError):
    """Raised when configuration would violate an implementation invariant."""


def _float_tuple(value: Any, *, name: str, size: int = ACTION_DIM) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be a sequence of numbers") from error
    if len(result) != size:
        raise ConfigError(f"{name} must contain {size} values, got {len(result)}")
    if not all(isfinite(item) for item in result):
        raise ConfigError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    backend: str = "fake"
    camera_hz: float = 30.0
    policy_hz: float = 10.0
    servo_hz: float = 50.0
    context_seconds: float = 30.0
    prompt_min_seconds: float = 3.0
    prompt_max_seconds: float = 12.0
    action_horizon: int = 10
    primary_cameras: tuple[str, str] = PRIMARY_CAMERA_KEYS
    use_head_camera: bool = False
    actuation_enabled: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"fake", "mujoco", "lerobot"}:
            raise ConfigError("runtime.backend must be 'fake', 'mujoco', or 'lerobot'")
        if min(self.camera_hz, self.policy_hz, self.servo_hz) <= 0:
            raise ConfigError("camera_hz, policy_hz, and servo_hz must be positive")
        if self.policy_hz > self.camera_hz:
            raise ConfigError("policy_hz cannot exceed camera_hz")
        if self.servo_hz < self.policy_hz:
            raise ConfigError("servo_hz cannot be lower than policy_hz")
        if self.context_seconds < self.prompt_max_seconds:
            raise ConfigError("context_seconds must fit the longest physical prompt")
        if not 0 < self.prompt_min_seconds <= self.prompt_max_seconds:
            raise ConfigError("invalid physical prompt duration bounds")
        if self.action_horizon < 1:
            raise ConfigError("action_horizon must be positive")
        if tuple(self.primary_cameras) != PRIMARY_CAMERA_KEYS:
            raise ConfigError(f"core benchmark cameras must be exactly {PRIMARY_CAMERA_KEYS}")


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    max_delta_per_servo_tick: tuple[float, ...]
    max_observation_age_s: float = 0.10
    max_camera_skew_s: float = 0.017
    watchdog_timeout_s: float = 0.10
    calibrated: bool = False
    observation_joint_lower: tuple[float, ...] | None = None
    observation_joint_upper: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.observation_joint_lower is not None:
            object.__setattr__(
                self,
                "observation_joint_lower",
                _float_tuple(
                    self.observation_joint_lower,
                    name="safety.observation_joint_lower",
                ),
            )
        if self.observation_joint_upper is not None:
            object.__setattr__(
                self,
                "observation_joint_upper",
                _float_tuple(
                    self.observation_joint_upper,
                    name="safety.observation_joint_upper",
                ),
            )
        if not all(low < high for low, high in zip(self.joint_lower, self.joint_upper)):
            raise ConfigError("each joint lower bound must be below its upper bound")
        observation_lower = self.observation_joint_lower
        observation_upper = self.observation_joint_upper
        if (observation_lower is None) != (observation_upper is None):
            raise ConfigError("observation joint bounds must be supplied together")
        if observation_lower is not None and observation_upper is not None:
            if not all(
                low < high
                for low, high in zip(observation_lower, observation_upper, strict=True)
            ):
                raise ConfigError(
                    "each observation joint lower bound must be below its upper bound"
                )
        if not all(delta > 0 for delta in self.max_delta_per_servo_tick):
            raise ConfigError("max joint deltas must be positive")
        if min(self.max_observation_age_s, self.max_camera_skew_s, self.watchdog_timeout_s) <= 0:
            raise ConfigError("safety timeouts must be positive")

    @classmethod
    def fake_normalized(cls) -> "SafetyConfig":
        """Safe fake-device ranges; never a substitute for measured calibration."""

        return cls(
            joint_lower=(-100.0,) * 10 + (0.0, 0.0),
            joint_upper=(100.0,) * 12,
            max_delta_per_servo_tick=(2.0,) * 10 + (4.0, 4.0),
        )


@dataclass(frozen=True, slots=True)
class LeRobotConfig:
    version: str = LEROBOT_VERSION_PIN
    robot_type: str = "bi_so_follower"
    robot_id: str = "so101_wam"
    left_port: str = ""
    right_port: str = ""
    left_wrist_camera: str | int = ""
    right_wrist_camera: str | int = ""
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    calibration_dir: str = ""
    hardware_id: str = ""
    calibration_id: str = ""
    home_joint_position: tuple[float, ...] | None = None
    home_joint_tolerance: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "robot_type",
            "robot_id",
            "left_port",
            "right_port",
            "calibration_dir",
            "hardware_id",
            "calibration_id",
        ):
            if not isinstance(getattr(self, name), str):
                raise ConfigError(f"lerobot.{name} must be a string")
        if self.version != LEROBOT_VERSION_PIN:
            raise ConfigError(f"this adapter is verified only for lerobot=={LEROBOT_VERSION_PIN}")
        if self.robot_type != "bi_so_follower":
            raise ConfigError("only the LeRobot bi_so_follower contract is supported")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (
            self.camera_width,
            self.camera_height,
            self.camera_fps,
        )):
            raise ConfigError("LeRobot camera width, height, and fps must be integers")
        if min(self.camera_width, self.camera_height, self.camera_fps) <= 0:
            raise ConfigError("LeRobot camera width, height, and fps must be positive")
        for name, source in (
            ("left_wrist_camera", self.left_wrist_camera),
            ("right_wrist_camera", self.right_wrist_camera),
        ):
            if not isinstance(source, (str, int)) or isinstance(source, bool):
                raise ConfigError(f"lerobot.{name} must be a device path or non-negative integer index")
            if isinstance(source, int) and source < 0:
                raise ConfigError(f"lerobot.{name} integer index must be non-negative")
        if self.home_joint_position is not None:
            object.__setattr__(
                self,
                "home_joint_position",
                _float_tuple(self.home_joint_position, name="lerobot.home_joint_position"),
            )
        if self.home_joint_tolerance is not None:
            tolerance = _float_tuple(
                self.home_joint_tolerance,
                name="lerobot.home_joint_tolerance",
            )
            if not all(value > 0 for value in tolerance):
                raise ConfigError("lerobot.home_joint_tolerance values must be positive")
            object.__setattr__(self, "home_joint_tolerance", tolerance)

    def require_hardware_session(self) -> None:
        missing = tuple(
            name
            for name, value in (
                ("left_port", self.left_port),
                ("right_port", self.right_port),
                ("calibration_dir", self.calibration_dir),
            )
            if not value.strip()
        )
        if missing:
            raise ConfigError(f"LeRobot hardware session requires fields: {missing}")
        missing_cameras = tuple(
            name
            for name, value in (
                ("left_wrist_camera", self.left_wrist_camera),
                ("right_wrist_camera", self.right_wrist_camera),
            )
            if isinstance(value, str) and not value.strip()
        )
        if missing_cameras:
            raise ConfigError(f"LeRobot hardware session requires wrist cameras: {missing_cameras}")
        if self.left_port == self.right_port:
            raise ConfigError("left_port and right_port must be distinct")
        if self.left_wrist_camera == self.right_wrist_camera:
            raise ConfigError("left_wrist_camera and right_wrist_camera must be distinct")

    def require_actuation_identity(self) -> None:
        self.require_hardware_session()
        missing = tuple(
            name
            for name, value in (
                ("hardware_id", self.hardware_id),
                ("calibration_id", self.calibration_id),
            )
            if not value.strip()
        )
        if missing:
            raise ConfigError(f"real actuation requires LeRobot identity fields: {missing}")
        if self.home_joint_position is None:
            raise ConfigError("real actuation requires a measured arms-down home_joint_position")
        if self.home_joint_tolerance is None:
            raise ConfigError("real actuation requires a measured home_joint_tolerance")


@dataclass(frozen=True, slots=True)
class LeRobotLeaderConfig:
    robot_id: str = "zero01_leader"
    left_port: str = ""
    right_port: str = ""
    calibration_dir: str = ""
    calibration_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "robot_id",
            "left_port",
            "right_port",
            "calibration_dir",
            "calibration_id",
        ):
            if not isinstance(getattr(self, name), str):
                raise ConfigError(f"leader.{name} must be a string")

    def require_hardware_session(self) -> None:
        missing = tuple(
            name
            for name, value in (
                ("robot_id", self.robot_id),
                ("left_port", self.left_port),
                ("right_port", self.right_port),
                ("calibration_dir", self.calibration_dir),
                ("calibration_id", self.calibration_id),
            )
            if not value.strip()
        )
        if missing:
            raise ConfigError(f"leader hardware session requires fields: {missing}")
        if self.left_port == self.right_port:
            raise ConfigError("leader left_port and right_port must be distinct")


@dataclass(frozen=True, slots=True)
class MujocoConfig:
    """MuJoCo scene and renderer settings in the canonical LeRobot units."""

    model_path: str = ""
    camera_width: int = 320
    camera_height: int = 240
    gl_backend: str = "egl"
    home_joint_position: tuple[float, ...] = (0.0,) * ACTION_DIM
    forbid_collisions: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str):
            raise ConfigError("mujoco.model_path must be a string")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.camera_width, self.camera_height)
        ):
            raise ConfigError("MuJoCo camera width and height must be integers")
        if min(self.camera_width, self.camera_height) < 2:
            raise ConfigError("MuJoCo camera width and height must be at least 2")
        if self.gl_backend not in {"egl", "glfw", "osmesa"}:
            raise ConfigError("mujoco.gl_backend must be 'egl', 'glfw', or 'osmesa'")
        if not isinstance(self.forbid_collisions, bool):
            raise ConfigError("mujoco.forbid_collisions must be a boolean")
        object.__setattr__(
            self,
            "home_joint_position",
            _float_tuple(self.home_joint_position, name="mujoco.home_joint_position"),
        )


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig.fake_normalized)
    lerobot: LeRobotConfig = field(default_factory=LeRobotConfig)
    mujoco: MujocoConfig = field(default_factory=MujocoConfig)
    leader: LeRobotLeaderConfig = field(default_factory=LeRobotLeaderConfig)

    def __post_init__(self) -> None:
        if self.runtime.backend == "lerobot" and self.lerobot.camera_fps != self.runtime.camera_hz:
            raise ConfigError("lerobot.camera_fps must match runtime.camera_hz")
        if self.runtime.backend == "lerobot" and self.runtime.actuation_enabled:
            if not self.safety.calibrated:
                raise ConfigError("real LeRobot actuation requires a measured calibrated safety profile")
            self.lerobot.require_actuation_identity()
            home = self.lerobot.home_joint_position
            tolerance = self.lerobot.home_joint_tolerance
            if home is None or tolerance is None:
                raise ConfigError("real actuation requires a measured arms-down home_joint_position")
            outside = tuple(
                index
                for index, (low, value, high) in enumerate(
                    zip(self.safety.joint_lower, home, self.safety.joint_upper, strict=True)
                )
                if value < low or value > high
            )
            if outside:
                raise ConfigError(f"lerobot.home_joint_position is outside safety limits at indices {outside}")
            unsafe_band = tuple(
                index
                for index, (low, value, margin, high) in enumerate(
                    zip(
                        self.safety.joint_lower,
                        home,
                        tolerance,
                        self.safety.joint_upper,
                        strict=True,
                    )
                )
                if value - margin < low or value + margin > high
            )
            if unsafe_band:
                raise ConfigError(
                    "lerobot home tolerance band extends outside safety limits at indices "
                    f"{unsafe_band}"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProjectConfig":
        runtime_data = dict(value.get("runtime", {}))
        if "primary_cameras" in runtime_data:
            runtime_data["primary_cameras"] = tuple(runtime_data["primary_cameras"])
        runtime = RuntimeConfig(**runtime_data)

        safety_data = value.get("safety")
        if safety_data is None:
            safety = SafetyConfig.fake_normalized()
        else:
            safety_dict = dict(safety_data)
            safety = SafetyConfig(
                joint_lower=_float_tuple(safety_dict.pop("joint_lower"), name="safety.joint_lower"),
                joint_upper=_float_tuple(safety_dict.pop("joint_upper"), name="safety.joint_upper"),
                max_delta_per_servo_tick=_float_tuple(
                    safety_dict.pop("max_delta_per_servo_tick"),
                    name="safety.max_delta_per_servo_tick",
                ),
                observation_joint_lower=safety_dict.pop("observation_joint_lower", None),
                observation_joint_upper=safety_dict.pop("observation_joint_upper", None),
                **safety_dict,
            )

        lerobot = LeRobotConfig(**dict(value.get("lerobot", {})))
        leader = LeRobotLeaderConfig(**dict(value.get("leader", {})))
        mujoco_data = dict(value.get("mujoco", {}))
        if "home_joint_position" in mujoco_data:
            mujoco_data["home_joint_position"] = tuple(mujoco_data["home_joint_position"])
        mujoco = MujocoConfig(**mujoco_data)
        return cls(
            runtime=runtime,
            safety=safety,
            lerobot=lerobot,
            leader=leader,
            mujoco=mujoco,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("rb") as config_file:
            return cls.from_mapping(tomllib.load(config_file))
