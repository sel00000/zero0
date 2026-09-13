"""Dependency-free duck-typed adapter for LeRobot 0.6.1 BiSOFollower."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np

from so101_wam.config import LeRobotConfig
from so101_wam.constants import (
    ACTION_DIM,
    ARM_JOINT_NAMES,
    ARM_SIDES,
    JOINT_KEYS,
    PRIMARY_CAMERA_KEYS,
)
from so101_wam.contracts import ActionChunk, ContractError, SensorimotorFrame


class ActuationDisabledError(PermissionError):
    """Raised when a caller attempts real actuation without enabling it."""


class AtomicObservationUnavailable(ContractError):
    """Raised when a robot lacks the pinned atomic observation surface."""


def _as_joint_vector(values: Mapping[str, Any], *, name: str) -> np.ndarray:
    missing = tuple(key for key in JOINT_KEYS if key not in values)
    if missing:
        raise ContractError(f"{name} missing joint key(s): {missing}")
    vector = np.array([values[key] for key in JOINT_KEYS], dtype=np.float32)
    if vector.shape != (ACTION_DIM,):
        raise ContractError(f"{name} must contain {ACTION_DIM} joint values")
    if not np.isfinite(vector).all():
        raise ContractError(f"{name} contains NaN or infinity")
    return vector


def observation_from_lerobot(
    observation: Mapping[str, Any],
    *,
    timestamp_s: float,
    image_timestamps_s: Mapping[str, float] | None = None,
) -> SensorimotorFrame:
    expected = set(JOINT_KEYS) | set(PRIMARY_CAMERA_KEYS)
    actual = set(observation)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected))
        raise ContractError(
            f"LeRobot observation keys must be exact; missing={missing}, extra={extra}"
        )

    images = {key: observation[key] for key in PRIMARY_CAMERA_KEYS}
    joints = _as_joint_vector(observation, name="LeRobot observation")
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images=images,
        joint_position=joints,
        image_timestamps_s=image_timestamps_s,
    )


def _camera_timestamps(
    candidates: Mapping[str, Any],
    *,
    require_timestamps: bool,
    camera_clock: Callable[[], float],
    max_camera_age_s: float | None,
) -> dict[str, float] | None:
    if set(candidates) != set(PRIMARY_CAMERA_KEYS):
        if require_timestamps:
            raise ContractError(
                "atomic LeRobot observation requires both wrist camera timestamps"
            )
        return None
    result: dict[str, float] = {}
    for key in PRIMARY_CAMERA_KEYS:
        value = candidates[key]
        if isinstance(value, bool):
            if require_timestamps:
                raise ContractError(f"{key} camera timestamp is invalid")
            return None
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            if require_timestamps:
                raise ContractError(f"{key} camera timestamp is invalid") from None
            return None
        if not np.isfinite(timestamp) or timestamp <= 0:
            if require_timestamps:
                raise ContractError(f"{key} camera timestamp is invalid")
            return None
        result[key] = timestamp

    if max_camera_age_s is not None:
        if not isfinite(max_camera_age_s) or max_camera_age_s <= 0:
            raise ContractError("max_camera_age_s must be finite and positive")
        now_s = float(camera_clock())
        if not isfinite(now_s) or now_s < 0:
            raise ContractError("camera clock returned an invalid timestamp")
        for key, timestamp in result.items():
            age_s = now_s - timestamp
            if age_s < -1e-6:
                raise ContractError(f"{key} camera timestamp is in the future")
            if age_s > max_camera_age_s:
                raise ContractError(
                    f"{key} camera frame is stale: {age_s:.6f}s > "
                    f"{max_camera_age_s:.6f}s"
                )
    return result


def _atomic_camera_snapshot(camera: Any, *, key: str) -> tuple[np.ndarray, Any]:
    frame_lock = getattr(camera, "frame_lock", None)
    if frame_lock is None or not hasattr(frame_lock, "__enter__"):
        raise AtomicObservationUnavailable(
            f"{key} camera lacks LeRobot v0.6.1 frame_lock"
        )
    thread = getattr(camera, "thread", None)
    is_alive = getattr(thread, "is_alive", None)
    if thread is None or not callable(is_alive):
        raise AtomicObservationUnavailable(
            f"{key} camera lacks LeRobot v0.6.1 read-thread state"
        )
    if not bool(is_alive()):
        raise ContractError(f"{key} camera read thread is not running")

    with frame_lock:
        frame = getattr(camera, "latest_frame", None)
        timestamp = getattr(camera, "latest_timestamp", None)
        if frame is None:
            raise ContractError(f"{key} camera has not captured a frame")
        # Copy while holding the same lock used by OpenCVCamera._read_loop so the
        # returned pixels and latest_timestamp identify one exact buffer entry.
        image = np.array(frame, copy=True)
    return image, timestamp


def _pinned_atomic_observation(
    robot: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms: dict[str, Any] = {}
    observation: dict[str, Any] = {}
    timestamps: dict[str, Any] = {}

    for side in ARM_SIDES:
        arm = getattr(robot, f"{side}_arm", None)
        if arm is None:
            raise AtomicObservationUnavailable(
                f"LeRobot v0.6.1 BiSOFollower is missing {side}_arm"
            )
        bus = getattr(arm, "bus", None)
        sync_read = getattr(bus, "sync_read", None)
        if bus is None or not callable(sync_read):
            raise AtomicObservationUnavailable(
                f"LeRobot v0.6.1 {side}_arm lacks motor-bus sync_read"
            )
        arm_config = getattr(arm, "config", None)
        num_read_retries = getattr(arm_config, "num_read_retries", None)
        if (
            not isinstance(num_read_retries, int)
            or isinstance(num_read_retries, bool)
            or num_read_retries < 0
        ):
            raise AtomicObservationUnavailable(
                f"LeRobot v0.6.1 {side}_arm lacks num_read_retries"
            )
        positions = sync_read(
            "Present_Position",
            num_retry=num_read_retries,
        )
        if not isinstance(positions, Mapping):
            raise ContractError(f"{side} Present_Position read must return a mapping")
        actual_joints = set(positions)
        expected_joints = set(ARM_JOINT_NAMES)
        if actual_joints != expected_joints:
            missing = tuple(sorted(expected_joints - actual_joints))
            extra = tuple(sorted(actual_joints - expected_joints))
            raise ContractError(
                f"{side} Present_Position keys must be exact; "
                f"missing={missing}, extra={extra}"
            )
        observation.update(
            {
                f"{side}_{joint}.pos": positions[joint]
                for joint in ARM_JOINT_NAMES
            }
        )
        arms[side] = arm

    for side, key in zip(ARM_SIDES, PRIMARY_CAMERA_KEYS, strict=True):
        cameras = getattr(arms[side], "cameras", None)
        if not isinstance(cameras, Mapping) or set(cameras) != {"wrist"}:
            raise AtomicObservationUnavailable(
                f"LeRobot v0.6.1 {side}_arm cameras must be exactly ('wrist',)"
            )
        image, timestamp = _atomic_camera_snapshot(cameras["wrist"], key=key)
        observation[key] = image
        timestamps[key] = timestamp

    return observation, timestamps


def read_lerobot_observation_atomic(
    robot: Any,
    *,
    timestamp_s: float,
    require_timestamps: bool = True,
    camera_clock: Callable[[], float] = perf_counter,
    max_camera_age_s: float | None = None,
    allow_external_provider: bool = True,
) -> SensorimotorFrame:
    """Read motors plus exact frame/timestamp pairs from a BiSOFollower.

    LeRobot 0.6.1 returns images without timestamps from ``get_observation()``.
    Reading ``latest_timestamp`` after that call can race the camera thread. The
    pinned path below instead copies ``latest_frame`` and ``latest_timestamp``
    under the camera's own ``frame_lock``. A duck-typed external provider is
    accepted only when it explicitly returns an atomic ``(observation,
    timestamps)`` pair and real actuation is not enabled. Real actuation must
    use the pinned internal path.
    """

    provider = (
        getattr(robot, "get_atomic_observation", None)
        if allow_external_provider
        else None
    )
    if callable(provider):
        supplied = provider()
        if not isinstance(supplied, tuple) or len(supplied) != 2:
            raise ContractError(
                "get_atomic_observation() must return (observation, timestamps)"
            )
        observation, raw_timestamps = supplied
        if not isinstance(observation, Mapping) or not isinstance(
            raw_timestamps, Mapping
        ):
            raise ContractError(
                "get_atomic_observation() must return two mappings"
            )
    else:
        observation, raw_timestamps = _pinned_atomic_observation(robot)

    timestamps = _camera_timestamps(
        raw_timestamps,
        require_timestamps=require_timestamps,
        camera_clock=camera_clock,
        max_camera_age_s=max_camera_age_s,
    )
    return observation_from_lerobot(
        observation,
        timestamp_s=timestamp_s,
        image_timestamps_s=timestamps,
    )


def action_to_lerobot(action: ActionChunk | np.ndarray) -> dict[str, float]:
    if isinstance(action, ActionChunk):
        if action.horizon != 1:
            raise ContractError(
                "ActionChunk horizon must be 1; send one row per servo tick"
            )
        target = np.array(action.target_joint_position[0], dtype=np.float32, copy=True)
    else:
        target = np.array(action, dtype=np.float32, copy=True)
    if target.shape != (ACTION_DIM,):
        raise ContractError(
            f"action must have shape ({ACTION_DIM},), got {target.shape}"
        )
    if not np.isfinite(target).all():
        raise ContractError("action contains NaN or infinity")
    return {key: float(value) for key, value in zip(JOINT_KEYS, target, strict=True)}


def _map_send_return(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        actual = set(value)
        expected = set(JOINT_KEYS)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            extra = tuple(sorted(actual - expected))
            raise ContractError(
                f"LeRobot send_action return keys must be exact; missing={missing}, extra={extra}"
            )
        return _as_joint_vector(value, name="LeRobot send_action return")
    array = np.array(value, dtype=np.float32, copy=True)
    if array.shape != (ACTION_DIM,):
        raise ContractError(
            f"LeRobot send_action return must have shape ({ACTION_DIM},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ContractError("LeRobot send_action return contains NaN or infinity")
    array.flags.writeable = False
    return array


def _disconnect_all_resources(
    robot: Any,
    *,
    report_top_level_error: bool,
) -> None:
    """Visit both official arms even when top-level disconnect short-circuits."""

    top_level_error: Exception | None = None
    try:
        robot.disconnect()
    except Exception as error:
        # BiSOFollower.disconnect() is decorated as "connected only". A left-arm
        # success followed by a right-arm failure makes the bimanual predicate
        # false, so that public cleanup path itself rejects the call.
        top_level_error = error

    arms: list[tuple[str, Any]] = []
    cleanup_errors: list[str] = []
    for side in ARM_SIDES:
        arm = getattr(robot, f"{side}_arm", None)
        if arm is None:
            continue
        arms.append((side, arm))

        arm_disconnect = getattr(arm, "disconnect", None)
        if bool(getattr(arm, "is_connected", False)) and callable(arm_disconnect):
            try:
                arm_disconnect()
            except Exception:
                # Fall through to resource-level cleanup below. It bypasses the
                # same all-components-connected decorator on a partially opened arm.
                pass

        bus = getattr(arm, "bus", None)
        if bool(getattr(bus, "is_connected", False)):
            bus_disconnect = getattr(bus, "disconnect", None)
            if not callable(bus_disconnect):
                cleanup_errors.append(f"{side}_bus:disconnect_unavailable")
            else:
                try:
                    bus_disconnect(
                        bool(
                            getattr(
                                getattr(arm, "config", None),
                                "disable_torque_on_disconnect",
                                True,
                            )
                        )
                    )
                except Exception as error:
                    cleanup_errors.append(
                        f"{side}_bus:{type(error).__name__}:{error}"
                    )

        cameras = getattr(arm, "cameras", None)
        if isinstance(cameras, Mapping):
            for camera_name, camera in cameras.items():
                camera_open = bool(getattr(camera, "is_connected", False)) or (
                    getattr(camera, "thread", None) is not None
                )
                if not camera_open:
                    continue
                camera_disconnect = getattr(camera, "disconnect", None)
                if not callable(camera_disconnect):
                    cleanup_errors.append(
                        f"{side}_{camera_name}:disconnect_unavailable"
                    )
                else:
                    try:
                        camera_disconnect()
                    except Exception as error:
                        cleanup_errors.append(
                            f"{side}_{camera_name}:{type(error).__name__}:{error}"
                        )

    remaining: list[str] = []
    for side, arm in arms:
        if bool(getattr(arm, "is_connected", False)):
            remaining.append(f"{side}_arm")
        bus = getattr(arm, "bus", None)
        if bool(getattr(bus, "is_connected", False)):
            remaining.append(f"{side}_bus")
        cameras = getattr(arm, "cameras", None)
        if isinstance(cameras, Mapping):
            remaining.extend(
                f"{side}_{name}"
                for name, camera in cameras.items()
                if bool(getattr(camera, "is_connected", False))
                or getattr(camera, "thread", None) is not None
            )
    if not arms and bool(getattr(robot, "is_connected", False)):
        remaining.append("robot")

    if cleanup_errors or remaining or (top_level_error is not None and not arms):
        details = tuple(cleanup_errors + [f"still_connected:{x}" for x in remaining])
        raise RuntimeError(
            f"top-level disconnect failed ({top_level_error}); "
            f"partial cleanup failed: {details}"
        ) from top_level_error
    if report_top_level_error and top_level_error is not None:
        raise RuntimeError(
            f"top-level disconnect failed ({top_level_error}); "
            "independent arm cleanup completed"
        ) from top_level_error


class LeRobotBiSOAdapter:
    """Small duck-typed wrapper around a LeRobot 0.6.1 BiSOFollower-like object."""

    joint_keys = JOINT_KEYS
    camera_keys = PRIMARY_CAMERA_KEYS
    backend = "lerobot"

    def __init__(
        self,
        robot: Any,
        *,
        config: LeRobotConfig | None = None,
        actuation_enabled: bool = False,
        max_camera_age_s: float = 0.10,
        camera_clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.robot = robot
        self.config = config or LeRobotConfig()
        if actuation_enabled:
            self.config.require_actuation_identity()
        if not isfinite(max_camera_age_s) or max_camera_age_s <= 0:
            raise ValueError("max_camera_age_s must be finite and positive")
        self.actuation_enabled = actuation_enabled
        self.max_camera_age_s = float(max_camera_age_s)
        self.camera_clock = camera_clock

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self.robot, "is_connected", False))

    @property
    def is_calibrated(self) -> bool:
        return bool(getattr(self.robot, "is_calibrated", False))

    def connect(self, *, calibrate: bool = True) -> None:
        if not hasattr(self.robot, "connect"):
            raise TypeError("robot must expose connect(calibrate=...)")
        if not hasattr(self.robot, "disconnect"):
            raise TypeError("robot must expose disconnect()")
        try:
            self.robot.connect(calibrate=calibrate)
            if not self.is_connected:
                raise RuntimeError("LeRobot robot did not report a connected state")
            if self.actuation_enabled and not self.is_calibrated:
                raise RuntimeError(
                    "real actuation requires robot.is_calibrated=true after connect"
                )
        except Exception as connect_error:
            try:
                _disconnect_all_resources(
                    self.robot,
                    report_top_level_error=False,
                )
            except Exception as disconnect_error:
                raise RuntimeError(
                    f"LeRobot connect failed ({connect_error}) and cleanup disconnect failed"
                ) from disconnect_error
            raise

    def disconnect(self) -> None:
        if not hasattr(self.robot, "disconnect"):
            raise TypeError("robot must expose disconnect()")
        _disconnect_all_resources(
            self.robot,
            report_top_level_error=True,
        )
        if self.is_connected:
            raise RuntimeError("LeRobot robot remained connected after disconnect()")

    def get_observation(self, *, timestamp_s: float) -> SensorimotorFrame:
        if not self.is_connected:
            raise RuntimeError("LeRobot robot must be connected before observation")
        try:
            return read_lerobot_observation_atomic(
                self.robot,
                timestamp_s=timestamp_s,
                require_timestamps=self.actuation_enabled,
                camera_clock=self.camera_clock,
                max_camera_age_s=(
                    self.max_camera_age_s if self.actuation_enabled else None
                ),
                allow_external_provider=not self.actuation_enabled,
            )
        except AtomicObservationUnavailable:
            if self.actuation_enabled:
                raise ContractError(
                    "real actuation requires atomic LeRobot wrist frame/timestamp reads"
                ) from None
            if not hasattr(self.robot, "get_observation"):
                raise TypeError("robot must expose get_observation()") from None
            return observation_from_lerobot(
                self.robot.get_observation(),
                timestamp_s=timestamp_s,
            )

    def send_action(self, action: ActionChunk | np.ndarray) -> np.ndarray | None:
        if not self.actuation_enabled:
            raise ActuationDisabledError(
                "real LeRobot actuation is disabled by default"
            )
        if not self.is_connected:
            raise RuntimeError("LeRobot robot must be connected before actuation")
        if not self.is_calibrated:
            raise RuntimeError("real actuation requires robot.is_calibrated=true")
        if not hasattr(self.robot, "send_action"):
            raise TypeError("robot must expose send_action(action)")
        return _map_send_return(self.robot.send_action(action_to_lerobot(action)))


class LeRobotBiSOLeader:
    """Read a calibrated LeRobot 0.6.1 BiSOLeader without follower writes."""

    joint_keys = JOINT_KEYS
    backend = "lerobot_leader"

    def __init__(self, raw_leader: Any) -> None:
        self._leader = raw_leader

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._leader, "is_connected", False))

    @property
    def is_calibrated(self) -> bool:
        return bool(getattr(self._leader, "is_calibrated", False))

    def connect(self) -> None:
        connect = getattr(self._leader, "connect", None)
        if not callable(connect):
            raise TypeError("leader must expose connect(calibrate=...)")
        disconnect = getattr(self._leader, "disconnect", None)
        if not callable(disconnect):
            raise TypeError("leader must expose disconnect()")
        try:
            connect(calibrate=False)
            if not self.is_connected:
                raise RuntimeError("LeRobot leader did not report a connected state")
            if not self.is_calibrated:
                raise RuntimeError("LeRobot leader requires is_calibrated=true")
        except Exception:
            _disconnect_all_resources(self._leader, report_top_level_error=False)
            raise

    def disconnect(self) -> None:
        if not hasattr(self._leader, "disconnect"):
            raise TypeError("leader must expose disconnect()")
        _disconnect_all_resources(self._leader, report_top_level_error=True)
        if self.is_connected:
            raise RuntimeError("LeRobot leader remained connected after disconnect()")

    def get_action(self) -> np.ndarray:
        if not self.is_connected:
            raise RuntimeError("LeRobot leader must be connected before action reads")
        if not self.is_calibrated:
            raise RuntimeError("LeRobot leader requires is_calibrated=true")
        get_action = getattr(self._leader, "get_action", None)
        if not callable(get_action):
            raise TypeError("leader must expose get_action()")

        action = get_action()
        if not isinstance(action, Mapping):
            raise ContractError("LeRobot leader action must be a mapping")
        actual = set(action)
        expected = set(JOINT_KEYS)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            extra = tuple(sorted(actual - expected))
            raise ContractError(
                f"LeRobot leader action keys must be exact; missing={missing}, extra={extra}"
            )
        return _as_joint_vector(action, name="LeRobot leader action")


__all__ = [
    "ActuationDisabledError",
    "AtomicObservationUnavailable",
    "LeRobotBiSOAdapter",
    "LeRobotBiSOLeader",
    "action_to_lerobot",
    "observation_from_lerobot",
    "read_lerobot_observation_atomic",
]
