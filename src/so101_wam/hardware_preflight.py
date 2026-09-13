"""Observation-only G6/G7 preflight for a real LeRobot dual SO-101."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from math import ceil, isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, perf_counter, sleep
from typing import Any

import numpy as np

from .config import ConfigError, LeRobotConfig, ProjectConfig
from .constants import ACTION_DIM, ARM_JOINT_NAMES, ARM_SIDES, PRIMARY_CAMERA_KEYS
from .contracts import ContractError, SensorimotorFrame
from .deployment import project_config_fingerprint
from .adapters.lerobot import read_lerobot_observation_atomic
from .lerobot_factory import LeRobotFactoryError, create_bi_so_follower


class HardwarePreflightError(RuntimeError):
    """Raised when a read-only hardware evidence session cannot be trusted."""


class HardwarePreflightReportError(OSError):
    """Raised when a report would overwrite or corrupt an existing artifact."""


class _ConnectionResource:
    def __init__(
        self,
        kind: str,
        value: Any,
        *,
        disable_torque_on_disconnect: bool = True,
    ) -> None:
        self.kind = kind
        self.value = value
        self.disable_torque_on_disconnect = disable_torque_on_disconnect


def _finite_time(clock: Callable[[], float], *, name: str) -> float:
    value = float(clock())
    if not isfinite(value) or value < 0:
        raise HardwarePreflightError(f"{name} returned an invalid timestamp")
    return value


def _sleep_until(
    deadline_s: float,
    *,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> float:
    now_s = _finite_time(clock, name="preflight clock")
    if now_s < deadline_s:
        sleeper(deadline_s - now_s)
        now_s = _finite_time(clock, name="preflight clock")
    if now_s + 1e-9 < deadline_s:
        raise HardwarePreflightError(
            "preflight clock did not advance to the requested deadline"
        )
    return now_s


def _bimanual_arms(robot: Any) -> tuple[Any, Any]:
    arms: list[Any] = []
    for side in ARM_SIDES:
        arm = getattr(robot, f"{side}_arm", None)
        if arm is None:
            raise HardwarePreflightError(
                f"LeRobot v0.6.1 BiSOFollower is missing {side}_arm"
            )
        if not hasattr(arm, "bus") or not hasattr(arm, "cameras"):
            raise HardwarePreflightError(
                f"LeRobot v0.6.1 {side}_arm is missing bus/cameras"
            )
        arms.append(arm)
    return arms[0], arms[1]


def _disconnect_resources(resources: list[_ConnectionResource]) -> None:
    failures: list[str] = []
    for resource in reversed(resources):
        try:
            if getattr(resource.value, "is_connected", None) is False:
                continue
            if resource.kind == "camera":
                resource.value.disconnect()
            else:
                resource.value.disconnect(resource.disable_torque_on_disconnect)
        except Exception as error:  # best-effort cleanup must visit every resource
            failures.append(f"{resource.kind}:{type(error).__name__}:{error}")
    resources.clear()
    if failures:
        raise HardwarePreflightError(
            f"observation-only hardware cleanup failed: {tuple(failures)}"
        )


def _read_present_joint_position(arms: Sequence[Any]) -> np.ndarray:
    values: list[float] = []
    expected_joints = set(ARM_JOINT_NAMES)
    for side, arm in zip(ARM_SIDES, arms, strict=True):
        bus = arm.bus
        sync_read = getattr(bus, "sync_read", None)
        if not callable(sync_read):
            raise HardwarePreflightError(
                f"{side} motor bus cannot read Present_Position"
            )
        num_read_retries = getattr(getattr(arm, "config", None), "num_read_retries", None)
        if (
            not isinstance(num_read_retries, int)
            or isinstance(num_read_retries, bool)
            or num_read_retries < 0
        ):
            raise HardwarePreflightError(
                f"{side} arm lacks a valid num_read_retries setting"
            )
        positions = sync_read(
            "Present_Position",
            num_retry=num_read_retries,
        )
        if not isinstance(positions, Mapping):
            raise HardwarePreflightError(
                f"{side} Present_Position read must return a mapping"
            )
        actual_joints = set(positions)
        if actual_joints != expected_joints:
            missing = tuple(sorted(expected_joints - actual_joints))
            extra = tuple(sorted(actual_joints - expected_joints))
            raise HardwarePreflightError(
                f"{side} Present_Position keys must be exact; "
                f"missing={missing}, extra={extra}"
            )
        values.extend(float(positions[joint]) for joint in ARM_JOINT_NAMES)

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (ACTION_DIM,) or not np.isfinite(vector).all():
        raise HardwarePreflightError(
            f"Present_Position must contain {ACTION_DIM} finite values"
        )
    return vector


def _require_safe_torque_release_pose(
    arms: Sequence[Any],
    config: ProjectConfig,
) -> None:
    if not config.safety.calibrated:
        raise HardwarePreflightError(
            "torque release requires a measured calibrated safety profile"
        )
    home = config.lerobot.home_joint_position
    tolerance = config.lerobot.home_joint_tolerance
    if home is None or tolerance is None:
        raise HardwarePreflightError(
            "torque release requires a measured home position and tolerance"
        )

    position = _read_present_joint_position(arms)
    lower = np.asarray(config.safety.joint_lower, dtype=np.float64)
    upper = np.asarray(config.safety.joint_upper, dtype=np.float64)
    outside_limits = np.flatnonzero((position < lower) | (position > upper))
    if outside_limits.size:
        raise HardwarePreflightError(
            "torque release pose is outside measured safety limits at indices "
            f"{tuple(int(index) for index in outside_limits)}"
        )

    home_error = np.abs(position - np.asarray(home, dtype=np.float64))
    outside_home = np.flatnonzero(
        home_error > np.asarray(tolerance, dtype=np.float64)
    )
    if outside_home.size:
        raise HardwarePreflightError(
            "torque release pose exceeds measured home tolerance at indices "
            f"{tuple(int(index) for index in outside_home)}"
        )


@contextmanager
def observation_only_connection(
    robot: Any,
    *,
    disable_torque_for_session: bool = False,
    torque_release_config: ProjectConfig | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Connect buses/cameras without SOFollower.configure() or Goal_Position writes.

    Official ``SOFollower.connect()`` calls ``configure()``, whose torque-disabled
    context re-enables torque. This path deliberately performs only motor-bus
    handshakes, calibration verification, optional torque disable, camera
    connection, normalized reads, and the configured torque-safe bus disconnect.
    """

    arms = _bimanual_arms(robot)
    resources: list[_ConnectionResource] = []
    try:
        for side, arm in zip(ARM_SIDES, arms, strict=True):
            bus = arm.bus
            if not hasattr(bus, "connect") or not hasattr(bus, "disconnect"):
                raise HardwarePreflightError(f"{side} motor bus has no lifecycle API")
            resources.append(
                _ConnectionResource(
                    "bus",
                    bus,
                    disable_torque_on_disconnect=bool(
                        getattr(
                            getattr(arm, "config", None),
                            "disable_torque_on_disconnect",
                            True,
                        )
                    ),
                )
            )
            bus.connect()
            if not bool(getattr(arm, "is_calibrated", False)):
                raise HardwarePreflightError(
                    f"{side} arm must have valid LeRobot calibration before "
                    "observation-only use"
                )
        if disable_torque_for_session:
            if torque_release_config is None:
                raise HardwarePreflightError(
                    "torque-off kinesthetic capture requires an explicit "
                    "torque-release safety config"
                )
            disable_torque_calls: list[Callable[[], None]] = []
            for side, arm in zip(ARM_SIDES, arms, strict=True):
                disable_torque = getattr(arm.bus, "disable_torque", None)
                if not callable(disable_torque):
                    raise HardwarePreflightError(
                        f"{side} motor bus cannot guarantee torque-off "
                        "kinesthetic capture"
                    )
                disable_torque_calls.append(disable_torque)
            _require_safe_torque_release_pose(arms, torque_release_config)
            for disable_torque in disable_torque_calls:
                disable_torque()

        for side, arm in zip(ARM_SIDES, arms, strict=True):
            cameras = arm.cameras
            if not isinstance(cameras, Mapping) or set(cameras) != {"wrist"}:
                raise HardwarePreflightError(
                    f"{side} arm cameras must contain exactly the local 'wrist' camera"
                )
            camera = cameras["wrist"]
            if not hasattr(camera, "connect") or not hasattr(camera, "disconnect"):
                raise HardwarePreflightError(
                    f"{side} wrist camera has no lifecycle API"
                )
            resources.append(_ConnectionResource("camera", camera))
            camera.connect()

        if not bool(getattr(robot, "is_connected", False)):
            raise HardwarePreflightError(
                "BiSOFollower did not report connected after read-only bus/camera setup"
            )
    except BaseException as primary_error:
        try:
            _disconnect_resources(resources)
        except Exception as cleanup_error:
            raise HardwarePreflightError(
                "observation-only connection failed "
                f"({primary_error}) and cleanup also failed"
            ) from cleanup_error
        raise

    try:
        yield arms
    except BaseException as primary_error:
        try:
            _disconnect_resources(resources)
        except Exception as cleanup_error:
            raise HardwarePreflightError(
                "observation-only session failed "
                f"({primary_error}) and cleanup also failed"
            ) from cleanup_error
        raise
    else:
        _disconnect_resources(resources)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _effective_fps(values: Sequence[float | None]) -> float | None:
    available = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    if available.size < 2:
        return None
    unique = np.unique(available)
    if unique.size < 2:
        return None
    span = float(unique[-1] - unique[0])
    return float((unique.size - 1) / span) if span > 0 else None


def _timestamp_regressed(values: Sequence[float | None]) -> bool:
    available = [value for value in values if value is not None]
    return any(
        current < previous for previous, current in zip(available, available[1:])
    )


def _validate_request(
    config: ProjectConfig,
    *,
    seconds: float,
    sample_hz: float,
) -> None:
    if config.runtime.backend != "lerobot":
        raise HardwarePreflightError(
            "hardware preflight requires runtime.backend='lerobot'"
        )
    config.lerobot.require_hardware_session()
    if not isfinite(seconds) or seconds <= 0:
        raise HardwarePreflightError("preflight seconds must be finite and positive")
    if not isfinite(sample_hz) or sample_hz <= 0:
        raise HardwarePreflightError("preflight sample_hz must be finite and positive")
    if sample_hz > config.lerobot.camera_fps:
        raise HardwarePreflightError(
            "preflight sample_hz cannot exceed the configured camera_fps"
        )
    if seconds + 1e-9 < 1.0 / sample_hz:
        raise HardwarePreflightError(
            "preflight seconds must cover at least one sampling interval"
        )


def run_hardware_preflight(
    config: ProjectConfig,
    *,
    seconds: float = 10.0,
    sample_hz: float | None = None,
    robot_factory: Callable[[LeRobotConfig], Any] | None = None,
    clock: Callable[[], float] = monotonic,
    camera_clock: Callable[[], float] = perf_counter,
    sleeper: Callable[[float], None] = sleep,
) -> dict[str, Any]:
    """Collect G6/G7 evidence without calling robot.connect() or send_action()."""

    effective_sample_hz = float(
        config.lerobot.camera_fps if sample_hz is None else sample_hz
    )
    _validate_request(
        config,
        seconds=float(seconds),
        sample_hz=effective_sample_hz,
    )
    factory = robot_factory or create_bi_so_follower
    robot = factory(config.lerobot)
    sample_count = max(2, ceil(float(seconds) * effective_sample_hz) + 1)

    capture_started_s: list[float] = []
    capture_ended_s: list[float] = []
    observation_latency_s: list[float] = []
    left_camera_timestamp_s: list[float | None] = []
    right_camera_timestamp_s: list[float | None] = []
    left_camera_age_s: list[float | None] = []
    right_camera_age_s: list[float | None] = []
    joint_positions: list[np.ndarray] = []
    resolutions: set[tuple[int, int]] = set()
    calibrated = False

    with observation_only_connection(robot):
        calibrated = bool(getattr(robot, "is_calibrated", False))
        start_s = _finite_time(clock, name="preflight clock")
        for index in range(sample_count):
            deadline_s = start_s + index / effective_sample_hz
            capture_started = _sleep_until(
                deadline_s,
                clock=clock,
                sleeper=sleeper,
            )
            frame: SensorimotorFrame = read_lerobot_observation_atomic(
                robot,
                timestamp_s=capture_started,
                require_timestamps=False,
                camera_clock=camera_clock,
                max_camera_age_s=None,
                allow_external_provider=False,
            )
            capture_ended = _finite_time(clock, name="preflight clock")
            if capture_ended < capture_started:
                raise HardwarePreflightError(
                    "preflight clock moved backwards during observation capture"
                )

            capture_started_s.append(capture_started)
            capture_ended_s.append(capture_ended)
            observation_latency_s.append(capture_ended - capture_started)
            frame_timestamps = frame.image_timestamps_s
            if frame_timestamps is None:
                left_timestamp = None
                right_timestamp = None
            else:
                left_timestamp = frame_timestamps[PRIMARY_CAMERA_KEYS[0]]
                right_timestamp = frame_timestamps[PRIMARY_CAMERA_KEYS[1]]
            left_camera_timestamp_s.append(left_timestamp)
            right_camera_timestamp_s.append(right_timestamp)
            camera_now_s = _finite_time(camera_clock, name="camera clock")
            left_camera_age_s.append(
                None
                if left_timestamp is None
                else camera_now_s - left_timestamp
            )
            right_camera_age_s.append(
                None
                if right_timestamp is None
                else camera_now_s - right_timestamp
            )
            joint_positions.append(
                np.array(frame.joint_position, dtype=np.float64, copy=True)
            )
            image_shape = frame.primary_images[0].shape
            resolutions.add((int(image_shape[0]), int(image_shape[1])))

    positions = np.stack(joint_positions, axis=0)
    joint_median = np.median(positions, axis=0)
    joint_repeatability = np.max(np.abs(positions - joint_median), axis=0)
    observation_p95 = _percentile(observation_latency_s, 95.0)
    observation_max = max(observation_latency_s)
    poll_span_s = capture_started_s[-1] - capture_started_s[0]
    poll_fps = (sample_count - 1) / poll_span_s if poll_span_s > 0 else None
    camera_timestamps_complete = all(
        value is not None
        for value in (*left_camera_timestamp_s, *right_camera_timestamp_s)
    )
    skew_samples = [
        abs(left - right)
        for left, right in zip(
            left_camera_timestamp_s,
            right_camera_timestamp_s,
            strict=True,
        )
        if left is not None and right is not None
    ]
    camera_skew_p95 = _percentile(skew_samples, 95.0)
    camera_fps = {
        "left_wrist": _effective_fps(left_camera_timestamp_s),
        "right_wrist": _effective_fps(right_camera_timestamp_s),
    }
    available_camera_ages = [
        value
        for value in (*left_camera_age_s, *right_camera_age_s)
        if value is not None and value >= 0
    ]
    camera_age_p50 = _percentile(available_camera_ages, 50.0)
    camera_age_p95 = _percentile(available_camera_ages, 95.0)
    camera_age_max = (
        max(available_camera_ages) if available_camera_ages else None
    )

    home = config.lerobot.home_joint_position
    tolerance = config.lerobot.home_joint_tolerance
    home_error: np.ndarray | None = None
    home_within_tolerance: bool | None = None
    if home is not None and tolerance is not None:
        home_error = np.max(
            np.abs(positions - np.asarray(home, dtype=np.float64)), axis=0
        )
        home_within_tolerance = bool(
            np.all(home_error <= np.asarray(tolerance, dtype=np.float64))
        )

    expected_resolution = (
        config.lerobot.camera_height,
        config.lerobot.camera_width,
    )
    fps_floor = config.lerobot.camera_fps * 0.95
    sampling_floor = effective_sample_hz * 0.95
    failures: list[str] = []
    if not config.lerobot.hardware_id.strip():
        failures.append("hardware_id_missing")
    if not config.lerobot.calibration_id.strip():
        failures.append("calibration_id_missing")
    if not calibrated:
        failures.append("lerobot_not_calibrated")
    if resolutions != {expected_resolution}:
        failures.append("camera_resolution_mismatch")
    if poll_fps is None or poll_fps < sampling_floor:
        failures.append("observation_sampling_fps_below_target")
    if effective_sample_hz < fps_floor:
        failures.append("sampling_rate_below_camera_target")
    if observation_p95 is None or (
        observation_p95 > config.safety.max_observation_age_s
    ):
        failures.append("observation_latency_p95_exceeded")
    if not camera_timestamps_complete:
        failures.append("camera_timestamp_unavailable")
    if any(
        value is not None and value < -1e-6
        for value in (*left_camera_age_s, *right_camera_age_s)
    ):
        failures.append("camera_timestamp_in_future")
    if camera_age_p95 is None:
        failures.append("camera_frame_age_unavailable")
    elif camera_age_p95 > config.safety.max_observation_age_s:
        failures.append("camera_frame_age_p95_exceeded")
    if _timestamp_regressed(left_camera_timestamp_s) or _timestamp_regressed(
        right_camera_timestamp_s
    ):
        failures.append("camera_timestamp_regression")
    for side in PRIMARY_CAMERA_KEYS:
        estimate = camera_fps[side]
        if estimate is None or estimate < fps_floor:
            failures.append(f"{side}_fps_below_target")
    if camera_skew_p95 is None:
        failures.append("camera_skew_unavailable")
    elif camera_skew_p95 > config.safety.max_camera_skew_s:
        failures.append("camera_skew_p95_exceeded")
    if home is None or tolerance is None:
        failures.append("home_reference_missing")
    elif home_within_tolerance is not True:
        failures.append("home_tolerance_exceeded")

    unique_failures = list(dict.fromkeys(failures))
    automated_checks_passed = not unique_failures
    resolution = list(next(iter(resolutions))) if len(resolutions) == 1 else None
    manual_checks = [
        "confirm left/right wrist camera identity from live images",
        "confirm arms-down pose and joint direction labels",
        "confirm cable strain relief and collision clearance",
    ]

    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "real",
        "gate": "G6/G7-preflight",
        "mode": "observation_only",
        "result": "partial" if automated_checks_passed else "fail",
        "automated_checks_passed": automated_checks_passed,
        "failures": unique_failures,
        "manual_checks_required": manual_checks,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "config_fingerprint": project_config_fingerprint(config),
        "configured_actuation_enabled": config.runtime.actuation_enabled,
        "effective_actuation_enabled": False,
        "goal_position_commands_sent": 0,
        "lerobot_calibrated": calibrated,
        "duration_requested_s": float(seconds),
        "duration_observed_s": float(capture_ended_s[-1] - capture_started_s[0]),
        "sample_hz": effective_sample_hz,
        "frame_count": sample_count,
        "camera_keys": list(PRIMARY_CAMERA_KEYS),
        "camera_resolution": resolution,
        "camera_resolution_expected": list(expected_resolution),
        "observation_sampling_fps": poll_fps,
        "observation_latency_p50_s": _percentile(observation_latency_s, 50.0),
        "observation_latency_p95_s": observation_p95,
        "observation_latency_max_s": observation_max,
        "observation_latency_limit_s": config.safety.max_observation_age_s,
        "camera_timestamp_source": (
            "OpenCVCamera.frame_lock(latest_frame, latest_timestamp) "
            "atomic buffer snapshot (LeRobot v0.6.1 internal)"
            if camera_timestamps_complete
            else "unavailable"
        ),
        "camera_timestamp_samples": len(skew_samples),
        "camera_fps_estimate_hz": camera_fps,
        "camera_fps_floor_hz": fps_floor,
        "camera_frame_age_p50_s": camera_age_p50,
        "camera_frame_age_p95_s": camera_age_p95,
        "camera_frame_age_max_s": camera_age_max,
        "camera_frame_age_limit_s": config.safety.max_observation_age_s,
        "camera_skew_p95_s": camera_skew_p95,
        "camera_skew_limit_s": config.safety.max_camera_skew_s,
        "joint_order_width": ACTION_DIM,
        "joint_mean": np.mean(positions, axis=0).tolist(),
        "joint_std": np.std(positions, axis=0).tolist(),
        "home_position_candidate": joint_median.tolist(),
        "home_repeatability_max_abs": joint_repeatability.tolist(),
        "home_reference_configured": home is not None and tolerance is not None,
        "home_error_max_abs": None if home_error is None else home_error.tolist(),
        "home_within_tolerance": home_within_tolerance,
        "notes": [
            "No BiSOFollower.connect(), SOFollower.configure(), send_action(), or Goal_Position write is used.",
            "Bus disconnect follows each arm's disable_torque_on_disconnect setting.",
            "Each camera frame and timestamp is copied under the same version-pinned OpenCVCamera frame_lock.",
            "Camera timestamp telemetry is internal evidence, not a public v0.6.1 observation field.",
            "A partial result still requires the listed manual hardware checks before G6/G7 can pass.",
        ],
    }


def write_hardware_preflight_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            temp_path.replace(target)
            temp_path = None
        else:
            try:
                # A same-directory hard link publishes the already-fsynced file
                # atomically and fails if another process created the target.
                os.link(temp_path, target)
            except FileExistsError as error:
                raise HardwarePreflightReportError(
                    "preflight report already exists "
                    f"(use --overwrite explicitly): {target}"
                ) from error
            temp_path.unlink()
            temp_path = None
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect observation-only G6/G7 evidence from dual SO-101 hardware."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = run_hardware_preflight(
            ProjectConfig.load(args.config),
            seconds=args.seconds,
            sample_hz=args.sample_hz,
        )
        write_hardware_preflight_report(
            args.report,
            report,
            overwrite=args.overwrite,
        )
    except (
        ConfigError,
        ContractError,
        HardwarePreflightError,
        HardwarePreflightReportError,
        LeRobotFactoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if (
        report.get("automated_checks_passed") is not True
        or report.get("result") != "partial"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HardwarePreflightError",
    "HardwarePreflightReportError",
    "observation_only_connection",
    "run_hardware_preflight",
    "write_hardware_preflight_report",
]
