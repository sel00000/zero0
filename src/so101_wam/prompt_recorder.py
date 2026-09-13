"""Observation-only G9 physical-prompt recording for dual SO-101 hardware."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from time import monotonic, perf_counter, sleep
from typing import Any

import numpy as np

from .adapters.lerobot import read_lerobot_observation_atomic
from .config import ConfigError, LeRobotConfig, ProjectConfig
from .constants import ACTION_DIM, ARM_SIDES, PRIMARY_CAMERA_KEYS
from .contracts import ContractError, SensorimotorFrame
from .dataset import (
    DatasetArtifactExistsError,
    DatasetError,
    EpisodeBuffer,
    FPS_RELATIVE_TOLERANCE,
    load_episode,
    physical_prompt_from_episode,
    save_episode,
)
from .hardware_preflight import (
    HardwarePreflightError,
    observation_only_connection,
)
from .lerobot_factory import LeRobotFactoryError, create_bi_so_follower

ACTION_SOURCE = "measured_present_position_no_goal_write"
CAPTURE_MODE = "observation_only_kinesthetic"


class PromptRecorderError(RuntimeError):
    """Raised when a physical-prompt artifact cannot be trusted."""


def _finite_time(clock: Callable[[], float], *, name: str) -> float:
    value = float(clock())
    if not isfinite(value) or value < 0:
        raise PromptRecorderError(f"{name} returned an invalid timestamp")
    return value


def _sleep_until(
    deadline_s: float,
    *,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> float:
    now_s = _finite_time(clock, name="prompt recorder clock")
    if now_s < deadline_s:
        sleeper(deadline_s - now_s)
        now_s = _finite_time(clock, name="prompt recorder clock")
    if now_s + 1e-9 < deadline_s:
        raise PromptRecorderError(
            "prompt recorder clock did not advance to the requested deadline"
        )
    return now_s


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise PromptRecorderError("cannot summarize an empty timing series")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _effective_fps(timestamps_s: Sequence[float]) -> float | None:
    if len(timestamps_s) < 2:
        return None
    span_s = float(timestamps_s[-1] - timestamps_s[0])
    return (len(timestamps_s) - 1) / span_s if span_s > 0 else None


def _validate_recording_request(
    config: ProjectConfig,
    *,
    task: str,
    duration_s: float,
    episode_index: int,
    task_index: int,
) -> tuple[float, int]:
    if config.runtime.backend != "lerobot":
        raise PromptRecorderError(
            "physical-prompt recording requires runtime.backend='lerobot'"
        )
    config.lerobot.require_hardware_session()
    missing_identity = tuple(
        name
        for name, value in (
            ("hardware_id", config.lerobot.hardware_id),
            ("calibration_id", config.lerobot.calibration_id),
        )
        if not value.strip()
    )
    if missing_identity:
        raise PromptRecorderError(
            f"physical-prompt recording requires identity fields: {missing_identity}"
        )
    if not config.safety.calibrated:
        raise PromptRecorderError(
            "physical-prompt recording requires a measured calibrated safety profile"
        )
    if (
        config.lerobot.home_joint_position is None
        or config.lerobot.home_joint_tolerance is None
    ):
        raise PromptRecorderError(
            "physical-prompt recording requires a measured home position and tolerance"
        )
    if not isinstance(task, str) or not task.strip():
        raise PromptRecorderError("task must be a non-empty string")
    if (
        not isinstance(episode_index, int)
        or isinstance(episode_index, bool)
        or episode_index < 0
    ):
        raise PromptRecorderError("episode_index must be a non-negative integer")
    if (
        not isinstance(task_index, int)
        or isinstance(task_index, bool)
        or task_index < 0
    ):
        raise PromptRecorderError("task_index must be a non-negative integer")
    if not isfinite(duration_s):
        raise PromptRecorderError("duration_s must be finite")
    if not (
        config.runtime.prompt_min_seconds
        <= duration_s
        <= config.runtime.prompt_max_seconds
    ):
        raise PromptRecorderError(
            "duration_s must be within configured physical-prompt bounds "
            f"[{config.runtime.prompt_min_seconds:g}, "
            f"{config.runtime.prompt_max_seconds:g}]"
        )

    fps = float(config.runtime.camera_hz)
    interval_count = round(duration_s * fps)
    effective_duration_s = interval_count / fps
    if abs(effective_duration_s - duration_s) > 1e-9:
        raise PromptRecorderError(
            f"duration_s must align to the {fps:g} Hz capture grid; "
            f"nearest duration is {effective_duration_s:.9f}s"
        )
    if interval_count < 1:
        raise PromptRecorderError("duration_s must contain at least one interval")
    return fps, interval_count


def _artifact_paths(
    output_dir: str | Path,
    *,
    episode_index: int,
) -> tuple[Path, Path, str]:
    directory = Path(output_dir)
    stem = f"prompt_{episode_index:06d}"
    return directory / f"{stem}.npz", directory / f"{stem}.json", stem


def _require_available_artifact_paths(
    npz_path: Path,
    manifest_path: Path,
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if npz_path.exists() or manifest_path.exists():
        raise DatasetArtifactExistsError(
            "prompt artifact already exists; choose a new --episode-index: "
            f"{npz_path}, {manifest_path}"
        )


def _relative(values: Sequence[float]) -> list[float]:
    origin = values[0]
    return [float(value - origin) for value in values]


def record_physical_prompt(
    config: ProjectConfig,
    *,
    output_dir: str | Path,
    task: str,
    duration_s: float = 3.0,
    episode_index: int = 0,
    task_index: int = 0,
    robot_factory: Callable[[LeRobotConfig], Any] | None = None,
    clock: Callable[[], float] = monotonic,
    camera_clock: Callable[[], float] = perf_counter,
    sleeper: Callable[[float], None] = sleep,
) -> dict[str, Any]:
    """Record and checksum one no-command, dual-wrist kinesthetic prompt.

    The dataset ``action`` column is a measured joint-position trajectory, not a
    value sent through ``Goal_Position``. This distinction is persisted in the
    manifest metadata. The command path never calls ``robot.connect()``,
    ``configure()``, or ``send_action()``. It reads all 12 joints after the
    direct bus handshake and disables torque only when the measured pose is
    inside both the safety limits and arms-down home tolerance.
    """

    fps, interval_count = _validate_recording_request(
        config,
        task=task,
        duration_s=float(duration_s),
        episode_index=episode_index,
        task_index=task_index,
    )
    npz_path, manifest_path, stem = _artifact_paths(
        output_dir,
        episode_index=episode_index,
    )
    _require_available_artifact_paths(
        npz_path,
        manifest_path,
    )

    factory = robot_factory or create_bi_so_follower
    robot = factory(config.lerobot)
    expected_resolution = (
        config.lerobot.camera_height,
        config.lerobot.camera_width,
    )
    expected_dt_s = 1.0 / fps
    max_dt_error_s = expected_dt_s * FPS_RELATIVE_TOLERANCE
    frame_count = interval_count + 1
    base_metadata: dict[str, Any] = {
        "gate": "G9-prompt-record",
        "evidence_level": "real",
        "capture_mode": CAPTURE_MODE,
        "action_source": ACTION_SOURCE,
        "action_interpretation": (
            "kinesthetic measured-position target proxy; no motor target was emitted"
        ),
        "goal_position_commands_sent": 0,
        "torque_disabled_for_capture": True,
        "torque_disable_operations": len(ARM_SIDES),
        "torque_release_home_verified": True,
        "configured_actuation_enabled": config.runtime.actuation_enabled,
        "effective_actuation_enabled": False,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "camera_keys": list(PRIMARY_CAMERA_KEYS),
        "head_camera_included": False,
        "camera_timestamp_source": (
            "OpenCVCamera.frame_lock(latest_frame, latest_timestamp) atomic "
            "buffer snapshot (LeRobot v0.6.1 internal)"
        ),
        "dataset_timestamp_source": (
            "fixed-rate capture grid validated against monotonic capture starts"
        ),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    buffer = EpisodeBuffer(
        fps=fps,
        task=task.strip(),
        episode_index=episode_index,
        task_index=task_index,
        metadata=base_metadata,
    )

    capture_started_s: list[float] = []
    capture_lateness_s: list[float] = []
    observation_latency_s: list[float] = []
    camera_timestamps_s: dict[str, list[float]] = {
        key: [] for key in PRIMARY_CAMERA_KEYS
    }
    camera_age_s: list[float] = []
    camera_skew_s: list[float] = []

    with observation_only_connection(
        robot,
        disable_torque_for_session=True,
        torque_release_config=config,
    ):
        session_start_s = _finite_time(clock, name="prompt recorder clock")
        for index in range(frame_count):
            deadline_s = session_start_s + index * expected_dt_s
            capture_started = _sleep_until(
                deadline_s,
                clock=clock,
                sleeper=sleeper,
            )
            if capture_started_s:
                actual_dt_s = capture_started - capture_started_s[-1]
                if abs(actual_dt_s - expected_dt_s) > max_dt_error_s:
                    raise PromptRecorderError(
                        f"capture timestamp step {actual_dt_s:.6f}s exceeds the "
                        f"{FPS_RELATIVE_TOLERANCE:.0%} tolerance at frame {index}"
                    )

            frame = read_lerobot_observation_atomic(
                robot,
                timestamp_s=index * expected_dt_s,
                require_timestamps=True,
                camera_clock=camera_clock,
                max_camera_age_s=config.safety.max_observation_age_s,
                allow_external_provider=False,
            )
            capture_ended = _finite_time(clock, name="prompt recorder clock")
            if capture_ended < capture_started:
                raise PromptRecorderError(
                    "prompt recorder clock moved backwards during capture"
                )
            observation_latency = capture_ended - capture_started
            if observation_latency > config.safety.max_observation_age_s:
                raise PromptRecorderError(
                    f"observation assembly latency {observation_latency:.6f}s "
                    f"exceeds {config.safety.max_observation_age_s:.6f}s"
                )
            if frame.primary_images[0].shape[:2] != expected_resolution:
                raise PromptRecorderError(
                    "wrist camera resolution mismatch: expected "
                    f"{expected_resolution}, got {frame.primary_images[0].shape[:2]}"
                )

            timestamps = frame.image_timestamps_s
            if timestamps is None:  # guarded by require_timestamps, kept explicit
                raise PromptRecorderError("wrist camera timestamps are unavailable")
            current_timestamps = {
                key: float(timestamps[key]) for key in PRIMARY_CAMERA_KEYS
            }
            for key in PRIMARY_CAMERA_KEYS:
                prior = camera_timestamps_s[key]
                if prior and current_timestamps[key] <= prior[-1]:
                    raise PromptRecorderError(
                        f"{key} camera timestamp did not advance at frame {index}"
                    )
                if prior:
                    camera_dt_s = current_timestamps[key] - prior[-1]
                    if abs(camera_dt_s - expected_dt_s) > max_dt_error_s:
                        raise PromptRecorderError(
                            f"{key} camera timestamp step {camera_dt_s:.6f}s "
                            f"exceeds the {FPS_RELATIVE_TOLERANCE:.0%} "
                            f"tolerance at frame {index}"
                        )

            camera_now_s = _finite_time(camera_clock, name="camera clock")
            current_ages = [
                camera_now_s - current_timestamps[key]
                for key in PRIMARY_CAMERA_KEYS
            ]
            for key, age_s in zip(
                PRIMARY_CAMERA_KEYS,
                current_ages,
                strict=True,
            ):
                if age_s < -1e-6:
                    raise PromptRecorderError(
                        f"{key} camera timestamp is in the future"
                    )
                if age_s > config.safety.max_observation_age_s:
                    raise PromptRecorderError(
                        f"{key} camera frame age {age_s:.6f}s exceeds "
                        f"{config.safety.max_observation_age_s:.6f}s"
                    )
            skew_s = abs(
                current_timestamps[PRIMARY_CAMERA_KEYS[0]]
                - current_timestamps[PRIMARY_CAMERA_KEYS[1]]
            )
            if skew_s > config.safety.max_camera_skew_s:
                raise PromptRecorderError(
                    f"wrist camera skew {skew_s:.6f}s exceeds "
                    f"{config.safety.max_camera_skew_s:.6f}s"
                )

            demonstrated = SensorimotorFrame(
                timestamp_s=index * expected_dt_s,
                images=frame.images,
                joint_position=frame.joint_position,
                executed_action=frame.joint_position,
                image_timestamps_s=frame.image_timestamps_s,
            )
            buffer.append(demonstrated)
            capture_started_s.append(capture_started)
            capture_lateness_s.append(capture_started - deadline_s)
            observation_latency_s.append(observation_latency)
            for key in PRIMARY_CAMERA_KEYS:
                camera_timestamps_s[key].append(current_timestamps[key])
            camera_age_s.extend(max(0.0, value) for value in current_ages)
            camera_skew_s.append(skew_s)

    capture_intervals_s = np.diff(
        np.asarray(capture_started_s, dtype=np.float64)
    )
    interval_errors_s = np.abs(capture_intervals_s - expected_dt_s)
    camera_fps = {
        key: _effective_fps(camera_timestamps_s[key])
        for key in PRIMARY_CAMERA_KEYS
    }
    camera_interval_max_relative_error = {
        key: float(
            np.max(
                np.abs(
                    np.diff(np.asarray(camera_timestamps_s[key]))
                    - expected_dt_s
                )
            )
            / expected_dt_s
        )
        for key in PRIMARY_CAMERA_KEYS
    }
    fps_floor = fps * (1.0 - FPS_RELATIVE_TOLERANCE)
    for key, measured_fps in camera_fps.items():
        if measured_fps is None or measured_fps < fps_floor:
            raise PromptRecorderError(
                f"{key} effective camera FPS is below {fps_floor:g}: "
                f"{measured_fps}"
            )

    capture_metadata: dict[str, Any] = {
        "duration_requested_s": float(duration_s),
        "duration_recorded_s": interval_count / fps,
        "fps": fps,
        "fps_tolerance": FPS_RELATIVE_TOLERANCE,
        "frame_count": frame_count,
        "camera_resolution": list(expected_resolution),
        "camera_fps_estimate_hz": camera_fps,
        "camera_interval_max_relative_error": (
            camera_interval_max_relative_error
        ),
        "capture_interval_max_error_s": float(np.max(interval_errors_s)),
        "capture_interval_max_relative_error": float(
            np.max(interval_errors_s) / expected_dt_s
        ),
        "capture_lateness_max_s": max(capture_lateness_s),
        "observation_latency_p95_s": _percentile(
            observation_latency_s, 95.0
        ),
        "observation_latency_max_s": max(observation_latency_s),
        "camera_frame_age_p95_s": _percentile(camera_age_s, 95.0),
        "camera_frame_age_max_s": max(camera_age_s),
        "camera_skew_p95_s": _percentile(camera_skew_s, 95.0),
        "camera_skew_max_s": max(camera_skew_s),
        "capture_timestamp_offsets_s": _relative(capture_started_s),
        "camera_timestamp_offsets_s": {
            key: _relative(camera_timestamps_s[key])
            for key in PRIMARY_CAMERA_KEYS
        },
        "joint_width": ACTION_DIM,
    }
    buffer.metadata.update(capture_metadata)
    draft = buffer.to_episode_data()
    prompt = physical_prompt_from_episode(
        draft,
        policy_hz=config.runtime.policy_hz,
    )
    buffer.metadata.update(
        {
            "policy_hz_verified": config.runtime.policy_hz,
            "policy_prompt_frame_count": len(prompt.frames),
            "prompt_fingerprint": prompt.fingerprint,
        }
    )
    data = buffer.to_episode_data()
    saved_npz, saved_manifest = save_episode(
        data,
        npz_path.parent,
        stem=stem,
    )
    loaded = load_episode(saved_npz, saved_manifest)
    replayed_prompt = physical_prompt_from_episode(
        loaded,
        policy_hz=config.runtime.policy_hz,
    )
    if replayed_prompt.fingerprint != prompt.fingerprint:
        raise PromptRecorderError(
            "physical-prompt replay fingerprint does not match the recording"
        )

    try:
        manifest: Mapping[str, Any] = json.loads(
            saved_manifest.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise PromptRecorderError(
            f"failed to read the verified prompt manifest: {error}"
        ) from error

    return {
        "schema_version": 1,
        "gate": "G9-prompt-record",
        "result": "recorded",
        "evidence_level": "real",
        "capture_mode": CAPTURE_MODE,
        "action_source": ACTION_SOURCE,
        "goal_position_commands_sent": 0,
        "torque_disabled_for_capture": True,
        "npz_path": str(saved_npz),
        "manifest_path": str(saved_manifest),
        "npz_sha256": manifest["npz_sha256"],
        "episode_fingerprint": loaded.fingerprint,
        "prompt_fingerprint": replayed_prompt.fingerprint,
        "round_trip_verified": True,
        "duration_s": replayed_prompt.duration_s,
        "recorded_frame_count": loaded.frame_count,
        "policy_prompt_frame_count": len(replayed_prompt.frames),
        "fps": loaded.fps,
        "policy_hz": config.runtime.policy_hz,
        "camera_keys": list(PRIMARY_CAMERA_KEYS),
        "core_policy_features": [
            "observation.images.left_wrist",
            "observation.images.right_wrist",
            "observation.state",
            "action",
        ],
        "head_camera_included": False,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "capture_interval_max_relative_error": capture_metadata[
            "capture_interval_max_relative_error"
        ],
        "camera_fps_estimate_hz": camera_fps,
        "camera_interval_max_relative_error": (
            camera_interval_max_relative_error
        ),
        "camera_frame_age_p95_s": capture_metadata[
            "camera_frame_age_p95_s"
        ],
        "camera_skew_p95_s": capture_metadata["camera_skew_p95_s"],
        "observation_latency_p95_s": capture_metadata[
            "observation_latency_p95_s"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record a checksum-verified, observation-only dual-SO-101 physical prompt."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        result = record_physical_prompt(
            ProjectConfig.load(args.config),
            output_dir=args.output_dir,
            task=args.task,
            duration_s=args.duration,
            episode_index=args.episode_index,
            task_index=args.task_index,
        )
    except (
        ConfigError,
        ContractError,
        DatasetError,
        HardwarePreflightError,
        LeRobotFactoryError,
        OSError,
        PromptRecorderError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_SOURCE",
    "CAPTURE_MODE",
    "PromptRecorderError",
    "record_physical_prompt",
]
