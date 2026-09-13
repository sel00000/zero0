"""Record leader-follower demonstrations with confirmed command labels."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from time import monotonic, perf_counter, sleep
from typing import Any

import numpy as np

from .adapters.lerobot import LeRobotBiSOAdapter, LeRobotBiSOLeader
from .config import LeRobotConfig, LeRobotLeaderConfig, ProjectConfig
from .constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from .contracts import ActionChunk, SensorimotorFrame
from .dataset import (
    DatasetArtifactExistsError,
    EpisodeBuffer,
    FPS_RELATIVE_TOLERANCE,
    save_episode,
)
from .lerobot_factory import create_bi_so_follower, create_bi_so_leader
from .safety import SafetySupervisor

_DEFAULT_DURATION_S = 3.0
_TIME_EPSILON_S = 1e-9


class DemoRecorderError(RuntimeError):
    """Raised when a demonstration cannot be recorded faithfully."""


def _time(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not isfinite(value) or value < 0:
        raise DemoRecorderError("recording clock must be finite and non-negative")
    return value


def _validate_request(
    config: ProjectConfig,
    task: str,
    duration_s: float,
    episode_index: int,
    task_index: int,
) -> tuple[float, int]:
    if config.runtime.backend != "lerobot" or not config.runtime.actuation_enabled:
        raise DemoRecorderError("record-demo requires the lerobot backend with actuation_enabled=true")
    config.lerobot.require_actuation_identity()
    config.leader.require_hardware_session()
    if not config.safety.calibrated:
        raise DemoRecorderError("record-demo requires measured safety calibration")
    ports = (
        config.lerobot.left_port, config.lerobot.right_port,
        config.leader.left_port, config.leader.right_port,
    )
    if len({Path(port).resolve() for port in ports}) != len(ports):
        raise DemoRecorderError("all four leader/follower serial ports must be distinct")
    if (
        config.leader.robot_id == config.lerobot.robot_id
        and Path(config.leader.calibration_dir).resolve()
        == Path(config.lerobot.calibration_dir).resolve()
    ):
        raise DemoRecorderError("leader and follower must use separate calibration identities")
    if not isinstance(task, str) or not task.strip():
        raise DemoRecorderError("task must be a non-empty string")
    for name, value in (("episode_index", episode_index), ("task_index", task_index)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DemoRecorderError(f"{name} must be a non-negative integer")
    fps = float(config.runtime.camera_hz)
    if not isfinite(fps) or not 0 < fps <= config.runtime.servo_hz:
        raise DemoRecorderError("camera_hz must be positive and no higher than servo_hz")
    if not isfinite(duration_s) or duration_s <= 0:
        raise DemoRecorderError("duration_s must be finite and positive")
    intervals = round(duration_s * fps)
    if intervals < 1 or abs(intervals / fps - duration_s) > _TIME_EPSILON_S:
        raise DemoRecorderError(f"duration_s must align to the {fps:g} Hz capture grid")
    return fps, intervals


def _check_home(
    frame: SensorimotorFrame, requested: np.ndarray, config: LeRobotConfig,
) -> None:
    home = np.asarray(config.home_joint_position, dtype=np.float64)
    tolerance = np.asarray(config.home_joint_tolerance, dtype=np.float64)
    if np.any(np.abs(frame.joint_position - home) > tolerance):
        raise DemoRecorderError("follower is outside measured home tolerance")
    if np.any(np.abs(requested - frame.joint_position) > tolerance):
        raise DemoRecorderError("leader must be aligned with the follower before recording")


def _check_cameras(
    frame: SensorimotorFrame,
    previous: dict[str, float],
    config: ProjectConfig,
    camera_now: float,
) -> dict[str, float]:
    resolution = (config.lerobot.camera_height, config.lerobot.camera_width)
    if frame.primary_images[0].shape[:2] != resolution:
        raise DemoRecorderError(f"wrist camera resolution must be {resolution}")
    if frame.image_timestamps_s is None:
        raise DemoRecorderError("wrist camera timestamps are required")
    current = dict(frame.image_timestamps_s)
    for key in PRIMARY_CAMERA_KEYS:
        age = camera_now - current[key]
        if age < -1e-6 or age > config.safety.max_observation_age_s:
            raise DemoRecorderError(f"{key} camera timestamp is stale or in the future")
        if key in previous and current[key] <= previous[key]:
            raise DemoRecorderError(f"{key} camera timestamp did not advance")
    return current


def record_teleop_demo(
    config: ProjectConfig,
    *,
    output_dir: str | Path,
    task: str,
    duration_s: float = _DEFAULT_DURATION_S,
    episode_index: int = 0,
    task_index: int = 0,
    follower_factory: Callable[[LeRobotConfig], Any] | None = None,
    leader_factory: Callable[[LeRobotLeaderConfig], Any] | None = None,
    clock: Callable[[], float] = monotonic,
    camera_clock: Callable[[], float] = perf_counter,
    sleeper: Callable[[float], None] = sleep,
) -> dict[str, Any]:
    """Pair each pre-command observation with LeRobot's reported sent target.

    One command is sent per camera-rate row. The existing per-servo-tick
    displacement bound also limits each of these slower commands. Full episodes
    may exceed the separate 3–12 second physical-prompt contract.
    """
    fps, intervals = _validate_request(config, task, duration_s, episode_index, task_index)
    directory = Path(output_dir)
    stem = f"demo_{episode_index:06d}"
    if any((directory / f"{stem}{suffix}").exists() for suffix in (".npz", ".json")):
        raise DatasetArtifactExistsError("demo artifact already exists; choose a new episode index")
    directory.mkdir(parents=True, exist_ok=True)
    supervisor = SafetySupervisor(config.safety)
    follower = LeRobotBiSOAdapter(
        (follower_factory or create_bi_so_follower)(config.lerobot),
        config=config.lerobot,
        actuation_enabled=True,
        max_camera_age_s=config.safety.max_observation_age_s,
        camera_clock=camera_clock,
    )
    leader = LeRobotBiSOLeader((leader_factory or create_bi_so_leader)(config.leader))
    injected = follower_factory is not None or leader_factory is not None
    metadata: dict[str, Any] = {
        "capture_mode": "leader_follower_teleoperation",
        "action_source": "lerobot_send_action_return",
        "action_interpretation": "target reported sent by the driver; not measured joint motion",
        "evidence_level": "injected_devices_unverified" if injected else "hardware_recording",
        "task_success": None,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "leader_id": config.leader.robot_id,
        "leader_calibration_id": config.leader.calibration_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_hz": fps,
        "camera_hz": config.runtime.camera_hz,
        "requested_action": [],
        "commanded_action": [],
    }
    timing: dict[str, Any] = {
        "clock": "monotonic",
        "camera_clock": "perf_counter",
        "camera_timestamp_source": "atomic OpenCVCamera frame/timestamp buffer snapshot",
        "observation_started_s": [], "observation_finished_s": [],
        "leader_read_started_s": [], "leader_read_finished_s": [],
        "send_started_s": [], "send_finished_s": [],
        "camera_timestamp_s": {key: [] for key in PRIMARY_CAMERA_KEYS},
    }
    buffer = EpisodeBuffer(fps=fps, task=task.strip(), episode_index=episode_index, task_index=task_index)
    previous_cameras: dict[str, float] = {}
    previous_start: float | None = None
    dt = 1.0 / fps

    # Register cleanup before connecting so interrupts and partial connections
    # still close both devices. No episode is published until cleanup succeeds.
    with ExitStack() as resources:
        resources.callback(follower.disconnect)
        follower.connect(calibrate=False)
        resources.callback(leader.disconnect)
        leader.connect()
        origin = _time(clock)
        timing["origin_s"] = origin
        timing["camera_origin_s"] = _time(camera_clock)
        for index in range(intervals + 1):
            deadline = origin + index * dt
            now = _time(clock)
            if now < deadline:
                sleeper(deadline - now)
            started = _time(clock)
            if started + _TIME_EPSILON_S < deadline:
                raise DemoRecorderError("clock did not advance to the capture deadline")
            if previous_start is not None and abs(started - previous_start - dt) > dt * FPS_RELATIVE_TOLERANCE:
                raise DemoRecorderError("capture cadence exceeds dataset FPS tolerance")
            frame = follower.get_observation(timestamp_s=started)
            observed = _time(clock)
            leader_started = _time(clock)
            requested = leader.get_action()
            leader_finished = _time(clock)
            read_times = (started, observed, leader_started, leader_finished)
            if any(after < before for before, after in zip(read_times, read_times[1:])):
                raise DemoRecorderError("recording clock moved backwards before transmission")
            if index == 0:
                _check_home(frame, requested, config.lerobot)
            chunk = ActionChunk(requested.reshape(1, ACTION_DIM), dt_s=dt, created_at_s=leader_started)
            decision = supervisor.evaluate(frame, chunk, now_s=_time(clock))
            if not decision.accepted:
                raise DemoRecorderError(f"teleoperation safety rejection: {decision.reasons}")
            # Safety evaluation also takes time; check camera freshness at the
            # transmission boundary, after that work has finished.
            current_cameras = _check_cameras(frame, previous_cameras, config, _time(camera_clock))
            send_started = _time(clock)
            if send_started < leader_finished:
                raise DemoRecorderError("recording clock moved backwards before transmission")
            if (
                send_started - started > config.safety.max_observation_age_s
                or send_started - leader_started > config.safety.watchdog_timeout_s
            ):
                raise DemoRecorderError("observation or leader command expired before transmission")
            reported = follower.send_action(decision.action)
            send_finished = _time(clock)
            events = (started, observed, leader_started, leader_finished, send_started, send_finished)
            if any(after < before for before, after in zip(events, events[1:])):
                raise DemoRecorderError("recording clock moved backwards")
            if send_finished - started > dt:
                raise DemoRecorderError("device latency exceeded the recording interval")
            if reported is None:
                raise DemoRecorderError("send_action returned no confirmed command; refusing an inferred label")
            receipt = ActionChunk(reported.reshape(1, ACTION_DIM), dt_s=dt, created_at_s=send_started)
            checked = supervisor.evaluate(frame, receipt, now_s=send_finished)
            if not checked.accepted or checked.clipped:
                raise DemoRecorderError("driver reported a command outside the permitted bounds")

            buffer.append(replace(frame, timestamp_s=started - origin, executed_action=reported))
            metadata["requested_action"].append(requested.tolist())
            metadata["commanded_action"].append(decision.action.target_joint_position[0].tolist())
            for key, value in zip(
                ("observation_started_s", "observation_finished_s", "leader_read_started_s",
                 "leader_read_finished_s", "send_started_s", "send_finished_s"),
                events, strict=True,
            ):
                timing[key].append(value)
            for key in PRIMARY_CAMERA_KEYS:
                timing["camera_timestamp_s"][key].append(current_cameras[key])
            previous_start, previous_cameras = started, current_cameras

    metadata.update(timing=timing, confirmed_command_count=intervals + 1, duration_requested_s=duration_s)
    buffer.metadata.update(metadata)
    data = buffer.to_episode_data()
    npz_path, manifest_path = save_episode(data, directory, stem=stem)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "result": "recorded",
        "npz_path": str(npz_path), "manifest_path": str(manifest_path),
        "npz_sha256": manifest["npz_sha256"],
        "round_trip_verified": True,
        "recorded_frame_count": data.frame_count,
        "duration_s": float(data.timestamps_s[-1]),
        "fps": fps,
        "action_source": metadata["action_source"],
        "evidence_level": metadata["evidence_level"],
        "task_success": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record dual-SO-101 teleoperation with confirmed sent commands.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--duration", type=float, default=_DEFAULT_DURATION_S)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = record_teleop_demo(
            ProjectConfig.load(args.config), output_dir=args.output_dir,
            task=args.task, duration_s=args.duration,
            episode_index=args.episode_index, task_index=args.task_index,
        )
    except KeyboardInterrupt:
        parser.exit(130, "Recording interrupted; no completed episode was published.\n")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
