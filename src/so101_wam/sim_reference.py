"""Actuator-driven simulation references, not learned or zero-shot policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil, isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np

from .adapters.mujoco import MujocoBiSOAdapter, MujocoCollisionError
from .checkpoint import load_compact_wam_bundle
from .config import ProjectConfig
from .constants import ACTION_DIM
from .context import ContextSnapshot, Gen15Context
from .contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from .dataset import EpisodeBuffer, load_episode, physical_prompt_from_episode
from .deployment import file_sha256, project_config_sha256
from .mujoco_cli import _capture_home_prompt, _checkpoint_identity, write_mujoco_report
from .mujoco_semantic_benchmark import SemanticTask, score_object_position
from .policy import CompactWAMPolicy, PolicyError
from .rollout import RolloutError, SafetyRejectedError, run_managed_rollout
from .runtime import PolicyStep, SO101WAMRuntime, ServoStep, SnapshotPolicy


class ReferenceError(ValueError):
    """A reference cannot preserve its simulation-only evidence contract."""


class TrialKind(StrEnum):
    REFERENCE = "reference"
    HOLD = "hold"
    LEARNED = "learned"


INITIAL_OBJECT_ARM_TOLERANCE_M = 1e-6
ACTION_TIMING = "observation_then_command"
REFERENCE_SOURCE_KIND = "mujoco_reference"


class _VirtualClock:
    def __init__(self, start_s: float = 100.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s


@dataclass(frozen=True, slots=True)
class SimTrialConfig:
    config: ProjectConfig
    task: SemanticTask
    seed: int
    output_dir: Path
    kind: TrialKind
    duration_s: float
    episode_index: int
    times_s: Sequence[float] | None = None
    targets: np.ndarray | None = None
    checkpoint_path: Path | None = None
    prompt_path: Path | None = None
    prompt_manifest_path: Path | None = None
    device: str = "cpu"


class _WaypointPolicy:
    """Interpolate a frozen joint command trajectory at servo deadlines."""

    required_history_steps = 1

    def __init__(
        self,
        times_s: Sequence[float],
        targets: np.ndarray,
        *,
        horizon: int,
        servo_hz: float,
    ) -> None:
        times = np.array(times_s, dtype=np.float64, copy=True)
        values = np.array(targets, dtype=np.float32, copy=True)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.isfinite(times).all()
            or times[0] != 0.0
            or not (np.diff(times) > 0.0).all()
        ):
            raise ReferenceError("waypoint times must start at zero and increase")
        if values.shape != (len(times), ACTION_DIM) or not np.isfinite(values).all():
            raise ReferenceError("waypoints must contain finite 12-axis targets")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ReferenceError("horizon must be a positive integer")
        if not np.isfinite(servo_hz) or servo_hz <= 0:
            raise ReferenceError("servo_hz must be finite and positive")
        self._times = times
        self._targets = values
        self._horizon = horizon
        self._servo_hz = servo_hz
        self._start_s: float | None = None

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        del snapshot
        if self._start_s is None:
            self._start_s = now_s
        sample_times = now_s - self._start_s + np.arange(self._horizon) / self._servo_hz
        targets = np.stack(
            [np.interp(sample_times, self._times, self._targets[:, axis]) for axis in range(ACTION_DIM)],
            axis=-1,
        ).astype(np.float32)
        return ActionChunk(
            target_joint_position=targets,
            dt_s=1.0 / self._servo_hz,
            created_at_s=now_s,
        )


@dataclass(slots=True)
class _HoldPolicy:
    target: np.ndarray
    horizon: int
    servo_hz: float
    required_history_steps: int = 1

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        del snapshot
        targets = np.repeat(self.target[None, :], self.horizon, axis=0).astype(
            np.float32
        )
        return ActionChunk(
            target_joint_position=targets,
            dt_s=1.0 / self.servo_hz,
            created_at_s=now_s,
        )


@dataclass(slots=True)
class _CroppedPromptPolicy:
    base: CompactWAMPolicy
    prompt: PhysicalPrompt
    policy_hz: float
    total_duration_s: float

    @property
    def required_history_steps(self) -> int:
        return max(1, int(self.base.required_history_steps))

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        live_count = int(self.base.required_history_steps)
        cropped = Gen15Context(
            self.prompt,
            policy_hz=self.policy_hz,
            total_duration_s=self.total_duration_s,
        )
        cropped.extend_live(tuple(snapshot.live_frames[-live_count:]))
        return self.base.predict(cropped.snapshot(), now_s=now_s)


@dataclass(slots=True)
class _TrialObserver:
    config: ProjectConfig
    task: SemanticTask
    buffer: EpisodeBuffer
    object_positions: list[dict[str, object]]
    contacts: list[dict[str, object]]
    sent_steps: int = 0
    initial_previous_action: list[float] | None = None
    max_command_observation_delta: float | None = None

    def on_policy_step(self, *, policy_index: int, step: PolicyStep) -> None:
        del policy_index, step

    def on_servo_step(
        self,
        *,
        policy_index: int,
        servo_index: int,
        step: ServoStep,
    ) -> None:
        del policy_index, servo_index
        if not step.sent:
            return
        timestamp_s = self.sent_steps / float(self.config.runtime.servo_hz)
        observed = step.observation
        if self.initial_previous_action is None:
            self.initial_previous_action = [
                float(value) for value in observed.joint_position
            ]
            self.buffer.metadata["initial_previous_action"] = self.initial_previous_action
        delta = np.max(
            np.abs(
                np.asarray(step.executed_action, dtype=np.float32)
                - np.asarray(observed.joint_position, dtype=np.float32)
            )
        )
        self.max_command_observation_delta = max(
            float(delta),
            self.max_command_observation_delta or 0.0,
        )
        self.buffer.append(
            SensorimotorFrame(
                timestamp_s=timestamp_s,
                images=observed.images,
                joint_position=observed.joint_position,
                executed_action=step.executed_action,
                image_timestamps_s={
                    key: timestamp_s for key in observed.images
                },
            )
        )
        self.sent_steps += 1


@dataclass(slots=True)
class _AdapterObserver:
    base: _TrialObserver
    adapter: MujocoBiSOAdapter

    def on_policy_step(self, *, policy_index: int, step: PolicyStep) -> None:
        self.base.on_policy_step(policy_index=policy_index, step=step)

    def on_servo_step(
        self,
        *,
        policy_index: int,
        servo_index: int,
        step: ServoStep,
    ) -> None:
        for contact in self.adapter.contacts():
            self.base.contacts.append(_contact_payload(contact))
        self.base.object_positions.append(
            _object_sample(self.base.task, self.adapter, self.base.sent_steps)
        )
        self.base.on_servo_step(
            policy_index=policy_index,
            servo_index=servo_index,
            step=step,
        )


def run_sim_trial(trial: SimTrialConfig) -> dict[str, Any]:
    """Record one MuJoCo object trial without claiming hardware or zero-shot evidence."""

    _validate_config(trial)
    report_path, episode_stem = _artifact_targets(trial)
    episode_npz = trial.output_dir / f"{episode_stem}.npz"
    episode_json = trial.output_dir / f"{episode_stem}.json"
    if report_path.exists() or episode_npz.exists() or episode_json.exists():
        raise ReferenceError(f"trial artifact already exists: {report_path}")

    adapter = _make_adapter(trial)
    penetration, initial_identity = _initial_penetration(adapter)
    if penetration is not None:
        report = _failure_report(
            trial,
            reason="initial_object_arm_penetration",
            evidence={"contacts": [penetration]},
            reference_success=False if trial.kind is TrialKind.REFERENCE else None,
            model_identity=initial_identity,
        )
        write_mujoco_report(report_path, report)
        return report

    prompt, policy, checkpoint_meta = _policy_inputs(trial, adapter)
    buffer = EpisodeBuffer(
        fps=trial.config.runtime.servo_hz,
        task=trial.task.label,
        episode_index=trial.episode_index,
        metadata={
            "source_kind": _source_kind(trial.kind),
            "action_timing": ACTION_TIMING,
            "success_label_available": False,
            "real_output": False,
            "safety_real_output": False,
        },
    )
    observer = _TrialObserver(trial.config, trial.task, buffer, [], [])
    terminal_object: tuple[float, float, float] | None = None
    terminal_identity: dict[str, object] | None = initial_identity
    object_profile: dict[str, object] | None = None
    object_profile_sha256: str | None = None

    def terminal(timestamp_s: float) -> None:
        del timestamp_s
        nonlocal terminal_object, terminal_identity
        nonlocal object_profile, object_profile_sha256
        terminal_object = adapter.object_body_position(trial.task.object_body)
        terminal_identity = adapter.model_identity()
        object_profile = adapter.object_physical_profile(trial.task.object_body)
        object_profile_sha256 = adapter.object_physical_profile_sha256(
            trial.task.object_body
        )

    try:
        runtime = SO101WAMRuntime(
            config=trial.config,
            prompt=prompt,
            robot=adapter,
            policy=cast(SnapshotPolicy, policy),
        )
        clock = _VirtualClock()
        summary = run_managed_rollout(
            runtime,
            adapter,
            policy_steps=_policy_steps(trial),
            calibrate_on_connect=False,
            clock=clock,
            sleeper=clock.sleep,
            terminal_observer=terminal,
            rollout_observer=_AdapterObserver(observer, adapter),
        )
    except (MujocoCollisionError, SafetyRejectedError, PolicyError, RolloutError) as error:
        report = _failure_report(
            trial,
            reason=_failure_reason(error),
            evidence={"error_type": type(error).__name__, "message": str(error)},
            reference_success=False if trial.kind is TrialKind.REFERENCE else None,
            model_identity=terminal_identity,
        )
        _attach_partial_episode(trial, report, observer, episode_stem)
        write_mujoco_report(report_path, report)
        return report

    if terminal_object is None:
        raise ReferenceError("trial did not capture terminal object position")
    outcome = score_object_position(trial.task, terminal_object)
    buffer.metadata["success_label_available"] = True
    buffer.metadata["object_success"] = outcome.success
    buffer.metadata["reference_success"] = (
        outcome.success if trial.kind is TrialKind.REFERENCE else None
    )
    npz_path: Path | None
    manifest_path: Path | None
    if observer.sent_steps:
        npz_path, manifest_path = buffer.save(trial.output_dir, stem=episode_stem)
    else:
        npz_path = None
        manifest_path = None
    report = _success_report(
        trial,
        summary=summary,
        terminal_object=terminal_object,
        observer=observer,
        checkpoint_meta=checkpoint_meta,
        npz_path=npz_path,
        manifest_path=manifest_path,
        reference_success=outcome.success if trial.kind is TrialKind.REFERENCE else None,
        model_identity=terminal_identity,
        object_profile=object_profile,
        object_profile_sha256=object_profile_sha256,
        object_position_error_m=outcome.object_position_error_m,
        failure_reason=outcome.failure_reason,
    )
    write_mujoco_report(report_path, report)
    return report


def _validate_config(trial: SimTrialConfig) -> None:
    if not isinstance(trial.seed, int) or isinstance(trial.seed, bool):
        raise ReferenceError("seed must be an integer")
    if trial.episode_index < 0:
        raise ReferenceError("episode_index must be non-negative")
    rates = (
        trial.config.runtime.camera_hz,
        trial.config.runtime.policy_hz,
        trial.config.runtime.servo_hz,
        trial.duration_s,
    )
    if any(not isfinite(float(value)) or float(value) <= 0.0 for value in rates):
        raise ReferenceError("trial rates and duration must be finite and positive")
    if _duration_policy_steps(trial) != trial.task.policy_steps:
        raise ReferenceError("duration_s must agree with task.policy_steps")
    if trial.config.runtime.backend != "mujoco":
        raise ReferenceError("runtime.backend must be 'mujoco'")
    if not trial.config.runtime.actuation_enabled:
        raise ReferenceError("runtime.actuation_enabled must be true")
    if not trial.config.mujoco.forbid_collisions:
        raise ReferenceError("mujoco.forbid_collisions must be true")
    _validate_kind_inputs(trial)


def _validate_kind_inputs(trial: SimTrialConfig) -> None:
    has_waypoints = trial.times_s is not None or trial.targets is not None
    has_checkpoint = trial.checkpoint_path is not None
    has_prompt = trial.prompt_path is not None or trial.prompt_manifest_path is not None
    if trial.kind is TrialKind.REFERENCE:
        if trial.times_s is None or trial.targets is None:
            raise ReferenceError("kind inputs: reference requires waypoints")
        if has_checkpoint or has_prompt:
            raise ReferenceError("kind inputs: reference forbids checkpoint and prompt")
        return
    if trial.kind is TrialKind.HOLD:
        if has_waypoints or has_checkpoint or has_prompt:
            raise ReferenceError("kind inputs: hold forbids waypoints, checkpoint, and prompt")
        return
    if trial.kind is TrialKind.LEARNED:
        if trial.checkpoint_path is None or trial.prompt_path is None:
            raise ReferenceError("kind inputs: learned requires checkpoint_path and prompt_path")
        if trial.prompt_manifest_path is not None and trial.prompt_path is None:
            raise ReferenceError("kind inputs: learned manifest requires prompt_path")
        if has_waypoints:
            raise ReferenceError("kind inputs: learned forbids waypoints")
        return
    raise ReferenceError("kind inputs: unknown trial kind")


def _artifact_targets(trial: SimTrialConfig) -> tuple[Path, str]:
    stem = f"{trial.kind.value}_{trial.task.task_id}_seed-{trial.seed}_ep-{trial.episode_index:06d}"
    return trial.output_dir / f"{stem}.json", f"{stem}_episode"


def _make_adapter(trial: SimTrialConfig) -> MujocoBiSOAdapter:
    return MujocoBiSOAdapter(
        config=trial.config.mujoco,
        servo_hz=trial.config.runtime.servo_hz,
        actuation_enabled=trial.config.runtime.actuation_enabled,
        include_head_camera=trial.config.runtime.use_head_camera,
        semantic_object_body=trial.task.object_body,
        initial_object_position=trial.task.initial_position(trial.seed),
    )


def _initial_penetration(
    adapter: MujocoBiSOAdapter,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    adapter.connect(calibrate=False)
    try:
        model_identity = adapter.model_identity()
        for contact in adapter.contacts():
            payload = _contact_payload(contact)
            categories = {payload["category1"], payload["category2"]}
            if categories != {"object", "left_arm"} and categories != {"object", "right_arm"}:
                continue
            if float(cast(Any, payload["distance"])) < -INITIAL_OBJECT_ARM_TOLERANCE_M:
                return payload, model_identity
    finally:
        adapter.disconnect()
    return None, model_identity


def _policy_inputs(
    trial: SimTrialConfig,
    adapter: MujocoBiSOAdapter,
) -> tuple[PhysicalPrompt, SnapshotPolicy, dict[str, str | None]]:
    if trial.kind is TrialKind.REFERENCE:
        assert trial.times_s is not None and trial.targets is not None
        return (
            _capture_home_prompt(adapter),
            _WaypointPolicy(
                trial.times_s,
                trial.targets,
                horizon=trial.config.runtime.action_horizon,
                servo_hz=trial.config.runtime.servo_hz,
            ),
            {"checkpoint_sha256": None, "checkpoint_id": None, "prompt_sha256": None},
        )
    if trial.kind is TrialKind.HOLD:
        return (
            _capture_home_prompt(adapter),
            _HoldPolicy(
                np.asarray(trial.config.mujoco.home_joint_position, dtype=np.float32),
                horizon=trial.config.runtime.action_horizon,
                servo_hz=trial.config.runtime.servo_hz,
            ),
            {"checkpoint_sha256": None, "checkpoint_id": None, "prompt_sha256": None},
        )

    assert trial.checkpoint_path is not None and trial.prompt_path is not None
    bundle = load_compact_wam_bundle(trial.checkpoint_path, device=trial.device)
    checkpoint_sha256, checkpoint_id = _checkpoint_identity(
        trial.checkpoint_path,
        bundle.metadata,
    )
    episode = load_episode(trial.prompt_path, trial.prompt_manifest_path)
    prompt = physical_prompt_from_episode(
        episode,
        policy_hz=trial.config.runtime.policy_hz,
    )
    base = CompactWAMPolicy(
        bundle.model,
        servo_hz=trial.config.runtime.servo_hz,
        device=trial.device,
        joint_lower=trial.config.safety.joint_lower,
        joint_upper=trial.config.safety.joint_upper,
    )
    return (
        prompt,
        cast(
            SnapshotPolicy,
            _CroppedPromptPolicy(
                base,
                prompt,
                policy_hz=trial.config.runtime.policy_hz,
                total_duration_s=trial.config.runtime.context_seconds,
            ),
        ),
        {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_id": checkpoint_id,
            "prompt_sha256": file_sha256(trial.prompt_path),
        },
    )


def _policy_steps(trial: SimTrialConfig) -> int:
    return trial.task.policy_steps


def _duration_policy_steps(trial: SimTrialConfig) -> int:
    return max(1, int(ceil(trial.duration_s * trial.config.runtime.policy_hz)))


def _source_kind(kind: TrialKind) -> str:
    if kind is TrialKind.REFERENCE:
        return REFERENCE_SOURCE_KIND
    return f"mujoco_{kind.value}"


def _object_sample(
    task: SemanticTask,
    adapter: MujocoBiSOAdapter,
    index: int,
) -> dict[str, object]:
    return {
        "servo_sample_index": index,
        "body": task.object_body,
        "position": list(adapter.object_body_position(task.object_body)),
    }


def _contact_payload(contact: object) -> dict[str, object]:
    if hasattr(contact, "__dataclass_fields__"):
        return dict(asdict(cast(Any, contact)))
    fields = (
        "geom1",
        "geom2",
        "body1",
        "body2",
        "category1",
        "category2",
        "distance",
        "forbidden",
    )
    return {
        field: getattr(contact, field)
        for field in fields
        if hasattr(contact, field)
    }


def _summary_payload(summary: object) -> dict[str, object]:
    if hasattr(summary, "__dataclass_fields__"):
        return dict(asdict(cast(Any, summary)))
    return dict(vars(summary))


def _base_report(trial: SimTrialConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "so101_wam.sim_trial",
        "evidence_level": "simulation",
        "real_output": False,
        "safety_real_output": False,
        "kind": trial.kind.value,
        "source_kind": _source_kind(trial.kind),
        "task_id": trial.task.task_id,
        "task_label": trial.task.label,
        "seed": trial.seed,
        "episode_index": trial.episode_index,
        "config_sha256": project_config_sha256(trial.config),
        "initial_object_position": list(trial.task.initial_position(trial.seed)),
        "target_object_position": list(trial.task.target_object_position),
        "action_timing": ACTION_TIMING,
        "attempts": 1,
    }


def _failure_report(
    trial: SimTrialConfig,
    *,
    reason: str,
    evidence: dict[str, object],
    reference_success: bool | None = None,
    model_identity: dict[str, object] | None = None,
) -> dict[str, Any]:
    return {
        **_base_report(trial),
        "status": "execution_failure",
        "result": "fail",
        "object_success": False,
        "command_count": 0,
        "failure_reason": reason,
        "failure_evidence": evidence,
        "reference_success": reference_success,
        "mujoco_model_identity": model_identity,
    }


def _success_report(
    trial: SimTrialConfig,
    *,
    summary: object,
    terminal_object: tuple[float, float, float],
    observer: _TrialObserver,
    checkpoint_meta: dict[str, str | None],
    npz_path: Path | None,
    manifest_path: Path | None,
    reference_success: bool | None,
    model_identity: dict[str, object] | None,
    object_profile: dict[str, object] | None,
    object_profile_sha256: str | None,
    object_position_error_m: float,
    failure_reason: str | None,
) -> dict[str, Any]:
    report = {
        **_base_report(trial),
        "status": "scored",
        "result": "pass",
        "object_success": failure_reason is None,
        "command_count": observer.sent_steps,
        "initial_previous_action": observer.initial_previous_action,
        "max_command_observation_delta": observer.max_command_observation_delta,
        "reference_success": reference_success,
        "object_position_error_m": object_position_error_m,
        "failure_reason": failure_reason,
        "terminal_object_position": list(terminal_object),
        "mujoco_model_identity": model_identity,
        "object_physical_profile": object_profile,
        "object_physical_profile_sha256": object_profile_sha256,
        "rollout": _summary_payload(summary),
        "object_position_trajectory": observer.object_positions,
        "contacts": observer.contacts,
        **checkpoint_meta,
    }
    if npz_path is None or manifest_path is None:
        return report
    report["episode"] = {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "npz_sha256": file_sha256(npz_path),
        "manifest_sha256": file_sha256(manifest_path),
        "frame_count": observer.sent_steps,
    }
    return report


def _attach_partial_episode(
    trial: SimTrialConfig,
    report: dict[str, Any],
    observer: _TrialObserver,
    episode_stem: str,
) -> None:
    report["object_position_trajectory"] = observer.object_positions
    report["contacts"] = observer.contacts
    report["command_count"] = observer.sent_steps
    report["initial_previous_action"] = observer.initial_previous_action
    report["max_command_observation_delta"] = observer.max_command_observation_delta
    if not observer.sent_steps:
        return
    observer.buffer.metadata["object_success"] = False
    observer.buffer.metadata["reference_success"] = (
        False if trial.kind is TrialKind.REFERENCE else None
    )
    npz_path, manifest_path = observer.buffer.save(trial.output_dir, stem=episode_stem)
    report["episode"] = {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "npz_sha256": file_sha256(npz_path),
        "manifest_sha256": file_sha256(manifest_path),
        "frame_count": observer.sent_steps,
        "partial": True,
    }


def _failure_reason(error: Exception) -> str:
    if isinstance(error, MujocoCollisionError):
        return "mujoco_collision"
    if isinstance(error, SafetyRejectedError):
        return "safety_rejection"
    if isinstance(error, PolicyError):
        return "policy_model_error"
    return "rollout_error"
