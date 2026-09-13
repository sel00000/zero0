"""Headless MuJoCo runner for the dual-SO-101 WAM embodiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Sequence, cast

import numpy as np

from .adapters.mujoco import MujocoAdapterError, MujocoBiSOAdapter
from .checkpoint import CheckpointError, load_compact_wam_bundle
from .config import ConfigError, DEFAULT_MUJOCO_CONFIG_PATH, ProjectConfig
from .constants import PRIMARY_CAMERA_KEYS
from .context import ContextSnapshot
from .contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from .dataset import DatasetError, load_episode, physical_prompt_from_episode
from .deployment import (
    DeploymentCertificationError,
    file_sha256,
    project_config_sha256,
)
from .policy import (
    CompactWAMPolicy,
    FutureLatentTelemetry,
    FutureLatentTelemetryMode,
    PolicyError,
)
from .rollout import RolloutError, RolloutObserver, RolloutSummary, run_managed_rollout
from .runtime import (
    PolicyStep,
    RuntimeErrorState,
    SO101WAMRuntime,
    ServoStep,
    SnapshotPolicy,
)
from .task_specs import TaskSpec, task_spec_kind


class MujocoCLIError(ValueError):
    """Raised when simulator CLI artifacts or configuration are inconsistent."""


class _VirtualClock:
    def __init__(self, start_s: float = 100.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s


class _WristRollSmokePolicy:
    """Small bounded motion used only to verify the simulator/runtime path."""

    required_history_steps = 1

    def __init__(self, *, horizon: int, servo_hz: float) -> None:
        self.horizon = horizon
        self.dt_s = 1.0 / servo_hz
        self.calls = 0

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        self.calls += 1
        current = snapshot.live_frames[-1].joint_position
        direction = 1.0 if self.calls % 2 else -1.0
        targets = np.repeat(current[None, :], self.horizon, axis=0).astype(np.float32)
        increments = direction * 0.5 * np.arange(1, self.horizon + 1, dtype=np.float32)
        targets[:, 4] = increments
        targets[:, 10] = -increments
        return ActionChunk(
            target_joint_position=targets, dt_s=self.dt_s, created_at_s=now_s
        )


@dataclass(slots=True)
class _TaskSpecSnapshotPolicy:
    """Keep the runtime snapshot boundary while excluding its physical prompt."""

    base: CompactWAMPolicy
    task_spec: TaskSpec

    @property
    def required_history_steps(self) -> int:
        return self.base.required_history_steps

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        return self.base.predict_task(
            self.task_spec,
            snapshot.live_frames,
            now_s=now_s,
        )

    def take_future_latent_telemetry(self) -> FutureLatentTelemetry | None:
        return self.base.take_future_latent_telemetry()


@dataclass(slots=True)
class _MujocoRolloutObserver:
    """Attach post-servo MuJoCo contacts to an existing rollout observer."""

    base: RolloutObserver
    adapter: MujocoBiSOAdapter
    telemetry_policy: object | None = None

    def on_policy_step(self, *, policy_index: int, step: PolicyStep) -> None:
        self.base.on_policy_step(policy_index=policy_index, step=step)
        recorder = getattr(self.base, "on_future_latent_step", None)
        take_telemetry = getattr(
            self.telemetry_policy,
            "take_future_latent_telemetry",
            None,
        )
        if recorder is None or take_telemetry is None:
            return
        telemetry = take_telemetry()
        if telemetry is None:
            raise MujocoCLIError("future latent telemetry is missing after policy step")
        recorder(
            policy_index=policy_index,
            observation_timestamp_s=telemetry.observation_timestamp_s,
            future_latents=telemetry.future_latents,
            observed_latent=telemetry.observed_latent,
        )

    def on_servo_step(
        self,
        *,
        policy_index: int,
        servo_index: int,
        step: ServoStep,
    ) -> None:
        self.base.on_servo_step(
            policy_index=policy_index,
            servo_index=servo_index,
            step=step,
        )
        contact_observer = getattr(self.base, "on_contact_sample", None)
        if contact_observer is None:
            return
        contact_observer(
            policy_index=policy_index,
            servo_index=servo_index,
            phase="servo",
            contacts=tuple(asdict(contact) for contact in self.adapter.contacts()),
        )


def _prompt_frame(frame: SensorimotorFrame, *, timestamp_s: float) -> SensorimotorFrame:
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images=frame.images,
        joint_position=frame.joint_position,
        executed_action=frame.joint_position,
    )


def _capture_home_prompt(adapter: MujocoBiSOAdapter) -> PhysicalPrompt:
    """Capture a deterministic three-second measured-hold prompt in simulation."""

    adapter.connect(calibrate=False)
    try:
        first = adapter.get_observation(timestamp_s=0.0)
        last = adapter.get_observation(timestamp_s=3.0)
    finally:
        adapter.disconnect()
    return PhysicalPrompt(
        (_prompt_frame(first, timestamp_s=0.0), _prompt_frame(last, timestamp_s=3.0))
    )


def _require_mujoco_backend(config: ProjectConfig) -> None:
    if config.runtime.backend != "mujoco":
        raise MujocoCLIError("MuJoCo runner requires runtime.backend='mujoco'")


def _adapter(
    config: ProjectConfig,
    *,
    semantic_object_body: str | None = None,
    initial_object_position: tuple[float, float, float] | None = None,
    initial_task_block_position: tuple[float, float, float] | None = None,
) -> MujocoBiSOAdapter:
    if initial_object_position is not None and initial_task_block_position is not None:
        raise MujocoCLIError(
            "initial_object_position and initial_task_block_position are exclusive"
        )
    object_position = (
        initial_task_block_position
        if initial_task_block_position is not None
        else initial_object_position
    )
    return MujocoBiSOAdapter(
        config=config.mujoco,
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=config.runtime.actuation_enabled,
        include_head_camera=config.runtime.use_head_camera,
        semantic_object_body=semantic_object_body,
        initial_object_position=object_position,
    )


def inspect_mujoco_object_profile(
    config: ProjectConfig,
    *,
    body_name: str,
) -> tuple[dict[str, object], str]:
    """Read one compiled semantic-object profile without running a policy."""

    profile, profile_sha256, _ = inspect_mujoco_scene(config, body_name=body_name)
    return profile, profile_sha256


def inspect_mujoco_scene(
    config: ProjectConfig,
    *,
    body_name: str,
) -> tuple[dict[str, object], str, dict[str, object]]:
    """Inspect expected scene identity without claiming a completed trial."""

    _require_mujoco_backend(config)
    adapter = _adapter(config, semantic_object_body=body_name)
    adapter.connect(calibrate=False)
    try:
        profile = adapter.object_physical_profile(body_name)
        profile_sha256 = adapter.object_physical_profile_sha256(body_name)
        identity = adapter.model_identity()
    finally:
        adapter.disconnect()
    return profile, profile_sha256, identity


def _result(
    *,
    config: ProjectConfig,
    adapter: MujocoBiSOAdapter,
    summary: RolloutSummary,
    policy_kind: str,
    checkpoint_sha256: str | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": "G8-simulation",
        "result": "pass",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "config_sha256": project_config_sha256(config),
        "mujoco_model_identity": adapter.model_identity(),
        "policy": policy_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "primary_cameras": list(PRIMARY_CAMERA_KEYS),
        "head_camera": config.runtime.use_head_camera,
        "camera_resolution": [config.mujoco.camera_height, config.mujoco.camera_width],
        "physics_steps_per_servo_tick": adapter.physics_steps_per_servo_tick,
        "physics_steps": adapter.physics_step_count,
        "collision_gate": config.mujoco.forbid_collisions,
        "rollout": asdict(summary),
    }


def run_mujoco_smoke(
    config: ProjectConfig,
    *,
    policy_steps: int = 1,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Run a small real-dynamics, real-rendering dual-arm simulation rollout."""

    _require_mujoco_backend(config)
    if policy_steps < 1:
        raise MujocoCLIError("policy_steps must be positive")
    adapter = _adapter(config)
    prompt = _capture_home_prompt(adapter)
    policy = _WristRollSmokePolicy(
        horizon=config.runtime.action_horizon,
        servo_hz=config.runtime.servo_hz,
    )
    runtime = SO101WAMRuntime(
        config=config, prompt=prompt, robot=adapter, policy=policy
    )
    virtual_clock = _VirtualClock()
    effective_clock = clock or virtual_clock
    effective_sleeper = sleeper or virtual_clock.sleep
    summary = run_managed_rollout(
        runtime,
        adapter,
        policy_steps=policy_steps,
        calibrate_on_connect=False,
        clock=effective_clock,
        sleeper=effective_sleeper,
    )
    return _result(
        config=config, adapter=adapter, summary=summary, policy_kind="wrist_roll_smoke"
    )


def _checkpoint_identity(
    checkpoint_path: str | Path,
    metadata: object,
) -> tuple[str, str]:
    if not isinstance(metadata, dict):
        metadata = dict(cast(Any, metadata))
    metadata_dict = cast(dict[str, Any], metadata)
    checkpoint_id = metadata_dict.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise MujocoCLIError(
            "checkpoint G8 report requires checkpoint metadata checkpoint_id"
        )
    return file_sha256(checkpoint_path), checkpoint_id.strip()


def _run_checkpoint_policy_session(
    config: ProjectConfig,
    *,
    adapter: MujocoBiSOAdapter,
    prompt: PhysicalPrompt,
    policy: SnapshotPolicy,
    policy_steps: int,
    policy_kind: str,
    checkpoint_sha256: str,
    checkpoint_id: str,
    clock: Callable[[], float] | None,
    sleeper: Callable[[float], None] | None,
    rollout_observer: RolloutObserver | None,
    semantic_object_body: str | None,
    initial_object_position: tuple[float, float, float] | None,
    initial_task_block_position: tuple[float, float, float] | None,
) -> dict[str, Any]:
    runtime = SO101WAMRuntime(
        config=config,
        prompt=prompt,
        robot=adapter,
        policy=policy,
    )
    terminal_frame: SensorimotorFrame | None = None
    terminal_object_position: tuple[float, float, float] | None = None
    object_profile: dict[str, object] | None = None
    object_profile_sha256: str | None = None
    object_body = semantic_object_body
    if object_body is None and (
        initial_object_position is not None or initial_task_block_position is not None
    ):
        object_body = "task_block"

    def observe_terminal(timestamp_s: float) -> None:
        nonlocal terminal_frame, terminal_object_position
        nonlocal object_profile, object_profile_sha256
        terminal_frame = adapter.get_observation(timestamp_s=timestamp_s)
        if object_body is not None:
            terminal_object_position = adapter.object_body_position(object_body)
            object_profile = adapter.object_physical_profile(object_body)
            object_profile_sha256 = adapter.object_physical_profile_sha256(object_body)

    virtual_clock = _VirtualClock()
    effective_clock = clock or virtual_clock
    effective_sleeper = sleeper or virtual_clock.sleep
    session_observer = (
        None
        if rollout_observer is None
        else _MujocoRolloutObserver(rollout_observer, adapter, policy)
    )
    summary = run_managed_rollout(
        runtime,
        adapter,
        policy_steps=policy_steps,
        calibrate_on_connect=False,
        clock=effective_clock,
        sleeper=effective_sleeper,
        terminal_observer=observe_terminal,
        rollout_observer=session_observer,
    )
    if terminal_frame is None:
        raise MujocoCLIError("checkpoint session did not capture terminal state")
    result = _result(
        config=config,
        adapter=adapter,
        summary=summary,
        policy_kind=policy_kind,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
    )
    result["terminal_joint_position"] = [
        float(value) for value in terminal_frame.joint_position
    ]
    if object_body is None:
        return result
    if terminal_object_position is None:
        raise MujocoCLIError("checkpoint session did not capture object state")
    if object_profile is None or object_profile_sha256 is None:
        raise MujocoCLIError("checkpoint session did not capture object profile")

    object_initial = (
        initial_task_block_position
        if initial_task_block_position is not None
        else initial_object_position
    )
    result["object_body"] = object_body
    if object_initial is not None:
        result["initial_object_position"] = list(object_initial)
    result["terminal_object_position"] = list(terminal_object_position)
    result["object_physical_profile"] = object_profile
    result["object_physical_profile_sha256"] = object_profile_sha256
    if initial_task_block_position is not None:
        result["initial_task_block_position"] = list(initial_task_block_position)
        result["terminal_task_block_position"] = list(terminal_object_position)
    return result


def run_mujoco_checkpoint_session(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    prompt_path: str | Path,
    manifest_path: str | Path | None = None,
    policy_steps: int,
    device: str = "cpu",
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
    rollout_observer: RolloutObserver | None = None,
    semantic_object_body: str | None = None,
    initial_object_position: tuple[float, float, float] | None = None,
    initial_task_block_position: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Run the same checkpoint/prompt contract as hardware against MuJoCo."""

    _require_mujoco_backend(config)
    if policy_steps < 1:
        raise MujocoCLIError("policy_steps must be positive")
    bundle = load_compact_wam_bundle(checkpoint_path, device=device)
    checkpoint_sha256, checkpoint_id = _checkpoint_identity(
        checkpoint_path,
        bundle.metadata,
    )
    if bundle.model.action_horizon != config.runtime.action_horizon:
        raise MujocoCLIError(
            "checkpoint action_horizon does not match runtime.action_horizon: "
            f"{bundle.model.action_horizon} != {config.runtime.action_horizon}"
        )
    episode = load_episode(prompt_path, manifest_path)
    prompt = physical_prompt_from_episode(episode, policy_hz=config.runtime.policy_hz)
    adapter = _adapter(
        config,
        semantic_object_body=semantic_object_body,
        initial_object_position=initial_object_position,
        initial_task_block_position=initial_task_block_position,
    )
    policy = CompactWAMPolicy(
        bundle.model,
        servo_hz=config.runtime.servo_hz,
        device=device,
        telemetry_mode=(
            FutureLatentTelemetryMode.CAPTURE
            if getattr(rollout_observer, "on_future_latent_step", None) is not None
            else FutureLatentTelemetryMode.DISABLED
        ),
    )
    return _run_checkpoint_policy_session(
        config,
        adapter=adapter,
        prompt=prompt,
        policy=policy,
        policy_steps=policy_steps,
        policy_kind="compact_wam",
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
        clock=clock,
        sleeper=sleeper,
        rollout_observer=rollout_observer,
        semantic_object_body=semantic_object_body,
        initial_object_position=initial_object_position,
        initial_task_block_position=initial_task_block_position,
    )


def run_mujoco_task_spec_checkpoint_session(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    task_spec: TaskSpec,
    policy_steps: int,
    device: str = "cpu",
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
    rollout_observer: RolloutObserver | None = None,
    semantic_object_body: str | None = None,
    initial_object_position: tuple[float, float, float] | None = None,
    initial_task_block_position: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Run a checkpoint from an action-free task spec in MuJoCo only."""

    _require_mujoco_backend(config)
    if policy_steps < 1:
        raise MujocoCLIError("policy_steps must be positive")
    kind = task_spec_kind(task_spec)
    bundle = load_compact_wam_bundle(checkpoint_path, device=device)
    checkpoint_sha256, checkpoint_id = _checkpoint_identity(
        checkpoint_path,
        bundle.metadata,
    )
    if bundle.model.action_horizon != config.runtime.action_horizon:
        raise MujocoCLIError(
            "checkpoint action_horizon does not match runtime.action_horizon: "
            f"{bundle.model.action_horizon} != {config.runtime.action_horizon}"
        )

    adapter = _adapter(
        config,
        semantic_object_body=semantic_object_body,
        initial_object_position=initial_object_position,
        initial_task_block_position=initial_task_block_position,
    )
    placeholder_prompt = _capture_home_prompt(adapter)
    base_policy = CompactWAMPolicy(
        bundle.model,
        servo_hz=config.runtime.servo_hz,
        device=device,
        telemetry_mode=(
            FutureLatentTelemetryMode.CAPTURE
            if getattr(rollout_observer, "on_future_latent_step", None) is not None
            else FutureLatentTelemetryMode.DISABLED
        ),
    )
    policy = _TaskSpecSnapshotPolicy(base_policy, task_spec)
    result = _run_checkpoint_policy_session(
        config,
        adapter=adapter,
        prompt=placeholder_prompt,
        policy=policy,
        policy_steps=policy_steps,
        policy_kind="compact_wam_task_spec",
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
        clock=clock,
        sleeper=sleeper,
        rollout_observer=rollout_observer,
        semantic_object_body=semantic_object_body,
        initial_object_position=initial_object_position,
        initial_task_block_position=initial_task_block_position,
    )
    result.update(
        {
            "task_spec_kind": kind.value,
            "task_spec_fingerprint": task_spec.fingerprint,
            "task_spec_source_id": task_spec.provenance.source_id,
            "task_spec_source_sha256": task_spec.provenance.source_sha256,
            "runtime_placeholder_prompt": "mujoco_home_hold",
            "runtime_placeholder_prompt_fingerprint": (placeholder_prompt.fingerprint),
            "runtime_placeholder_prompt_used_by_model": False,
        }
    )
    return result


def write_mujoco_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Atomically publish one G8 report without overwriting an existing file."""

    target = Path(path)
    if target.exists():
        raise MujocoCLIError(f"MuJoCo report already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report,
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
        with temp_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != report:
                raise MujocoCLIError("MuJoCo report round-trip validation failed")
        try:
            os.link(temp_path, target)
        except FileExistsError as error:
            raise MujocoCLIError(f"MuJoCo report already exists: {target}") from error
        temp_path.unlink()
        temp_path = None
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed-torso dual-SO-101 MuJoCo scene."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_MUJOCO_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", type=Path, help="SO101-WAM episode .npz")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report",
        type=Path,
        help="atomically write a no-overwrite G8 simulation JSON report",
    )
    parser.add_argument(
        "--shadow", action="store_true", help="render/observe without stepping controls"
    )
    parser.add_argument(
        "--head-camera",
        action="store_true",
        help="also render the diagnostic head view",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if (args.checkpoint is None) != (args.prompt is None):
        parser.error("--checkpoint and --prompt must be supplied together")
    if args.report is not None and args.report.exists():
        parser.error(f"MuJoCo report already exists: {args.report}")

    try:
        config = ProjectConfig.load(args.config)
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                actuation_enabled=False
                if args.shadow
                else config.runtime.actuation_enabled,
                use_head_camera=args.head_camera or config.runtime.use_head_camera,
            ),
        )
        if args.checkpoint is None:
            result = run_mujoco_smoke(config, policy_steps=args.steps)
        else:
            assert args.prompt is not None
            result = run_mujoco_checkpoint_session(
                config,
                checkpoint_path=args.checkpoint,
                prompt_path=args.prompt,
                manifest_path=args.manifest,
                policy_steps=args.steps,
                device=args.device,
            )
    except (
        CheckpointError,
        ConfigError,
        DatasetError,
        MujocoAdapterError,
        MujocoCLIError,
        DeploymentCertificationError,
        PolicyError,
        RolloutError,
        RuntimeErrorState,
    ) as error:
        parser.error(str(error))

    if args.report is not None:
        write_mujoco_report(args.report, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
