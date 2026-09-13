"""Fail-closed prompt, policy, and one-row-at-a-time servo runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import ProjectConfig
from .constants import ACTION_DIM, JOINT_KEYS
from .context import ContextSnapshot, Gen15Context
from .contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from .safety import SafetyDecision, SafetySupervisor
from .tensorizer import CompactWAMBatch, tensorize_context


class RuntimeErrorState(RuntimeError):
    """Raised when the runtime cannot advance without violating an invariant."""


class RuntimeState(str, Enum):
    BOOT = "boot"
    PROMPT_CACHED = "prompt_cached"
    ROLLOUT_READY = "rollout_ready"
    ROLLING = "rolling"
    RECOVERY = "recovery"
    HALT = "halt"


class RobotAdapter(Protocol):
    def get_observation(self, *, timestamp_s: float) -> SensorimotorFrame: ...

    def send_action(self, action: ActionChunk | NDArray[np.float32]) -> object: ...


class SnapshotPolicy(Protocol):
    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk: ...


def _readonly_vector(value: object, *, name: str) -> NDArray[np.float32]:
    if isinstance(value, Mapping):
        actual = set(value)
        expected = set(JOINT_KEYS)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            extra = tuple(sorted(actual - expected))
            raise RuntimeErrorState(
                f"{name} joint keys must be exact; missing={missing}, extra={extra}"
            )
        value = [value[key] for key in JOINT_KEYS]
    array = np.array(value, dtype=np.float32, copy=True)
    if array.shape != (ACTION_DIM,):
        raise RuntimeErrorState(
            f"{name} must have shape ({ACTION_DIM},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise RuntimeErrorState(f"{name} contains NaN or infinity")
    array.flags.writeable = False
    return array


@dataclass(slots=True)
class ServoExecutor:
    """Receding-horizon queue that emits exactly one target row per tick."""

    servo_hz: float
    _targets: NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)
    _created_at_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isfinite(self.servo_hz) or self.servo_hz <= 0:
            raise RuntimeErrorState("servo_hz must be finite and positive")

    @property
    def dt_s(self) -> float:
        return 1.0 / self.servo_hz

    @property
    def pending(self) -> int:
        if self._targets is None:
            return 0
        return int(self._targets.shape[0] - self._index)

    def clear(self) -> None:
        self._targets = None
        self._index = 0
        self._created_at_s = None

    def load(self, chunk: ActionChunk) -> None:
        tolerance = max(1e-9, self.dt_s * 1e-6)
        if abs(chunk.dt_s - self.dt_s) > tolerance:
            raise RuntimeErrorState(
                f"action dt_s must match servo period {self.dt_s:.9f}s, got {chunk.dt_s:.9f}s"
            )
        self._targets = np.array(
            chunk.target_joint_position, dtype=np.float32, copy=True
        )
        self._index = 0
        self._created_at_s = chunk.created_at_s

    def next_tick(self, *, now_s: float) -> ActionChunk:
        if self._targets is None or self._index >= self._targets.shape[0]:
            raise RuntimeErrorState("no pending action chunk")
        if self._created_at_s is None:
            raise RuntimeErrorState("queued action chunk has no creation timestamp")
        created_at_s = self._created_at_s
        row = self._targets[self._index : self._index + 1]
        self._index += 1
        action = ActionChunk(
            target_joint_position=row, dt_s=self.dt_s, created_at_s=created_at_s
        )
        if self._index >= self._targets.shape[0]:
            self.clear()
        return action


@dataclass(frozen=True, slots=True)
class PolicyStep:
    snapshot: ContextSnapshot
    batch: CompactWAMBatch
    action: ActionChunk


@dataclass(frozen=True, slots=True)
class ServoStep:
    state: RuntimeState
    observation: SensorimotorFrame
    action: ActionChunk
    safety: SafetyDecision
    executed_action: NDArray[np.float32]
    sent: bool
    shadow: bool
    pending: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "executed_action",
            _readonly_vector(self.executed_action, name="executed_action"),
        )


@dataclass(slots=True)
class SO101WAMRuntime:
    """GEN-1.5-style pinned prompt with separate policy and servo clocks."""

    config: ProjectConfig
    prompt: PhysicalPrompt
    robot: RobotAdapter
    policy: SnapshotPolicy
    safety: SafetySupervisor | None = None
    clock: Callable[[], float] | None = None
    state: RuntimeState = RuntimeState.BOOT
    context: Gen15Context = field(init=False)
    executor: ServoExecutor = field(init=False)
    last_executed_action: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        robot_backend = getattr(self.robot, "backend", None)
        if robot_backend is not None and robot_backend != self.config.runtime.backend:
            raise RuntimeErrorState(
                f"robot backend {robot_backend!r} does not match runtime backend {self.config.runtime.backend!r}"
            )
        robot_actuation = getattr(self.robot, "actuation_enabled", None)
        if (
            robot_actuation is not None
            and bool(robot_actuation) != self.config.runtime.actuation_enabled
        ):
            raise RuntimeErrorState(
                "robot adapter actuation flag must match runtime.actuation_enabled"
            )
        self.context = Gen15Context(
            self.prompt,
            policy_hz=self.config.runtime.policy_hz,
            total_duration_s=self.config.runtime.context_seconds,
        )
        if self.safety is None:
            self.safety = SafetySupervisor(self.config.safety)
        self.executor = ServoExecutor(self.config.runtime.servo_hz)
        self.state = RuntimeState.PROMPT_CACHED

    @property
    def prompt_fingerprint(self) -> str:
        return self.prompt.fingerprint

    def _require_active(self) -> None:
        if self.state is RuntimeState.HALT:
            raise RuntimeErrorState("runtime is halted")
        if self.state is RuntimeState.RECOVERY:
            raise RuntimeErrorState(
                "runtime requires explicit recovery before continuing"
            )

    def _capture_policy_frame(self, *, now_s: float) -> SensorimotorFrame:
        observed = self.robot.get_observation(timestamp_s=now_s)
        aligned_action = (
            observed.joint_position
            if self.last_executed_action is None
            else self.last_executed_action
        )
        frame = SensorimotorFrame(
            timestamp_s=observed.timestamp_s,
            images=observed.images,
            joint_position=observed.joint_position,
            executed_action=aligned_action,
            image_timestamps_s=observed.image_timestamps_s,
        )
        self.context.append_live(frame)
        return frame

    def prime_live(self, *, now_s: float) -> ContextSnapshot:
        """Add one policy-rate measured hold frame without requesting an action."""

        self._require_active()
        if self.state not in {RuntimeState.PROMPT_CACHED, RuntimeState.ROLLOUT_READY}:
            raise RuntimeErrorState(
                f"prime_live requires no pending rollout, got {self.state.value}"
            )
        if self.executor.pending:
            raise RuntimeErrorState("prime_live cannot run with queued action rows")
        self._capture_policy_frame(now_s=now_s)
        self.state = RuntimeState.ROLLOUT_READY
        return self.context.snapshot()

    def policy_step(self, *, now_s: float) -> PolicyStep:
        """Capture one policy-rate frame and replace the receding action horizon."""

        self._require_active()
        self._capture_policy_frame(now_s=now_s)
        snapshot = self.context.snapshot()
        batch = tensorize_context(snapshot)
        try:
            action = self.policy.predict(snapshot, now_s=now_s)
            if action.horizon != self.config.runtime.action_horizon:
                raise RuntimeErrorState(
                    f"policy horizon must be {self.config.runtime.action_horizon}, got {action.horizon}"
                )
            self.executor.load(action)
        except Exception:
            self.executor.clear()
            self.state = RuntimeState.RECOVERY
            raise
        self.state = RuntimeState.ROLLING
        return PolicyStep(snapshot=snapshot, batch=batch, action=action)

    def servo_step(self, *, now_s: float) -> ServoStep:
        """Consume and, if enabled, send one already-predicted action row."""

        self._require_active()
        if self.state is not RuntimeState.ROLLING:
            raise RuntimeErrorState(
                f"servo step requires rolling state, got {self.state.value}"
            )
        if self.safety is None:
            raise RuntimeErrorState("safety supervisor is not initialized")

        observation = self.robot.get_observation(timestamp_s=now_s)
        action = self.executor.next_tick(now_s=now_s)
        evaluation_now_s = now_s if self.clock is None else self.clock()
        decision = self.safety.evaluate(observation, action, now_s=evaluation_now_s)
        sent = False
        shadow = not self.config.runtime.actuation_enabled

        executed: object
        if not decision.accepted:
            self.executor.clear()
            executed = observation.joint_position
            self.state = RuntimeState.RECOVERY
        elif shadow:
            # No command crossed the adapter boundary. Keep the measured hold,
            # not the policy target, as the next sensorimotor history action.
            executed = observation.joint_position
            self.state = (
                RuntimeState.ROLLING
                if self.executor.pending
                else RuntimeState.ROLLOUT_READY
            )
        else:
            try:
                returned = self.robot.send_action(decision.action)
            except Exception:
                self.executor.clear()
                self.state = RuntimeState.RECOVERY
                raise
            executed = (
                decision.action.target_joint_position[0]
                if returned is None
                else returned
            )
            sent = True
            self.state = (
                RuntimeState.ROLLING
                if self.executor.pending
                else RuntimeState.ROLLOUT_READY
            )

        self.last_executed_action = _readonly_vector(
            executed, name="executed/sent action"
        )
        return ServoStep(
            state=self.state,
            observation=observation,
            action=decision.action,
            safety=decision,
            executed_action=self.last_executed_action,
            sent=sent,
            shadow=shadow,
            pending=self.executor.pending,
        )

    def recover(self) -> None:
        if self.state is not RuntimeState.RECOVERY:
            raise RuntimeErrorState(
                f"recover requires recovery state, got {self.state.value}"
            )
        if self.safety is None or self.safety.fault_latch.latched:
            raise RuntimeErrorState("cannot recover while a runtime fault is latched")
        self.executor.clear()
        self.state = RuntimeState.ROLLOUT_READY

    def pause_rollout(self) -> None:
        """Discard an accepted pending horizon without emitting another command."""

        self._require_active()
        self.executor.clear()
        self.state = RuntimeState.ROLLOUT_READY

    def halt(self, reason: str) -> None:
        if not reason:
            raise RuntimeErrorState("halt requires a reason")
        if self.safety is None:
            raise RuntimeErrorState("safety supervisor is not initialized")
        self.safety.trigger_fault(reason)
        self.executor.clear()
        self.state = RuntimeState.HALT


__all__ = [
    "PolicyStep",
    "RobotAdapter",
    "RuntimeErrorState",
    "RuntimeState",
    "SO101WAMRuntime",
    "ServoExecutor",
    "ServoStep",
    "SnapshotPolicy",
]
