"""Software safety checks for bimanual SO-101 action chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

import numpy as np

from .config import SafetyConfig
from .constants import ACTION_DIM, JOINT_KEYS
from .contracts import ActionChunk, SensorimotorFrame

_TIME_EPSILON_S = 1e-9


class SafetyError(RuntimeError):
    """Raised when a requested safety operation violates a safety invariant."""


@dataclass(slots=True)
class FaultLatch:
    """Latched runtime fault state.

    Triggering is sticky by design. The latch can only be cleared by an explicit
    reset call, which leaves a small audit trail for tests and runtime logs. It
    is process-level fault containment, not external emergency-stop hardware.
    """

    latched: bool = False
    reason: str | None = None
    reset_count: int = 0
    trigger_count: int = 0

    def trigger(self, reason: str) -> None:
        if not reason:
            raise SafetyError("fault latch trigger requires a reason")
        self.latched = True
        self.reason = reason
        self.trigger_count += 1

    def reset(self) -> None:
        self.latched = False
        self.reason = None
        self.reset_count += 1


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """Auditable result of applying the safety supervisor to one action chunk."""

    accepted: bool
    action: ActionChunk
    reasons: tuple[str, ...]
    clipped: bool = False
    hold: bool = False
    fault_latched: bool = False


@dataclass(slots=True)
class SafetySupervisor:
    """Validate observations/actions and return an auditable measured hold on rejection.

    A rejected decision is not permission to transmit the hold target. The
    managed hardware path sends nothing, latches a fault, and disconnects.
    """

    config: SafetyConfig
    fault_latch: FaultLatch = field(default_factory=FaultLatch)
    _lower: np.ndarray = field(init=False, repr=False)
    _upper: np.ndarray = field(init=False, repr=False)
    _observation_lower: np.ndarray = field(init=False, repr=False)
    _observation_upper: np.ndarray = field(init=False, repr=False)
    _max_delta: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lower = self._config_vector(self.config.joint_lower, "joint_lower")
        self._upper = self._config_vector(self.config.joint_upper, "joint_upper")
        self._observation_lower = self._config_vector(
            self.config.joint_lower
            if self.config.observation_joint_lower is None
            else self.config.observation_joint_lower,
            "observation_joint_lower",
        )
        self._observation_upper = self._config_vector(
            self.config.joint_upper
            if self.config.observation_joint_upper is None
            else self.config.observation_joint_upper,
            "observation_joint_upper",
        )
        self._max_delta = self._config_vector(
            self.config.max_delta_per_servo_tick,
            "max_delta_per_servo_tick",
        )

    def trigger_fault(self, reason: str) -> None:
        self.fault_latch.trigger(reason)

    def reset_fault(self) -> None:
        self.fault_latch.reset()

    def hold_action(
        self,
        observation: SensorimotorFrame | None,
        *,
        dt_s: float,
        created_at_s: float,
    ) -> ActionChunk:
        if observation is None:
            target = np.clip(
                np.zeros(ACTION_DIM, dtype=np.float32), self._lower, self._upper
            )
        else:
            target = self._require_frame(observation).joint_position
            target = np.clip(target, self._lower, self._upper)
        return ActionChunk(
            target_joint_position=target.reshape(1, ACTION_DIM),
            dt_s=dt_s,
            created_at_s=created_at_s,
        )

    def evaluate(
        self,
        observation: SensorimotorFrame,
        action: ActionChunk,
        *,
        now_s: float,
    ) -> SafetyDecision:
        if not isfinite(now_s) or now_s < 0:
            raise SafetyError("now_s must be finite and non-negative")

        reasons: list[str] = []
        frame = self._require_frame(observation)
        chunk = self._require_action_chunk(action)

        if self.fault_latch.latched:
            reasons.append("runtime_fault_latched")
            if self.fault_latch.reason:
                reasons.append(f"runtime_fault_reason:{self.fault_latch.reason}")

        if frame.timestamp_s - now_s > self.config.max_camera_skew_s + _TIME_EPSILON_S:
            reasons.append("future_observation")
        if chunk.created_at_s - now_s > chunk.dt_s + _TIME_EPSILON_S:
            reasons.append("future_action")
        if now_s - frame.timestamp_s > self.config.max_observation_age_s:
            reasons.append("stale_observation")
        if now_s - chunk.created_at_s > self.config.watchdog_timeout_s:
            reasons.append("stale_action")
        camera_skew_s = frame.primary_image_skew_s
        if camera_skew_s is not None and camera_skew_s > self.config.max_camera_skew_s:
            reasons.append("camera_skew")

        self._append_limit_reasons(
            frame.joint_position,
            self._observation_lower,
            self._observation_upper,
            reasons,
            prefix="observation",
        )
        for index, target in enumerate(chunk.target_joint_position):
            self._append_limit_reasons(
                target,
                self._lower,
                self._upper,
                reasons,
                prefix=f"action[{index}]",
            )

        # A valid measurement can lie outside the command envelope. Do not let
        # rate clipping produce an invalid command when that envelope is unreachable.
        if not reasons and np.any(
            (frame.joint_position + self._max_delta < self._lower)
            | (frame.joint_position - self._max_delta > self._upper)
        ):
            reasons.append("action_unreachable_from_observation")

        if reasons:
            return SafetyDecision(
                accepted=False,
                action=self.hold_action(frame, dt_s=chunk.dt_s, created_at_s=now_s),
                reasons=tuple(reasons),
                hold=True,
                fault_latched=self.fault_latch.latched,
            )

        clipped_targets, clipped = self._clip_per_tick(
            frame.joint_position, chunk.target_joint_position
        )
        output = ActionChunk(
            target_joint_position=clipped_targets,
            dt_s=chunk.dt_s,
            created_at_s=chunk.created_at_s,
        )
        output_reasons = ("max_delta_clipped",) if clipped else ()
        return SafetyDecision(
            accepted=True,
            action=output,
            reasons=output_reasons,
            clipped=clipped,
            hold=False,
            fault_latched=False,
        )

    @staticmethod
    def _require_frame(frame: SensorimotorFrame) -> SensorimotorFrame:
        # Reconstruct through the public contract so copied arrays are checked
        # for shape, dtype, camera presence, and NaN/Inf values.
        return SensorimotorFrame(
            timestamp_s=frame.timestamp_s,
            images=frame.images,
            joint_position=frame.joint_position,
            executed_action=frame.executed_action,
            image_timestamps_s=frame.image_timestamps_s,
        )

    @staticmethod
    def _require_action_chunk(chunk: ActionChunk) -> ActionChunk:
        return ActionChunk(
            target_joint_position=chunk.target_joint_position,
            dt_s=chunk.dt_s,
            created_at_s=chunk.created_at_s,
        )

    @staticmethod
    def _config_vector(values: Iterable[float], name: str) -> np.ndarray:
        array = np.array(tuple(values), dtype=np.float32)
        if array.shape != (ACTION_DIM,):
            raise SafetyError(f"{name} must contain {ACTION_DIM} values")
        if not np.isfinite(array).all():
            raise SafetyError(f"{name} contains NaN or infinity")
        array.flags.writeable = False
        return array

    @staticmethod
    def _append_limit_reasons(
        values: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        reasons: list[str],
        *,
        prefix: str,
    ) -> None:
        low = values < lower
        high = values > upper
        for index in np.flatnonzero(low | high):
            direction = "below" if low[index] else "above"
            reasons.append(f"{prefix}_joint_limit:{JOINT_KEYS[int(index)]}:{direction}")

    def _clip_per_tick(
        self,
        current_position: np.ndarray,
        target_joint_position: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        clipped = np.empty_like(target_joint_position, dtype=np.float32)
        previous = np.array(current_position, dtype=np.float32, copy=True)
        any_clipped = False

        for row_index, target in enumerate(target_joint_position):
            low = previous - self._max_delta
            high = previous + self._max_delta
            clipped[row_index] = np.clip(target, low, high)
            any_clipped = any_clipped or not np.array_equal(clipped[row_index], target)
            previous = clipped[row_index]

        return clipped, any_clipped


__all__ = [
    "SafetyDecision",
    "SafetyError",
    "SafetySupervisor",
    "FaultLatch",
]
