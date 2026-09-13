"""GEN-1.5-style prompt plus live sensorimotor context."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import floor, isfinite
from typing import Deque

import numpy as np
from numpy.typing import NDArray

from .contracts import ContractError, PhysicalPrompt, SensorimotorFrame

PROMPT_SEGMENT = "prompt"
LIVE_SEGMENT = "live"


def _readonly_bool_mask(value: NDArray[np.bool_]) -> NDArray[np.bool_]:
    mask = np.array(value, dtype=np.bool_, copy=True)
    mask.flags.writeable = False
    return mask


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Immutable model-facing view of a pinned prompt and rolling live frames."""

    prompt_fingerprint: str
    prompt_frames: tuple[SensorimotorFrame, ...]
    live_frames: tuple[SensorimotorFrame, ...]
    ordered_frames: tuple[SensorimotorFrame, ...]
    attention_mask: NDArray[np.bool_]
    prompt_mask: NDArray[np.bool_]
    live_mask: NDArray[np.bool_]
    causal_mask: NDArray[np.bool_]
    segment_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        prompt_frames = tuple(self.prompt_frames)
        live_frames = tuple(self.live_frames)
        ordered_frames = tuple(self.ordered_frames)
        expected_order = prompt_frames + live_frames
        if ordered_frames != expected_order:
            raise ContractError("ordered_frames must be prompt_frames followed by live_frames")
        if len(prompt_frames) < 2:
            raise ContractError("snapshot prompt_frames must contain at least two frames")
        prompt_duration_s = (
            prompt_frames[-1].timestamp_s - prompt_frames[0].timestamp_s
        )
        prompt = PhysicalPrompt(
            prompt_frames,
            min_duration_s=prompt_duration_s,
            max_duration_s=prompt_duration_s,
        )
        if self.prompt_fingerprint != prompt.fingerprint:
            raise ContractError(
                "prompt_fingerprint does not match snapshot prompt_frames"
            )

        size = len(ordered_frames)
        if len(self.segment_labels) != size:
            raise ContractError("segment_labels length must match ordered_frames")
        if self.segment_labels != (PROMPT_SEGMENT,) * len(prompt_frames) + (LIVE_SEGMENT,) * len(live_frames):
            raise ContractError("segment_labels must identify prompt and live frame segments")

        attention_mask = _readonly_bool_mask(self.attention_mask)
        prompt_mask = _readonly_bool_mask(self.prompt_mask)
        live_mask = _readonly_bool_mask(self.live_mask)
        causal_mask = _readonly_bool_mask(self.causal_mask)

        if attention_mask.shape != (size,):
            raise ContractError("attention_mask must have shape [context]")
        if prompt_mask.shape != (size,):
            raise ContractError("prompt_mask must have shape [context]")
        if live_mask.shape != (size,):
            raise ContractError("live_mask must have shape [context]")
        if causal_mask.shape != (size, size):
            raise ContractError("causal_mask must have shape [context, context]")
        if not attention_mask.all():
            raise ContractError("attention_mask must mark every emitted frame")
        expected_prompt_mask = np.array([True] * len(prompt_frames) + [False] * len(live_frames), dtype=np.bool_)
        expected_live_mask = ~expected_prompt_mask
        if not np.array_equal(prompt_mask, expected_prompt_mask):
            raise ContractError("prompt_mask must mark only prompt frames")
        if not np.array_equal(live_mask, expected_live_mask):
            raise ContractError("live_mask must mark only live frames")
        if not np.array_equal(causal_mask, np.tril(np.ones((size, size), dtype=np.bool_))):
            raise ContractError("causal_mask must be lower triangular")

        object.__setattr__(self, "prompt_frames", prompt_frames)
        object.__setattr__(self, "live_frames", live_frames)
        object.__setattr__(self, "ordered_frames", ordered_frames)
        object.__setattr__(self, "attention_mask", attention_mask)
        object.__setattr__(self, "prompt_mask", prompt_mask)
        object.__setattr__(self, "live_mask", live_mask)
        object.__setattr__(self, "causal_mask", causal_mask)

    @property
    def frames(self) -> tuple[SensorimotorFrame, ...]:
        return self.ordered_frames


class Gen15Context:
    """Pinned physical prompt with a deterministic live FIFO context window."""

    def __init__(
        self,
        prompt: PhysicalPrompt,
        *,
        policy_hz: float,
        total_duration_s: float = 30.0,
    ) -> None:
        if not isfinite(policy_hz) or policy_hz <= 0:
            raise ContractError("policy_hz must be finite and positive")
        if not isfinite(total_duration_s) or total_duration_s <= 0:
            raise ContractError("total_duration_s must be finite and positive")

        total_capacity = floor(total_duration_s * policy_hz)
        if total_capacity < 1:
            raise ContractError("context capacity must include at least one frame")
        if len(prompt.frames) > total_capacity:
            raise ContractError("physical prompt exceeds total context capacity")

        self._prompt = prompt
        self._policy_hz = float(policy_hz)
        self._total_duration_s = float(total_duration_s)
        self._total_capacity = int(total_capacity)
        self._live_capacity = self._total_capacity - len(prompt.frames)
        self._live_frames: Deque[SensorimotorFrame] = deque(maxlen=self._live_capacity)
        self._last_live_timestamp_s: float | None = None

    @property
    def prompt(self) -> PhysicalPrompt:
        return self._prompt

    @property
    def policy_hz(self) -> float:
        return self._policy_hz

    @property
    def total_duration_s(self) -> float:
        return self._total_duration_s

    @property
    def total_capacity(self) -> int:
        return self._total_capacity

    @property
    def live_capacity(self) -> int:
        return self._live_capacity

    @property
    def live_size(self) -> int:
        return len(self._live_frames)

    def append_live(self, frame: SensorimotorFrame) -> None:
        """Append one live frame, rejecting stale or nonmonotonic timestamps."""

        if self._live_capacity == 0:
            raise ContractError("live frame capacity is zero")
        if self._last_live_timestamp_s is not None and frame.timestamp_s <= self._last_live_timestamp_s:
            raise ContractError("live timestamps must be strictly increasing")

        self._live_frames.append(frame)
        self._last_live_timestamp_s = frame.timestamp_s

    def extend_live(self, frames: tuple[SensorimotorFrame, ...] | list[SensorimotorFrame]) -> None:
        for frame in frames:
            self.append_live(frame)

    def reset_live(self) -> None:
        """Clear live frames while preserving the pinned physical prompt."""

        self._live_frames.clear()
        self._last_live_timestamp_s = None

    def rebind_prompt(self, prompt: PhysicalPrompt) -> None:
        """Replace the pinned prompt and clear live state for a new episode."""

        if len(prompt.frames) > self._total_capacity:
            raise ContractError("physical prompt exceeds total context capacity")
        self._prompt = prompt
        self._live_capacity = self._total_capacity - len(prompt.frames)
        self._live_frames = deque(maxlen=self._live_capacity)
        self._last_live_timestamp_s = None

    def snapshot(self) -> ContextSnapshot:
        prompt_frames = self._prompt.frames
        live_frames = tuple(self._live_frames)
        ordered_frames = prompt_frames + live_frames
        size = len(ordered_frames)
        prompt_count = len(prompt_frames)

        attention_mask = np.ones(size, dtype=np.bool_)
        prompt_mask = np.zeros(size, dtype=np.bool_)
        prompt_mask[:prompt_count] = True
        live_mask = ~prompt_mask
        causal_mask = np.tril(np.ones((size, size), dtype=np.bool_))
        segment_labels = (PROMPT_SEGMENT,) * prompt_count + (LIVE_SEGMENT,) * len(live_frames)

        return ContextSnapshot(
            prompt_fingerprint=self._prompt.fingerprint,
            prompt_frames=prompt_frames,
            live_frames=live_frames,
            ordered_frames=ordered_frames,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            live_mask=live_mask,
            causal_mask=causal_mask,
            segment_labels=segment_labels,
        )
