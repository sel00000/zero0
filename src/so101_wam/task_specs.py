"""Task specifications that stay separate from robot sensorimotor prompts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import blake2b
from math import isfinite
import re
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .contracts import PhysicalPrompt, RGBImage


HUMAN_VIDEO_MIN_DURATION_S = 3.0
HUMAN_VIDEO_MAX_DURATION_S = 12.0
MAX_LANGUAGE_BYTES = 512
_MIN_IMAGE_SIDE = 8
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BYTE_TOKEN_OFFSET = 1
_BYTE_TOKEN_SCALE = 257.0
_SECONDARY_PHASE_SCALE = 0.5


class TaskSpecError(ValueError):
    """Raised when a task specification violates its modality contract."""


class TaskSpecKind(StrEnum):
    ROBOT_EPISODE = "robot_episode"
    HUMAN_VIDEO = "human_video"
    LANGUAGE = "language"


@dataclass(frozen=True, slots=True)
class TaskSpecProvenance:
    source_id: str
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise TaskSpecError("task-spec source_id must be non-empty")
        source_id = self.source_id.strip()
        if len(source_id) > 256:
            raise TaskSpecError("task-spec source_id is too long")
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.source_sha256) is None
        ):
            raise TaskSpecError("task-spec source_sha256 must be a lowercase SHA-256")
        object.__setattr__(self, "source_id", source_id)


def _require_provenance(value: object) -> TaskSpecProvenance:
    if not isinstance(value, TaskSpecProvenance):
        raise TaskSpecError("task specification requires validated provenance")
    return value


def _readonly_rgb(value: NDArray[np.generic]) -> RGBImage:
    image = np.array(value, copy=True)
    if image.dtype != np.uint8:
        raise TaskSpecError("human video RGB frames must use uint8 pixels")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise TaskSpecError("human video RGB frames must have shape [H, W, 3]")
    if min(image.shape[:2]) < _MIN_IMAGE_SIDE:
        raise TaskSpecError(
            f"human video RGB frames must be at least {_MIN_IMAGE_SIDE}x{_MIN_IMAGE_SIDE}"
        )
    image.flags.writeable = False
    return image


@dataclass(frozen=True, slots=True)
class HumanVideoFrame:
    """One action-free human demonstration frame."""

    timestamp_s: float
    rgb: RGBImage

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise TaskSpecError(
                "human video timestamp_s must be finite and non-negative"
            )
        object.__setattr__(self, "rgb", _readonly_rgb(self.rgb))


@dataclass(frozen=True, slots=True)
class HumanVideoPrompt:
    """RGB-only task demonstration with no robot state or action fields."""

    frames: tuple[HumanVideoFrame, ...]
    provenance: TaskSpecProvenance
    text_metadata: str | None = None
    min_duration_s: float = HUMAN_VIDEO_MIN_DURATION_S
    max_duration_s: float = HUMAN_VIDEO_MAX_DURATION_S

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if len(frames) < 2:
            raise TaskSpecError("a human video prompt needs at least two frames")
        if any(not isinstance(frame, HumanVideoFrame) for frame in frames):
            raise TaskSpecError("human video frames must use HumanVideoFrame")
        if any(
            current.timestamp_s <= previous.timestamp_s
            for previous, current in zip(frames, frames[1:])
        ):
            raise TaskSpecError(
                "human video timestamps must be strictly increasing"
            )
        if len({frame.rgb.shape for frame in frames}) != 1:
            raise TaskSpecError("human video frames must share one resolution")
        if not 0 < self.min_duration_s <= self.max_duration_s:
            raise TaskSpecError("invalid human video duration bounds")
        if not self.min_duration_s <= self.duration_s <= self.max_duration_s:
            raise TaskSpecError(
                "human video duration must be "
                f"{self.min_duration_s:g}-{self.max_duration_s:g}s, "
                f"got {self.duration_s:.3f}s"
            )
        metadata = self.text_metadata
        if metadata is not None:
            if not isinstance(metadata, str) or not metadata.strip():
                raise TaskSpecError("human video text_metadata must be non-empty")
            metadata = metadata.strip()
            if len(metadata.encode("utf-8")) > MAX_LANGUAGE_BYTES:
                raise TaskSpecError("human video text_metadata is too long")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "provenance", _require_provenance(self.provenance))
        object.__setattr__(self, "text_metadata", metadata)

    @property
    def duration_s(self) -> float:
        return self.frames[-1].timestamp_s - self.frames[0].timestamp_s

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.frames[0].rgb.shape[0]), int(self.frames[0].rgb.shape[1])

    @property
    def fingerprint(self) -> str:
        digest = blake2b(digest_size=16)
        digest.update(TaskSpecKind.HUMAN_VIDEO.value.encode("ascii"))
        if self.text_metadata is not None:
            digest.update(self.text_metadata.encode("utf-8"))
        for frame in self.frames:
            digest.update(np.float64(frame.timestamp_s).tobytes())
            digest.update(np.asarray(frame.rgb.shape, dtype=np.int32).tobytes())
            digest.update(frame.rgb.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LanguagePrompt:
    """Bounded text instruction with no robot observation fields."""

    text: str
    provenance: TaskSpecProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise TaskSpecError("language prompt text must be non-empty")
        text = self.text.strip()
        if len(text.encode("utf-8")) > MAX_LANGUAGE_BYTES:
            raise TaskSpecError(
                f"language prompt must be at most {MAX_LANGUAGE_BYTES} UTF-8 bytes"
            )
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "provenance", _require_provenance(self.provenance))

    @property
    def fingerprint(self) -> str:
        digest = blake2b(digest_size=16)
        digest.update(TaskSpecKind.LANGUAGE.value.encode("ascii"))
        digest.update(self.text.encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RobotEpisodePrompt:
    """Explicit wrapper for the existing robot sensorimotor prompt."""

    prompt: PhysicalPrompt
    provenance: TaskSpecProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, PhysicalPrompt):
            raise TaskSpecError("robot episode prompt must wrap PhysicalPrompt")
        object.__setattr__(self, "provenance", _require_provenance(self.provenance))

    @property
    def fingerprint(self) -> str:
        return self.prompt.fingerprint


TaskSpec = RobotEpisodePrompt | HumanVideoPrompt | LanguagePrompt


class LanguageTokenizer(Protocol):
    def tokenize(self, text: str) -> tuple[int, ...]: ...


class LanguageEncoder(Protocol):
    def encode(self, token_ids: tuple[int, ...], *, width: int) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class Utf8Tokenizer:
    """Dependency-free byte tokenizer for the compact offline interface."""

    max_tokens: int = MAX_LANGUAGE_BYTES

    def tokenize(self, text: str) -> tuple[int, ...]:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise TaskSpecError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise TaskSpecError("max_tokens must be positive")
        if not isinstance(text, str) or not text.strip():
            raise TaskSpecError("language prompt text must be non-empty")
        encoded = text.strip().encode("utf-8")
        if len(encoded) > self.max_tokens:
            raise TaskSpecError(
                f"language prompt exceeds tokenizer limit {self.max_tokens}"
            )
        return tuple(value + _BYTE_TOKEN_OFFSET for value in encoded)


@dataclass(frozen=True, slots=True)
class DeterministicLanguageEncoder:
    """Fixed smoke-test encoder; it does not claim learned language semantics."""

    def encode(
        self,
        token_ids: tuple[int, ...],
        *,
        width: int,
    ) -> NDArray[np.float32]:
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise TaskSpecError("language embedding width must be a positive integer")
        if not token_ids or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not _BYTE_TOKEN_OFFSET
            <= token_id
            <= 255 + _BYTE_TOKEN_OFFSET
            for token_id in token_ids
        ):
            raise TaskSpecError("language token IDs must be non-empty byte tokens")

        tokens = np.asarray(token_ids, dtype=np.float64).reshape(-1, 1)
        positions = np.arange(1, len(token_ids) + 1, dtype=np.float64).reshape(-1, 1)
        dimensions = np.arange(1, width + 1, dtype=np.float64).reshape(1, -1)
        phase = (tokens * dimensions + positions) / _BYTE_TOKEN_SCALE
        embedding = np.mean(
            np.sin(phase) + np.cos(phase * _SECONDARY_PHASE_SCALE),
            axis=0,
        )
        norm = float(np.linalg.norm(embedding))
        if not isfinite(norm) or norm <= 0:
            raise TaskSpecError("language encoder produced an invalid embedding")
        result = np.asarray(embedding / norm, dtype=np.float32)
        result.flags.writeable = False
        return result


def task_spec_kind(value: TaskSpec) -> TaskSpecKind:
    if isinstance(value, RobotEpisodePrompt):
        return TaskSpecKind.ROBOT_EPISODE
    if isinstance(value, HumanVideoPrompt):
        return TaskSpecKind.HUMAN_VIDEO
    if isinstance(value, LanguagePrompt):
        return TaskSpecKind.LANGUAGE
    raise TaskSpecError(f"unsupported task specification: {type(value).__name__}")


__all__ = [
    "DeterministicLanguageEncoder",
    "HumanVideoFrame",
    "HumanVideoPrompt",
    "LanguageEncoder",
    "LanguagePrompt",
    "LanguageTokenizer",
    "RobotEpisodePrompt",
    "TaskSpec",
    "TaskSpecError",
    "TaskSpecKind",
    "TaskSpecProvenance",
    "Utf8Tokenizer",
    "task_spec_kind",
]
