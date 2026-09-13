"""Typed, immutable contracts for two-wrist sensorimotor prompting."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .constants import ACTION_DIM, PRIMARY_CAMERA_KEYS

RGBImage = NDArray[np.uint8]
FloatVector = NDArray[np.float32]


class ContractError(ValueError):
    """Raised when a robot, dataset, or policy value violates a public contract."""


def _readonly_float32(
    value: NDArray[np.floating] | list[float] | tuple[float, ...], *, name: str
) -> FloatVector:
    array = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains NaN or infinity")
    array.flags.writeable = False
    return array


def _readonly_rgb(value: NDArray[np.generic], *, name: str) -> RGBImage:
    array = np.array(value, copy=True)
    if array.dtype != np.uint8:
        raise ContractError(f"{name} must use uint8 RGB pixels, got {array.dtype}")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ContractError(f"{name} must have shape [H, W, 3], got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ContractError(f"{name} is too small: {array.shape}")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class SensorimotorFrame:
    """One synchronized policy-rate observation and its executed action.

    Extra camera streams may be recorded, but only the two canonical wrist views
    are required by and exposed to the core benchmark policy.
    """

    timestamp_s: float
    images: Mapping[str, RGBImage]
    joint_position: FloatVector
    executed_action: FloatVector | None = None
    image_timestamps_s: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ContractError("timestamp_s must be finite and non-negative")

        missing = tuple(key for key in PRIMARY_CAMERA_KEYS if key not in self.images)
        if missing:
            raise ContractError(f"missing required wrist camera(s): {missing}")

        copied_images = {
            key: _readonly_rgb(value, name=f"images[{key!r}]")
            for key, value in self.images.items()
        }
        primary_shapes = {copied_images[key].shape for key in PRIMARY_CAMERA_KEYS}
        if len(primary_shapes) != 1:
            raise ContractError("left and right wrist images must share one resolution")

        state = _readonly_float32(self.joint_position, name="joint_position")
        if state.shape != (ACTION_DIM,):
            raise ContractError(
                f"joint_position must have shape ({ACTION_DIM},), got {state.shape}"
            )

        action = self.executed_action
        if action is not None:
            action = _readonly_float32(action, name="executed_action")
            if action.shape != (ACTION_DIM,):
                raise ContractError(
                    f"executed_action must have shape ({ACTION_DIM},), got {action.shape}"
                )

        image_timestamps = self.image_timestamps_s
        if image_timestamps is not None:
            actual_timestamp_keys = set(image_timestamps)
            expected_timestamp_keys = set(copied_images)
            if actual_timestamp_keys != expected_timestamp_keys:
                missing_timestamps = tuple(
                    sorted(expected_timestamp_keys - actual_timestamp_keys)
                )
                extra_timestamps = tuple(
                    sorted(actual_timestamp_keys - expected_timestamp_keys)
                )
                raise ContractError(
                    "image timestamp keys must exactly match image keys; "
                    f"missing={missing_timestamps}, extra={extra_timestamps}"
                )
            copied_timestamps: dict[str, float] = {}
            for key, value in image_timestamps.items():
                if isinstance(value, bool):
                    raise ContractError(
                        f"image_timestamps_s[{key!r}] must be a timestamp"
                    )
                timestamp = float(value)
                if not isfinite(timestamp) or timestamp < 0:
                    raise ContractError(
                        f"image_timestamps_s[{key!r}] must be finite and non-negative"
                    )
                copied_timestamps[key] = timestamp
            image_timestamps = MappingProxyType(copied_timestamps)

        object.__setattr__(self, "images", MappingProxyType(copied_images))
        object.__setattr__(self, "joint_position", state)
        object.__setattr__(self, "executed_action", action)
        object.__setattr__(self, "image_timestamps_s", image_timestamps)

    @property
    def primary_images(self) -> tuple[RGBImage, RGBImage]:
        return tuple(self.images[key] for key in PRIMARY_CAMERA_KEYS)  # type: ignore[return-value]

    @property
    def primary_image_skew_s(self) -> float | None:
        if self.image_timestamps_s is None:
            return None
        left, right = (self.image_timestamps_s[key] for key in PRIMARY_CAMERA_KEYS)
        return abs(left - right)


@dataclass(frozen=True, slots=True)
class PhysicalPrompt:
    """A frozen 3-12 second sensorimotor example used without weight updates."""

    frames: tuple[SensorimotorFrame, ...]
    min_duration_s: float = 3.0
    max_duration_s: float = 12.0

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if len(frames) < 2:
            raise ContractError("a physical prompt needs at least two frames")
        if any(frame.executed_action is None for frame in frames):
            raise ContractError("every prompt frame must include the executed action")
        if any(
            current.timestamp_s <= previous.timestamp_s
            for previous, current in zip(frames, frames[1:])
        ):
            raise ContractError("prompt timestamps must be strictly increasing")
        if not 0 < self.min_duration_s <= self.max_duration_s:
            raise ContractError("invalid prompt duration bounds")
        if not self.min_duration_s <= self.duration_s <= self.max_duration_s:
            raise ContractError(
                f"prompt duration must be {self.min_duration_s:g}-{self.max_duration_s:g}s, "
                f"got {self.duration_s:.3f}s"
            )
        object.__setattr__(self, "frames", frames)

    @property
    def duration_s(self) -> float:
        return self.frames[-1].timestamp_s - self.frames[0].timestamp_s

    @property
    def fingerprint(self) -> str:
        digest = blake2b(digest_size=16)
        for frame in self.frames:
            digest.update(np.float64(frame.timestamp_s).tobytes())
            digest.update(frame.joint_position.tobytes())
            assert frame.executed_action is not None
            digest.update(frame.executed_action.tobytes())
            for image in frame.primary_images:
                digest.update(np.asarray(image.shape, dtype=np.int32).tobytes())
                digest.update(image.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """A fixed-rate sequence of absolute bimanual joint targets."""

    target_joint_position: NDArray[np.float32]
    dt_s: float
    created_at_s: float

    def __post_init__(self) -> None:
        targets = _readonly_float32(
            self.target_joint_position, name="target_joint_position"
        )
        if targets.ndim != 2 or targets.shape[1] != ACTION_DIM or targets.shape[0] < 1:
            raise ContractError(
                f"target_joint_position must have shape [H, {ACTION_DIM}], got {targets.shape}"
            )
        if not isfinite(self.dt_s) or self.dt_s <= 0:
            raise ContractError("dt_s must be finite and positive")
        if not isfinite(self.created_at_s) or self.created_at_s < 0:
            raise ContractError("created_at_s must be finite and non-negative")
        object.__setattr__(self, "target_joint_position", targets)

    @property
    def horizon(self) -> int:
        return int(self.target_joint_position.shape[0])
