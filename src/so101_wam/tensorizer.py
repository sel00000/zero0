"""Tensor bridge from GEN-1.5 context snapshots to CompactWAM inputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from .constants import ACTION_DIM, PRIMARY_CAMERA_COUNT, PRIMARY_CAMERA_KEYS
from .context import ContextSnapshot
from .contracts import SensorimotorFrame
from .task_specs import (
    HumanVideoPrompt,
    LanguagePrompt,
    RobotEpisodePrompt,
    TaskSpec,
    TaskSpecKind,
    TaskSpecProvenance,
    task_spec_kind,
)
from .vision import compact_rgb_image


class TensorizerError(ValueError):
    """Raised when a context snapshot cannot be packed for CompactWAM."""


@dataclass(frozen=True, slots=True)
class CompactWAMBatch:
    """Immutable single-sample batch matching ``CompactWAM.forward``."""

    prompt_images: Tensor
    prompt_proprio: Tensor
    prompt_actions: Tensor
    live_images: Tensor
    live_proprio: Tensor
    live_actions: Tensor
    prompt_mask: Tensor

    def as_kwargs(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                "prompt_images": self.prompt_images,
                "prompt_proprio": self.prompt_proprio,
                "prompt_actions": self.prompt_actions,
                "live_images": self.live_images,
                "live_proprio": self.live_proprio,
                "live_actions": self.live_actions,
                "prompt_mask": self.prompt_mask,
            }
        )


@dataclass(frozen=True, slots=True)
class TaskSpecTensorBatch:
    """Model tensors plus auditable task-spec identity."""

    batch: CompactWAMBatch
    kind: TaskSpecKind
    fingerprint: str
    provenance: TaskSpecProvenance
    language_text: str | None


def _require_executed_action(frame: SensorimotorFrame, *, segment: str, index: int) -> np.ndarray:
    if frame.executed_action is None:
        raise TensorizerError(f"{segment} frame {index} is missing executed_action")
    return frame.executed_action


def _require_exact_primary_cameras(frame: SensorimotorFrame, *, segment: str, index: int) -> None:
    missing = tuple(key for key in PRIMARY_CAMERA_KEYS if key not in frame.images)
    if missing:
        raise TensorizerError(
            f"{segment} frame {index} is missing required wrist camera(s): {missing}"
        )


def _pack_frames(
    frames: tuple[SensorimotorFrame, ...],
    *,
    segment: str,
    resolution: tuple[int, int] | None,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, int]]:
    if not frames:
        raise TensorizerError(f"{segment} segment must contain at least one frame")

    images: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    expected_resolution = resolution

    for index, frame in enumerate(frames):
        _require_exact_primary_cameras(frame, segment=segment, index=index)
        left, right = frame.primary_images
        frame_resolution = (int(left.shape[0]), int(left.shape[1]))
        if right.shape[:2] != frame_resolution:
            raise TensorizerError(f"{segment} frame {index} wrist image resolutions do not match")
        if expected_resolution is None:
            expected_resolution = frame_resolution
        elif frame_resolution != expected_resolution:
            raise TensorizerError(
                f"{segment} frame {index} resolution {frame_resolution} does not match {expected_resolution}"
            )

        stacked = np.stack(
            [compact_rgb_image(left), compact_rgb_image(right)],
            axis=0,
        )
        images.append(np.transpose(stacked, (0, 3, 1, 2)))
        proprio.append(frame.joint_position)
        actions.append(_require_executed_action(frame, segment=segment, index=index))

    image_tensor = torch.from_numpy(np.stack(images, axis=0).copy()).unsqueeze(0)
    proprio_tensor = torch.from_numpy(np.stack(proprio, axis=0).astype(np.float32, copy=True)).unsqueeze(0)
    action_tensor = torch.from_numpy(np.stack(actions, axis=0).astype(np.float32, copy=True)).unsqueeze(0)

    if image_tensor.shape[2] != PRIMARY_CAMERA_COUNT:
        raise TensorizerError(f"{segment} images must contain exactly {PRIMARY_CAMERA_COUNT} wrist views")
    if proprio_tensor.shape[-1] != ACTION_DIM or action_tensor.shape[-1] != ACTION_DIM:
        raise TensorizerError(f"{segment} state/action tensors must have width {ACTION_DIM}")
    assert expected_resolution is not None
    return image_tensor, proprio_tensor, action_tensor, expected_resolution


def tensorize_context(snapshot: ContextSnapshot) -> CompactWAMBatch:
    """Pack one immutable context snapshot into model-ready PyTorch tensors."""

    if not snapshot.live_frames:
        raise TensorizerError("snapshot must contain at least one live frame")

    prompt_images, prompt_proprio, prompt_actions, resolution = _pack_frames(
        snapshot.prompt_frames,
        segment="prompt",
        resolution=None,
    )
    live_images, live_proprio, live_actions, _ = _pack_frames(
        snapshot.live_frames,
        segment="live",
        resolution=resolution,
    )
    prompt_mask = torch.ones((1, len(snapshot.prompt_frames)), dtype=torch.bool)

    return CompactWAMBatch(
        prompt_images=prompt_images,
        prompt_proprio=prompt_proprio,
        prompt_actions=prompt_actions,
        live_images=live_images,
        live_proprio=live_proprio,
        live_actions=live_actions,
        prompt_mask=prompt_mask,
    )


def _neutral_axis_tensors(
    value: np.ndarray | list[float] | tuple[float, ...],
    *,
    steps: int,
) -> tuple[Tensor, Tensor]:
    neutral = np.asarray(value, dtype=np.float32)
    if neutral.shape != (ACTION_DIM,):
        raise TensorizerError(
            f"neutral_axes must have shape ({ACTION_DIM},), got {neutral.shape}"
        )
    if not np.isfinite(neutral).all():
        raise TensorizerError("neutral_axes contains NaN or infinity")
    axes = torch.from_numpy(np.array(neutral, copy=True)).view(1, 1, -1)
    axes = axes.expand(1, steps, -1).clone()
    return axes, axes.clone()


def _pack_human_video(
    prompt: HumanVideoPrompt,
    *,
    resolution: tuple[int, int],
    neutral_axes: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    if prompt.resolution != resolution:
        raise TensorizerError(
            "human video resolution "
            f"{prompt.resolution} does not match live resolution {resolution}"
        )
    images = []
    for frame in prompt.frames:
        image = compact_rgb_image(frame.rgb)
        duplicated = np.stack((image, image), axis=0)
        images.append(np.transpose(duplicated, (0, 3, 1, 2)))
    image_tensor = torch.from_numpy(np.stack(images, axis=0).copy()).unsqueeze(0)
    proprio, actions = _neutral_axis_tensors(
        neutral_axes,
        steps=len(prompt.frames),
    )
    return image_tensor, proprio, actions


def _pack_language_placeholder(
    live_images: Tensor,
    *,
    neutral_axes: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    height, width = (int(value) for value in live_images.shape[-2:])
    images = torch.zeros(
        (1, 1, PRIMARY_CAMERA_COUNT, 3, height, width),
        dtype=torch.uint8,
    )
    proprio, actions = _neutral_axis_tensors(neutral_axes, steps=1)
    return images, proprio, actions


def tensorize_task_spec(
    task_spec: TaskSpec,
    live_frames: tuple[SensorimotorFrame, ...],
    *,
    neutral_axes: np.ndarray | list[float] | tuple[float, ...],
) -> TaskSpecTensorBatch:
    """Adapt one task modality to the compact model without source signal leakage."""

    kind = task_spec_kind(task_spec)
    live_images, live_proprio, live_actions, live_resolution = _pack_frames(
        tuple(live_frames),
        segment="live",
        resolution=None,
    )
    language_text: str | None = None
    if isinstance(task_spec, RobotEpisodePrompt):
        prompt_images, prompt_proprio, prompt_actions, _ = _pack_frames(
            task_spec.prompt.frames,
            segment="prompt",
            resolution=live_resolution,
        )
    elif isinstance(task_spec, HumanVideoPrompt):
        prompt_images, prompt_proprio, prompt_actions = _pack_human_video(
            task_spec,
            resolution=live_resolution,
            neutral_axes=neutral_axes,
        )
    elif isinstance(task_spec, LanguagePrompt):
        prompt_images, prompt_proprio, prompt_actions = (
            _pack_language_placeholder(
                live_images,
                neutral_axes=neutral_axes,
            )
        )
        language_text = task_spec.text
    else:
        raise TensorizerError(
            f"unsupported task specification: {type(task_spec).__name__}"
        )

    batch = CompactWAMBatch(
        prompt_images=prompt_images,
        prompt_proprio=prompt_proprio,
        prompt_actions=prompt_actions,
        live_images=live_images,
        live_proprio=live_proprio,
        live_actions=live_actions,
        prompt_mask=torch.ones(
            (1, prompt_images.shape[1]),
            dtype=torch.bool,
        ),
    )
    return TaskSpecTensorBatch(
        batch=batch,
        kind=kind,
        fingerprint=task_spec.fingerprint,
        provenance=task_spec.provenance,
        language_text=language_text,
    )


__all__ = [
    "CompactWAMBatch",
    "TaskSpecTensorBatch",
    "TensorizerError",
    "tensorize_context",
    "tensorize_task_spec",
]
