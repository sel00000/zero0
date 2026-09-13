"""Compact PyTorch WAM components for two-wrist SO-101 tests.

This module is a small research/test model, not a trained policy.  It encodes
exactly two RGB wrist views plus aligned 12-axis proprio/action signals, uses a
causal temporal transformer over physical prompt and live context tokens,
predicts future dual-wrist latents, then decodes an action chunk through an
inverse-dynamics head.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .constants import ACTION_DIM, PRIMARY_CAMERA_COUNT


class ModelContractError(ValueError):
    """Raised when model tensors violate the compact WAM input contract."""


class ActionDecoder(StrEnum):
    LEGACY_MEAN = "legacy_mean"
    MEAN_REPEAT_CONTROL = "mean_repeat_control"
    ORDERED_CONCAT = "ordered_concat"


class ActionRangeConstraint(StrEnum):
    UNBOUNDED = "unbounded"
    AFFINE_TANH = "affine_tanh"
    NORMALIZED_CLAMP = "normalized_clamp"


@dataclass(frozen=True, slots=True)
class WAMLossWeights:
    future_latent: float = 1.0
    action: float = 1.0
    ifp: float = 0.25


@dataclass(frozen=True, slots=True)
class PromptEncoding:
    """Reusable prompt tokens before causal temporal attention."""

    tokens: Tensor
    padding_mask: Tensor
    prompt_steps: int
    image_shape: tuple[int, int, int]
    model_id: int


def _require_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ModelContractError(f"{name} must be a torch.Tensor")
    return value


def _check_image_batch(images: Tensor, *, name: str) -> None:
    if images.ndim != 6:
        raise ModelContractError(f"{name} must have shape [B, T, V=2, C=3, H, W], got {tuple(images.shape)}")
    if images.shape[2] != PRIMARY_CAMERA_COUNT:
        raise ModelContractError(
            f"{name} must contain exactly {PRIMARY_CAMERA_COUNT} wrist views, got {images.shape[2]}"
        )
    if images.shape[3] != 3:
        raise ModelContractError(f"{name} must use RGB images with C=3, got C={images.shape[3]}")
    if images.shape[4] < 8 or images.shape[5] < 8:
        raise ModelContractError(f"{name} images must be at least 8x8, got {tuple(images.shape[4:])}")


def _check_axis_tensor(value: Tensor, *, name: str, ndim: int) -> None:
    if value.ndim != ndim or value.shape[-1] != ACTION_DIM:
        shape = tuple(value.shape)
        prefix = "B, " if ndim == 2 else "B, T, "
        raise ModelContractError(f"{name} must have shape [{prefix}{ACTION_DIM}], got {shape}")


def _check_aligned_axis_sequence(value: Tensor, *, name: str, batch: int, steps: int) -> None:
    _check_axis_tensor(value, name=name, ndim=3)
    if value.shape[0] != batch or value.shape[1] != steps:
        raise ModelContractError(f"{name} must align to image shape [B, T], got {tuple(value.shape[:2])}")


def _check_prompt_mask(prompt_mask: Tensor, *, batch: int, prompt_steps: int) -> None:
    if prompt_mask.shape != (batch, prompt_steps):
        raise ModelContractError(
            f"prompt_mask must have shape [B, prompt_T]={batch, prompt_steps}, got {tuple(prompt_mask.shape)}"
        )
    if prompt_mask.dtype is not torch.bool:
        raise ModelContractError("prompt_mask must be a bool tensor")
    if prompt_steps > 1:
        resurrected = (~prompt_mask[:, :-1]) & prompt_mask[:, 1:]
        if bool(resurrected.any()):
            raise ModelContractError("prompt_mask must be left-packed causal order: valid prompt frames first")
    if bool((prompt_mask.sum(dim=1) == 0).any()):
        raise ModelContractError("each batch item needs at least one valid prompt frame")


def _as_float_images(images: Tensor) -> Tensor:
    if images.dtype == torch.uint8:
        return images.float().div(255.0)
    if not images.is_floating_point():
        raise ModelContractError(f"image tensors must be uint8 or floating point, got {images.dtype}")
    return images.float()


class DualWristEncoder(nn.Module):
    """Shared CNN encoder with learned left/right view and prompt/live embeddings."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.view_embedding = nn.Embedding(PRIMARY_CAMERA_COUNT, latent_dim)
        self.segment_embedding = nn.Embedding(2, latent_dim)

    def forward(self, images: Tensor, *, segment_id: int) -> Tensor:
        _check_image_batch(images, name="images")
        if segment_id not in (0, 1):
            raise ModelContractError("segment_id must be 0 for prompt or 1 for live context")

        images = _as_float_images(images)
        batch, steps, views, channels, height, width = images.shape
        flat = images.reshape(batch * steps * views, channels, height, width)
        encoded = self.cnn(flat).reshape(batch, steps, views, self.latent_dim)

        view_ids = torch.arange(views, device=images.device)
        segment_ids = torch.full((1,), segment_id, dtype=torch.long, device=images.device)
        return encoded + self.view_embedding(view_ids).view(1, 1, views, -1) + self.segment_embedding(segment_ids).view(
            1, 1, 1, -1
        )


class FutureLatentPredictor(nn.Module):
    """Predicts future latent tokens for both wrists from the causal context state."""

    def __init__(self, latent_dim: int, future_steps: int) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.query = nn.Parameter(torch.zeros(future_steps, PRIMARY_CAMERA_COUNT, latent_dim))
        self.proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, context_state: Tensor) -> Tensor:
        base = self.proj(context_state).view(context_state.shape[0], 1, 1, -1)
        return base + self.query.view(1, self.future_steps, PRIMARY_CAMERA_COUNT, -1)


class InverseDynamicsActionHead(nn.Module):
    """Action decoder factored away from raw prompt embeddings."""

    def __init__(
        self,
        latent_dim: int,
        action_horizon: int,
        action_history_steps: int,
        *,
        future_steps: int | None = None,
        action_decoder: ActionDecoder = ActionDecoder.LEGACY_MEAN,
    ) -> None:
        super().__init__()
        if not isinstance(action_decoder, ActionDecoder):
            raise ModelContractError("action_decoder must be an ActionDecoder")
        if action_decoder is not ActionDecoder.LEGACY_MEAN and (
            not isinstance(future_steps, int)
            or isinstance(future_steps, bool)
            or future_steps < 1
        ):
            raise ModelContractError("fixed decoder requires positive future_steps")

        self.action_horizon = action_horizon
        self.action_history_steps = action_history_steps
        self.action_decoder = action_decoder
        self._future_steps = future_steps
        self._latent_dim = latent_dim
        slots = 1 if action_decoder is ActionDecoder.LEGACY_MEAN else cast(int, future_steps)
        input_dim = slots * latent_dim * PRIMARY_CAMERA_COUNT + ACTION_DIM + action_history_steps * ACTION_DIM
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, action_horizon * ACTION_DIM),
        )

    def forward(self, future_latents: Tensor, current_proprio: Tensor, action_history: Tensor) -> Tensor:
        if future_latents.ndim != 4 or future_latents.shape[2] != PRIMARY_CAMERA_COUNT:
            raise ModelContractError(
                "future_latents must have shape [B, future_T, V=2, latent_dim], "
                f"got {tuple(future_latents.shape)}"
            )
        _check_axis_tensor(current_proprio, name="current_proprio", ndim=2)
        _check_axis_tensor(action_history, name="action_history", ndim=3)
        if action_history.shape[1] != self.action_history_steps:
            raise ModelContractError(
                f"action_history must contain {self.action_history_steps} steps, got {action_history.shape[1]}"
            )
        if future_latents.shape[0] != current_proprio.shape[0] or action_history.shape[0] != current_proprio.shape[0]:
            raise ModelContractError("future_latents, current_proprio, and action_history must share batch size")

        if future_latents.shape[1] < 1:
            raise ModelContractError("latents must contain at least one future step")
        if self.action_decoder is ActionDecoder.LEGACY_MEAN and future_latents.shape[3] != self._latent_dim:
            raise ModelContractError("latents must match configured width")
        if self.action_decoder is not ActionDecoder.LEGACY_MEAN and (
            future_latents.shape[1] != self._future_steps
            or future_latents.shape[3] != self._latent_dim
        ):
            raise ModelContractError("latents must match configured future steps and width")

        if self.action_decoder is ActionDecoder.LEGACY_MEAN:
            latent_summary = future_latents.mean(dim=1).flatten(start_dim=1)
        elif self.action_decoder is ActionDecoder.MEAN_REPEAT_CONTROL:
            latent_summary = future_latents.mean(dim=1, keepdim=True).expand_as(future_latents).flatten(start_dim=1)
        else:
            latent_summary = future_latents.flatten(start_dim=1)
        history = action_history.flatten(start_dim=1)
        action = self.net(torch.cat([latent_summary, current_proprio, history], dim=-1))
        return action.view(current_proprio.shape[0], self.action_horizon, ACTION_DIM)


class CompactWAM(nn.Module):
    """Small causal world-action model for dual-wrist prompt-conditioned tests."""

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        future_steps: int = 3,
        action_horizon: int = 10,
        action_history_steps: int = 4,
        ifp_steps: int = 2,
        max_context_steps: int = 300,
        action_decoder: ActionDecoder = ActionDecoder.LEGACY_MEAN,
        action_range_constraint: ActionRangeConstraint = ActionRangeConstraint.UNBOUNDED,
        action_lower: Sequence[float] | Tensor | None = None,
        action_upper: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(action_decoder, ActionDecoder):
            raise ModelContractError("action_decoder must be an ActionDecoder")
        if not isinstance(action_range_constraint, ActionRangeConstraint):
            raise ModelContractError("action_range_constraint must be an ActionRangeConstraint")
        if latent_dim % transformer_heads != 0:
            raise ModelContractError("latent_dim must be divisible by transformer_heads")
        self.latent_dim = latent_dim
        self.future_steps = future_steps
        self.action_horizon = action_horizon
        self.action_history_steps = action_history_steps
        self.ifp_steps = ifp_steps
        self.max_context_steps = max_context_steps
        self.action_range_constraint = action_range_constraint
        if max_context_steps < 300:
            raise ModelContractError("max_context_steps must support at least 300 policy steps")

        # Joint positions and targets share the same calibrated SO-101 units.
        # Identity defaults preserve old/random checkpoints, while offline
        # training installs dataset statistics that travel in ``state_dict``.
        self.register_buffer("axis_mean", torch.zeros(ACTION_DIM, dtype=torch.float32))
        self.register_buffer("axis_scale", torch.ones(ACTION_DIM, dtype=torch.float32))
        self._configure_action_bounds(action_lower, action_upper)

        self.encoder = DualWristEncoder(latent_dim)
        self.sensorimotor_projection = nn.Sequential(
            nn.LayerNorm(ACTION_DIM * 2),
            nn.Linear(ACTION_DIM * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.temporal_position_embedding = nn.Embedding(max_context_steps, latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.future_predictor = FutureLatentPredictor(latent_dim, future_steps)
        self.action_head = InverseDynamicsActionHead(
            latent_dim,
            action_horizon,
            action_history_steps,
            future_steps=future_steps,
            action_decoder=action_decoder,
        )
        self.ifp_head = nn.Linear(latent_dim, ifp_steps * PRIMARY_CAMERA_COUNT * latent_dim) if ifp_steps > 0 else None

    @property
    def action_decoder(self) -> ActionDecoder:
        return self.action_head.action_decoder

    def _configure_action_bounds(
        self,
        action_lower: Sequence[float] | Tensor | None,
        action_upper: Sequence[float] | Tensor | None,
    ) -> None:
        if self.action_range_constraint is ActionRangeConstraint.UNBOUNDED:
            if action_lower is not None or action_upper is not None:
                raise ModelContractError("unbounded action range must not include action bounds")
            return

        if action_lower is None or action_upper is None:
            raise ModelContractError("bounded action range requires action_lower and action_upper")
        lower = torch.as_tensor(action_lower, dtype=torch.float64)
        upper = torch.as_tensor(action_upper, dtype=torch.float64)
        if lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,):
            raise ModelContractError(f"action bounds must have shape ({ACTION_DIM},)")
        if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
            raise ModelContractError("action bounds contain NaN or infinity")
        if not bool((lower < upper).all()):
            raise ModelContractError("action bounds must satisfy lower < upper")

        lower32 = self._round_lower_bound_inward(lower)
        upper32 = self._round_upper_bound_inward(upper)
        if not bool(torch.isfinite(lower32).all()) or not bool(torch.isfinite(upper32).all()):
            raise ModelContractError("action bounds exceed finite float32 range")
        if not bool(torch.isfinite(upper32 - lower32).all()):
            raise ModelContractError("action bounds produce non-finite range")
        if not bool((lower32 < upper32).all()):
            raise ModelContractError("action bounds must contain float32 interior")
        self.register_buffer("action_lower", lower.clone())
        self.register_buffer("action_upper", upper.clone())
        self.register_buffer("_action_lower_f32", lower32, persistent=False)
        self.register_buffer("_action_upper_f32", upper32, persistent=False)

    @staticmethod
    def _round_lower_bound_inward(lower: Tensor) -> Tensor:
        lower32 = lower.to(torch.float32)
        rounded_out = lower32.double() < lower
        return torch.where(
            rounded_out,
            torch.nextafter(lower32, torch.full_like(lower32, torch.inf)),
            lower32,
        )

    @staticmethod
    def _round_upper_bound_inward(upper: Tensor) -> Tensor:
        upper32 = upper.to(torch.float32)
        rounded_out = upper32.double() > upper
        return torch.where(
            rounded_out,
            torch.nextafter(upper32, torch.full_like(upper32, -torch.inf)),
            upper32,
        )

    def _axis_mean_buffer(self) -> Tensor:
        return cast(Tensor, self.axis_mean)

    def _axis_scale_buffer(self) -> Tensor:
        return cast(Tensor, self.axis_scale)

    def _action_lower_buffer(self) -> Tensor:
        return cast(Tensor, self.action_lower)

    def _action_upper_buffer(self) -> Tensor:
        return cast(Tensor, self.action_upper)

    def _action_lower_f32_buffer(self) -> Tensor:
        return cast(Tensor, self._action_lower_f32)

    def _action_upper_f32_buffer(self) -> Tensor:
        return cast(Tensor, self._action_upper_f32)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.action_range_constraint is not ActionRangeConstraint.UNBOUNDED:
            self._refresh_action_bounds()
        return result

    def _refresh_action_bounds(self) -> None:
        lower = self._action_lower_buffer()
        upper = self._action_upper_buffer()
        lower32 = self._round_lower_bound_inward(lower)
        upper32 = self._round_upper_bound_inward(upper)
        if not bool(torch.isfinite(lower32).all()) or not bool(torch.isfinite(upper32).all()):
            raise ModelContractError("action bounds exceed finite float32 range")
        if not bool(torch.isfinite(upper32 - lower32).all()):
            raise ModelContractError("action bounds produce non-finite range")
        if not bool((lower32 < upper32).all()):
            raise ModelContractError("action bounds must contain float32 interior")
        self._action_lower_f32_buffer().copy_(lower32)
        self._action_upper_f32_buffer().copy_(upper32)

    def _validate_prompt_inputs(
        self,
        prompt_images: Tensor,
        prompt_proprio: Tensor,
        prompt_actions: Tensor,
        prompt_mask: Tensor | None,
    ) -> Tensor:
        _check_image_batch(prompt_images, name="prompt_images")
        if prompt_images.shape[1] > self.max_context_steps:
            raise ModelContractError(
                f"prompt must fit max_context_steps={self.max_context_steps}, "
                f"got {prompt_images.shape[1]}"
            )
        _check_aligned_axis_sequence(
            prompt_proprio,
            name="prompt_proprio",
            batch=prompt_images.shape[0],
            steps=prompt_images.shape[1],
        )
        _check_aligned_axis_sequence(
            prompt_actions,
            name="prompt_actions",
            batch=prompt_images.shape[0],
            steps=prompt_images.shape[1],
        )

        if prompt_mask is None:
            prompt_mask = torch.ones(
                prompt_images.shape[0],
                prompt_images.shape[1],
                dtype=torch.bool,
                device=prompt_images.device,
            )
        else:
            prompt_mask = _require_tensor(
                prompt_mask,
                name="prompt_mask",
            ).to(device=prompt_images.device)
        _check_prompt_mask(
            prompt_mask,
            batch=prompt_images.shape[0],
            prompt_steps=prompt_images.shape[1],
        )
        return prompt_mask

    def _validate_cached_live_inputs(
        self,
        prompt: PromptEncoding,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
    ) -> None:
        if not isinstance(prompt, PromptEncoding):
            raise ModelContractError("prompt encoding must use PromptEncoding")
        if prompt.model_id != id(self):
            raise ModelContractError("prompt encoding belongs to another model")
        if (
            prompt.tokens.ndim != 3
            or prompt.tokens.shape[1]
            != prompt.prompt_steps * PRIMARY_CAMERA_COUNT
            or prompt.tokens.shape[2] != self.latent_dim
        ):
            raise ModelContractError("prompt encoding token shape is invalid")
        if prompt.padding_mask.shape != prompt.tokens.shape[:2]:
            raise ModelContractError("prompt encoding padding shape is invalid")
        if prompt.padding_mask.dtype is not torch.bool:
            raise ModelContractError("prompt encoding padding must be bool")
        if prompt.padding_mask.device != prompt.tokens.device:
            raise ModelContractError("prompt encoding tensors must share one device")
        if not bool(torch.isfinite(prompt.tokens).all()):
            raise ModelContractError("prompt encoding contains NaN or infinity")

        _check_image_batch(live_images, name="live_images")
        if live_images.shape[1] < 1:
            raise ModelContractError(
                "live_images must contain at least one live frame"
            )
        if prompt.tokens.shape[0] != live_images.shape[0]:
            raise ModelContractError("prompt encoding and live_images must share batch size")
        if tuple(live_images.shape[3:]) != prompt.image_shape:
            raise ModelContractError(
                "prompt encoding and live_images must share [C, H, W]"
            )
        if live_images.device != prompt.tokens.device:
            raise ModelContractError(
                "prompt encoding and live tensors must share one device"
            )
        if live_images.shape[1] < self.action_history_steps:
            raise ModelContractError(
                f"live_actions must contain at least {self.action_history_steps} steps for action history, "
                f"got {live_images.shape[1]}"
            )
        if prompt.prompt_steps + live_images.shape[1] > self.max_context_steps:
            raise ModelContractError(
                f"prompt plus live context must fit max_context_steps={self.max_context_steps}, "
                f"got {prompt.prompt_steps + live_images.shape[1]}"
            )
        _check_aligned_axis_sequence(
            live_proprio,
            name="live_proprio",
            batch=live_images.shape[0],
            steps=live_images.shape[1],
        )
        _check_aligned_axis_sequence(
            live_actions,
            name="live_actions",
            batch=live_images.shape[0],
            steps=live_images.shape[1],
        )

    def _fuse_sensorimotor(self, latents: Tensor, proprio: Tensor, actions: Tensor, *, start_position: int) -> Tensor:
        sensorimotor = self.sensorimotor_projection(
            torch.cat(
                [self.normalize_axes(proprio), self.normalize_axes(actions)],
                dim=-1,
            )
        )
        steps = latents.shape[1]
        positions = torch.arange(start_position, start_position + steps, device=latents.device)
        positional = self.temporal_position_embedding(positions)
        return latents + sensorimotor.unsqueeze(2) + positional.view(1, steps, 1, -1)

    def _action_inputs_from_live(self, live_proprio: Tensor, live_actions: Tensor) -> tuple[Tensor, Tensor]:
        current_proprio = self.normalize_axes(live_proprio[:, -1, :])
        action_history = self.normalize_axes(
            live_actions[:, -self.action_history_steps :, :]
        )
        return current_proprio, action_history

    def _apply_task_conditioning(
        self,
        context_state: Tensor,
        task_conditioning: Tensor | None,
    ) -> Tensor:
        if task_conditioning is None:
            return context_state
        task_conditioning = _require_tensor(
            task_conditioning,
            name="task_conditioning",
        )
        expected_shape = (context_state.shape[0], self.latent_dim)
        if task_conditioning.shape != expected_shape:
            raise ModelContractError(
                "task_conditioning must have shape "
                f"{expected_shape}, got {tuple(task_conditioning.shape)}"
            )
        if not task_conditioning.is_floating_point():
            raise ModelContractError("task_conditioning must be floating point")
        if task_conditioning.device != context_state.device:
            raise ModelContractError(
                "task_conditioning must be on the model input device"
            )
        if not bool(torch.isfinite(task_conditioning).all()):
            raise ModelContractError("task_conditioning contains NaN or infinity")
        return context_state + task_conditioning.float()

    def set_axis_normalization(self, mean: Tensor, scale: Tensor) -> None:
        """Install finite per-axis statistics used by training and inference."""

        axis_mean = self._axis_mean_buffer()
        axis_scale = self._axis_scale_buffer()
        mean = torch.as_tensor(mean, dtype=torch.float32, device=axis_mean.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=axis_scale.device)
        if mean.shape != (ACTION_DIM,) or scale.shape != (ACTION_DIM,):
            raise ModelContractError(
                f"axis normalization must have shape ({ACTION_DIM},)"
            )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise ModelContractError("axis normalization contains NaN or infinity")
        if bool((scale <= 0).any()):
            raise ModelContractError("axis normalization scale must be positive")
        axis_mean.copy_(mean)
        axis_scale.copy_(scale)

    def normalize_axes(self, value: Tensor) -> Tensor:
        """Normalize a tensor whose final dimension follows ``JOINT_KEYS``."""

        value = _require_tensor(value, name="axis tensor")
        if value.ndim < 1 or value.shape[-1] != ACTION_DIM:
            raise ModelContractError(
                f"axis tensor must end in width {ACTION_DIM}, got {tuple(value.shape)}"
            )
        return (value.float() - self._axis_mean_buffer()) / self._axis_scale_buffer()

    def denormalize_axes(self, value: Tensor) -> Tensor:
        """Restore calibrated SO-101 units from normalized model output."""

        value = _require_tensor(value, name="normalized axis tensor")
        if value.ndim < 1 or value.shape[-1] != ACTION_DIM:
            raise ModelContractError(
                "normalized axis tensor must end in width "
                f"{ACTION_DIM}, got {tuple(value.shape)}"
            )
        return value.float() * self._axis_scale_buffer() + self._axis_mean_buffer()

    def decode_actions(self, raw_head_output: Tensor) -> Tensor:
        """Map action-head output to native calibrated SO-101 units."""

        raw_head_output = _require_tensor(raw_head_output, name="raw action output")
        if raw_head_output.ndim < 1 or raw_head_output.shape[-1] != ACTION_DIM:
            raise ModelContractError(
                "raw action output must end in width "
                f"{ACTION_DIM}, got {tuple(raw_head_output.shape)}"
            )
        if self.action_range_constraint is ActionRangeConstraint.UNBOUNDED:
            return self.denormalize_axes(raw_head_output)

        if not bool(torch.isfinite(raw_head_output).all()):
            raise ModelContractError("raw action output contains NaN or infinity")
        raw32 = raw_head_output.float()
        if not bool(torch.isfinite(raw32).all()):
            raise ModelContractError("raw action output exceeds finite float32 range")

        lower = self._action_lower_f32_buffer().to(raw_head_output.device)
        upper = self._action_upper_f32_buffer().to(raw_head_output.device)
        action_range = upper - lower
        if not bool(torch.isfinite(action_range).all()):
            raise ModelContractError("action bounds produce non-finite range")
        if self.action_range_constraint is ActionRangeConstraint.NORMALIZED_CLAMP:
            actions = self.denormalize_axes(raw32)
            if not bool(torch.isfinite(actions).all()):
                raise ModelContractError("decoded native actions contain NaN or infinity")
            # Preserve endpoint gradients while stopping gradients outside bounds.
            actions = torch.where(actions < lower, lower, actions)
            return torch.where(actions > upper, upper, actions)

        midpoint = lower + action_range * 0.5
        half_range = action_range * 0.5
        actions = midpoint + torch.tanh(raw32) * half_range
        actions = torch.maximum(torch.minimum(actions, upper), lower)
        if not bool(torch.isfinite(actions).all()):
            raise ModelContractError("decoded actions contain NaN or infinity")
        return actions

    def encode_prompt(
        self,
        prompt_images: Tensor,
        prompt_proprio: Tensor,
        prompt_actions: Tensor,
        *,
        prompt_mask: Tensor | None = None,
    ) -> PromptEncoding:
        """Encode fixed prompt inputs once before temporal attention."""

        prompt_images = _require_tensor(prompt_images, name="prompt_images")
        prompt_proprio = _require_tensor(prompt_proprio, name="prompt_proprio")
        prompt_actions = _require_tensor(prompt_actions, name="prompt_actions")
        prompt_mask = self._validate_prompt_inputs(
            prompt_images,
            prompt_proprio,
            prompt_actions,
            prompt_mask,
        )
        latents = self._fuse_sensorimotor(
            self.encoder(prompt_images, segment_id=0),
            prompt_proprio,
            prompt_actions,
            start_position=0,
        )
        return PromptEncoding(
            tokens=latents.flatten(start_dim=1, end_dim=2),
            padding_mask=(~prompt_mask).repeat_interleave(
                PRIMARY_CAMERA_COUNT,
                dim=1,
            ),
            prompt_steps=int(prompt_images.shape[1]),
            image_shape=(
                int(prompt_images.shape[3]),
                int(prompt_images.shape[4]),
                int(prompt_images.shape[5]),
            ),
            model_id=id(self),
        )

    def encode_context_cached(
        self,
        prompt: PromptEncoding,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        task_conditioning: Tensor | None = None,
    ) -> Tensor:
        """Combine cached prompt tokens with one live causal context."""

        live_images = _require_tensor(live_images, name="live_images")
        live_proprio = _require_tensor(live_proprio, name="live_proprio")
        live_actions = _require_tensor(live_actions, name="live_actions")
        self._validate_cached_live_inputs(
            prompt,
            live_images,
            live_proprio,
            live_actions,
        )
        live_latents = self._fuse_sensorimotor(
            self.encoder(live_images, segment_id=1),
            live_proprio,
            live_actions,
            start_position=prompt.prompt_steps,
        )
        tokens = torch.cat(
            [
                prompt.tokens,
                live_latents.flatten(start_dim=1, end_dim=2),
            ],
            dim=1,
        )

        live_padding = torch.zeros(
            live_images.shape[0],
            live_images.shape[1] * PRIMARY_CAMERA_COUNT,
            dtype=torch.bool,
            device=live_images.device,
        )
        padding_mask = torch.cat([prompt.padding_mask, live_padding], dim=1)
        causal_mask = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        contextual = self.temporal(tokens, mask=causal_mask, src_key_padding_mask=padding_mask)

        live_token_count = live_images.shape[1] * PRIMARY_CAMERA_COUNT
        live_context = contextual[:, -live_token_count:, :]
        context_state = live_context[:, -PRIMARY_CAMERA_COUNT:, :].mean(dim=1)
        return self._apply_task_conditioning(context_state, task_conditioning)

    def encode_context(
        self,
        prompt_images: Tensor,
        prompt_proprio: Tensor,
        prompt_actions: Tensor,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        prompt_mask: Tensor | None = None,
        task_conditioning: Tensor | None = None,
    ) -> Tensor:
        prompt = self.encode_prompt(
            prompt_images,
            prompt_proprio,
            prompt_actions,
            prompt_mask=prompt_mask,
        )
        return self.encode_context_cached(
            prompt,
            live_images,
            live_proprio,
            live_actions,
            task_conditioning=task_conditioning,
        )

    def encode_context_features(
        self,
        prompt_images: Tensor,
        prompt_proprio: Tensor,
        prompt_actions: Tensor,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        *,
        prompt_mask: Tensor | None = None,
        task_conditioning: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the normal context plus one final-live summary per layer.

        The temporary hooks observe the unchanged Transformer execution. They
        are used only by the removable training auxiliary and add no state.
        """

        layer_outputs: list[Tensor] = []

        def capture_layer(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            layer_outputs.append(output)

        handles = [
            layer.register_forward_hook(capture_layer) for layer in self.temporal.layers
        ]
        try:
            context_state = self.encode_context(
                prompt_images,
                prompt_proprio,
                prompt_actions,
                live_images,
                live_proprio,
                live_actions,
                prompt_mask=prompt_mask,
                task_conditioning=task_conditioning,
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(layer_outputs) != len(self.temporal.layers):
            raise ModelContractError("failed to collect every temporal layer")
        features = tuple(
            output[:, -PRIMARY_CAMERA_COUNT:, :].mean(dim=1)
            for output in layer_outputs[:-1]
        ) + (context_state,)
        return context_state, torch.stack(features, dim=1)

    def _outputs_from_context(
        self,
        context_state: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        *,
        compute_ifp: bool,
    ) -> dict[str, Tensor | None]:
        future_latents = self.future_predictor(context_state)
        current_proprio, action_history = self._action_inputs_from_live(
            live_proprio.float(),
            live_actions.float(),
        )
        raw_actions = self.action_head(
            future_latents,
            current_proprio,
            action_history,
        )
        actions = self.decode_actions(raw_actions)

        ifp_latents: Tensor | None = None
        if self.training and compute_ifp and self.ifp_head is not None:
            ifp_latents = self.ifp_head(context_state).view(
                context_state.shape[0],
                self.ifp_steps,
                PRIMARY_CAMERA_COUNT,
                self.latent_dim,
            )

        return {
            "future_latents": future_latents,
            "actions": actions,
            "ifp_latents": ifp_latents,
        }

    def forward(
        self,
        prompt_images: Tensor,
        prompt_proprio: Tensor,
        prompt_actions: Tensor,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        *,
        prompt_mask: Tensor | None = None,
        task_conditioning: Tensor | None = None,
        compute_ifp: bool = True,
    ) -> dict[str, Tensor | None]:
        context_state = self.encode_context(
            prompt_images,
            prompt_proprio,
            prompt_actions,
            live_images,
            live_proprio,
            live_actions,
            prompt_mask=prompt_mask,
            task_conditioning=task_conditioning,
        )
        return self._outputs_from_context(
            context_state,
            live_proprio,
            live_actions,
            compute_ifp=compute_ifp,
        )

    def forward_with_context_features(
        self,
        prompt_images: Tensor,
        prompt_proprio: Tensor,
        prompt_actions: Tensor,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        *,
        prompt_mask: Tensor | None = None,
        task_conditioning: Tensor | None = None,
    ) -> tuple[dict[str, Tensor | None], Tensor]:
        """Run the main branch once and expose training-only layer features."""

        context_state, layer_features = self.encode_context_features(
            prompt_images,
            prompt_proprio,
            prompt_actions,
            live_images,
            live_proprio,
            live_actions,
            prompt_mask=prompt_mask,
            task_conditioning=task_conditioning,
        )
        outputs = self._outputs_from_context(
            context_state,
            live_proprio,
            live_actions,
            compute_ifp=False,
        )
        return outputs, layer_features

    @torch.no_grad()
    def infer_action(
        self,
        prompt_images: Tensor,
        prompt_proprio: Tensor,
        prompt_actions: Tensor,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        *,
        prompt_mask: Tensor | None = None,
        task_conditioning: Tensor | None = None,
    ) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return self(
                prompt_images,
                prompt_proprio,
                prompt_actions,
                live_images,
                live_proprio,
                live_actions,
                prompt_mask=prompt_mask,
                task_conditioning=task_conditioning,
                compute_ifp=False,
            )["actions"]  # type: ignore[return-value]
        finally:
            self.train(was_training)

    @torch.no_grad()
    def infer_action_cached(
        self,
        prompt: PromptEncoding,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        *,
        task_conditioning: Tensor | None = None,
    ) -> Tensor:
        """Infer with reusable prompt tokens and unchanged temporal attention."""

        return self.infer_outputs_cached(
            prompt,
            live_images,
            live_proprio,
            live_actions,
            task_conditioning=task_conditioning,
        )["actions"]  # type: ignore[return-value]

    @torch.no_grad()
    def infer_outputs_cached(
        self,
        prompt: PromptEncoding,
        live_images: Tensor,
        live_proprio: Tensor,
        live_actions: Tensor,
        *,
        task_conditioning: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Infer actions and future latents with reusable prompt tokens."""

        was_training = self.training
        self.eval()
        try:
            context_state = self.encode_context_cached(
                prompt,
                live_images,
                live_proprio,
                live_actions,
                task_conditioning=task_conditioning,
            )
            return self._outputs_from_context(
                context_state,
                live_proprio,
                live_actions,
                compute_ifp=False,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def encode_future_images(self, images: Tensor) -> Tensor:
        """Encode observed future frames with the training target branch."""

        was_training = self.training
        self.eval()
        try:
            return self.encoder(images, segment_id=1)
        finally:
            self.train(was_training)

    def loss(
        self,
        outputs: Mapping[str, Tensor | None],
        *,
        target_future_latents: Tensor,
        target_actions: Tensor,
        target_ifp_latents: Tensor | None = None,
        weights: WAMLossWeights = WAMLossWeights(),
        action_scale: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return wam_loss(
            outputs,
            target_future_latents=target_future_latents,
            target_actions=target_actions,
            target_ifp_latents=target_ifp_latents,
            weights=weights,
            action_scale=action_scale,
        )


class FusedIFP(nn.Module):
    """Training-only parallel future modules over fused main-branch features."""

    def __init__(self, model: CompactWAM, *, ifp_steps: int) -> None:
        super().__init__()
        if (
            not isinstance(ifp_steps, int)
            or isinstance(ifp_steps, bool)
            or ifp_steps < 1
        ):
            raise ModelContractError("fused IFP requires a positive ifp_steps")
        feature_layers = len(model.temporal.layers)
        input_dim = feature_layers * model.latent_dim
        self.ifp_steps = ifp_steps
        self.feature_layers = feature_layers
        self.latent_dim = model.latent_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model.latent_dim),
            nn.GELU(),
            nn.Linear(model.latent_dim, model.latent_dim),
        )
        self.queries = nn.Parameter(
            torch.zeros(ifp_steps, PRIMARY_CAMERA_COUNT, model.latent_dim)
        )
        template = model.temporal.layers[-1]
        self.future_modules = nn.ModuleList(
            deepcopy(template) for _ in range(ifp_steps)
        )

    def forward(self, layer_features: Tensor) -> Tensor:
        layer_features = _require_tensor(
            layer_features,
            name="layer_features",
        )
        expected = (self.feature_layers, self.latent_dim)
        if layer_features.ndim != 3 or tuple(layer_features.shape[1:]) != expected:
            raise ModelContractError(
                "layer_features must have shape "
                f"[B, {self.feature_layers}, {self.latent_dim}], "
                f"got {tuple(layer_features.shape)}"
            )
        if not layer_features.is_floating_point():
            raise ModelContractError("layer_features must be floating point")
        if not bool(torch.isfinite(layer_features).all()):
            raise ModelContractError("layer_features contains NaN or infinity")

        fused = self.fusion(layer_features.flatten(start_dim=1))
        predictions: list[Tensor] = []
        for index, module in enumerate(self.future_modules):
            queries = (
                self.queries[index]
                .unsqueeze(0)
                .expand(
                    layer_features.shape[0],
                    -1,
                    -1,
                )
            )
            tokens = torch.cat([fused.unsqueeze(1), queries], dim=1)
            predictions.append(module(tokens)[:, 1:, :])
        return torch.stack(predictions, dim=1)


def wam_loss(
    outputs: Mapping[str, Tensor | None],
    *,
    target_future_latents: Tensor,
    target_actions: Tensor,
    target_ifp_latents: Tensor | None = None,
    weights: WAMLossWeights = WAMLossWeights(),
    action_scale: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute future-latent, action-chunk, and optional training-only IFP loss."""

    future_latents = outputs["future_latents"]
    actions = outputs["actions"]
    if future_latents is None or actions is None:
        raise ModelContractError("outputs must include future_latents and actions")
    if target_future_latents.shape != future_latents.shape:
        raise ModelContractError(
            "target_future_latents must match future_latents shape, "
            f"got {tuple(target_future_latents.shape)} vs {tuple(future_latents.shape)}"
        )
    if target_actions.shape != actions.shape or target_actions.shape[-1] != ACTION_DIM:
        raise ModelContractError(
            f"target_actions must match actions shape [B, H, {ACTION_DIM}], "
            f"got {tuple(target_actions.shape)} vs {tuple(actions.shape)}"
        )

    future_loss = F.mse_loss(future_latents, target_future_latents)
    if action_scale is None:
        action_loss = F.mse_loss(actions, target_actions)
    else:
        action_scale = _require_tensor(action_scale, name="action_scale").to(
            device=actions.device,
            dtype=actions.dtype,
        )
        if action_scale.shape != (ACTION_DIM,):
            raise ModelContractError(
                f"action_scale must have shape ({ACTION_DIM},)"
            )
        if not bool(torch.isfinite(action_scale).all()) or bool(
            (action_scale <= 0).any()
        ):
            raise ModelContractError("action_scale must be finite and positive")
        action_loss = F.mse_loss(
            actions / action_scale,
            target_actions / action_scale,
        )
    total = weights.future_latent * future_loss + weights.action * action_loss
    losses = {
        "future_latent": future_loss,
        "action": action_loss,
    }

    ifp_latents = outputs.get("ifp_latents")
    if ifp_latents is not None and target_ifp_latents is not None:
        if target_ifp_latents.shape != ifp_latents.shape:
            raise ModelContractError(
                "target_ifp_latents must match ifp_latents shape, "
                f"got {tuple(target_ifp_latents.shape)} vs {tuple(ifp_latents.shape)}"
            )
        ifp_loss = F.mse_loss(ifp_latents, target_ifp_latents)
        total = total + weights.ifp * ifp_loss
        losses["ifp"] = ifp_loss
    else:
        losses["ifp"] = future_loss.new_zeros(())

    losses["total"] = total
    return losses
