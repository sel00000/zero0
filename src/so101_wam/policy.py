"""Inference-only CompactWAM policy wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from .constants import ACTION_DIM, PRIMARY_CAMERA_COUNT
from .context import ContextSnapshot
from .contracts import ActionChunk, ContractError, SensorimotorFrame
from .model import (
    ActionRangeConstraint,
    CompactWAM,
    ModelContractError,
    PromptEncoding,
)
from .task_specs import (
    DeterministicLanguageEncoder,
    LanguageEncoder,
    LanguageTokenizer,
    TaskSpec,
    Utf8Tokenizer,
)
from .tensorizer import CompactWAMBatch, tensorize_context, tensorize_task_spec


class PolicyError(ValueError):
    """Raised when policy inference cannot produce a valid action chunk."""


FUTURE_LATENT_TELEMETRY_CONTRACT = "compact_live_encoder_future_latent_v1"


class FutureLatentTelemetryMode(str, Enum):
    """Controls the optional future-latent evidence path."""

    DISABLED = "disabled"
    CAPTURE = "capture"


def _range_constraint(model: object) -> ActionRangeConstraint:
    if not hasattr(model, "action_range_constraint"):
        return ActionRangeConstraint.UNBOUNDED
    value = getattr(model, "action_range_constraint")
    if not isinstance(value, ActionRangeConstraint):
        raise PolicyError("action_range_constraint must be an ActionRangeConstraint")
    return value


def _range_vector(value: Sequence[float] | None, *, name: str) -> np.ndarray:
    if value is None:
        raise PolicyError("bounded model requires a joint range profile")
    array = np.array(tuple(value), dtype=np.float64)
    if array.shape != (ACTION_DIM,):
        raise PolicyError(f"{name} must contain {ACTION_DIM} finite values")
    if not np.isfinite(array).all():
        raise PolicyError(f"{name} must contain finite values")
    return array


def _model_range(model: object, *, name: str) -> np.ndarray:
    value = getattr(model, name, None)
    if not isinstance(value, Tensor):
        raise PolicyError("bounded model is missing native joint range buffers")
    if value.dtype != torch.float64:
        raise PolicyError("bounded model joint range buffers must be float64")
    if value.shape != (ACTION_DIM,):
        raise PolicyError(
            f"bounded model joint range buffers must contain {ACTION_DIM} values"
        )
    if not bool(torch.isfinite(value).all()):
        raise PolicyError("bounded model joint range buffers must be finite")
    return value.detach().cpu().numpy().astype(np.float64, copy=True)


def _validate_range_profile(
    model: object,
    *,
    joint_lower: Sequence[float] | None,
    joint_upper: Sequence[float] | None,
) -> None:
    if _range_constraint(model) is not ActionRangeConstraint.UNBOUNDED:
        lower = _range_vector(joint_lower, name="joint_lower")
        upper = _range_vector(joint_upper, name="joint_upper")
        if not np.all(lower < upper):
            raise PolicyError(
                "joint range profile lower bounds must be below upper bounds"
            )
        if not np.array_equal(lower, _model_range(model, name="action_lower")):
            raise PolicyError("joint range profile mismatch")
        if not np.array_equal(upper, _model_range(model, name="action_upper")):
            raise PolicyError("joint range profile mismatch")
        return

    if joint_lower is None and joint_upper is None:
        return
    lower = _range_vector(joint_lower, name="joint_lower")
    upper = _range_vector(joint_upper, name="joint_upper")
    if not np.all(lower < upper):
        raise PolicyError("joint range profile lower bounds must be below upper bounds")


def _readonly_latents(value: object, *, name: str) -> NDArray[np.float32]:
    array = np.array(value, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(array).all():
        raise PolicyError(f"{name} contains NaN or infinity")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class FutureLatentTelemetry:
    """One policy prediction and its current live-encoder observation."""

    observation_timestamp_s: float
    future_latents: NDArray[np.float32]
    observed_latent: NDArray[np.float32]
    contract: str = field(
        default=FUTURE_LATENT_TELEMETRY_CONTRACT,
        init=False,
    )

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.observation_timestamp_s)
            or self.observation_timestamp_s < 0.0
        ):
            raise PolicyError(
                "future latent observation timestamp must be finite and non-negative"
            )
        future = _readonly_latents(
            self.future_latents,
            name="future latent prediction",
        )
        observed = _readonly_latents(
            self.observed_latent,
            name="future latent observation",
        )
        if (
            future.ndim != 3
            or future.shape[0] < 1
            or future.shape[1] != PRIMARY_CAMERA_COUNT
            or future.shape[2] < 1
        ):
            raise PolicyError(
                "future latent prediction must have shape [future_T, V=2, latent_dim]"
            )
        if observed.shape != future.shape[1:]:
            raise PolicyError(
                "future latent observation must match prediction view and latent dimensions"
            )
        object.__setattr__(self, "future_latents", future)
        object.__setattr__(self, "observed_latent", observed)


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch(batch: CompactWAMBatch, device: torch.device) -> CompactWAMBatch:
    return CompactWAMBatch(
        prompt_images=batch.prompt_images.to(device=device),
        prompt_proprio=batch.prompt_proprio.to(device=device),
        prompt_actions=batch.prompt_actions.to(device=device),
        live_images=batch.live_images.to(device=device),
        live_proprio=batch.live_proprio.to(device=device),
        live_actions=batch.live_actions.to(device=device),
        prompt_mask=batch.prompt_mask.to(device=device),
    )


def _validate_actions(actions: Tensor, *, expected_horizon: int) -> np.ndarray:
    if not isinstance(actions, Tensor):
        raise PolicyError("CompactWAM.infer_action must return a torch.Tensor")
    if actions.dtype != torch.float32:
        raise PolicyError(f"model action output must be float32, got {actions.dtype}")
    if actions.shape != (1, expected_horizon, ACTION_DIM):
        raise PolicyError(
            f"model action output must have shape [1, {expected_horizon}, {ACTION_DIM}], got {tuple(actions.shape)}"
        )
    if not bool(torch.isfinite(actions).all()):
        raise PolicyError("model action output contains NaN or infinity")
    return actions.detach().cpu().numpy().astype(np.float32, copy=True).reshape(expected_horizon, ACTION_DIM)


@dataclass(slots=True)
class CompactWAMPolicy:
    """Adapts ``CompactWAM.infer_action`` to ``ContextSnapshot -> ActionChunk``."""

    model: CompactWAM
    servo_hz: float
    device: torch.device | str | None = None
    move_model: bool = False
    telemetry_mode: FutureLatentTelemetryMode = FutureLatentTelemetryMode.DISABLED
    joint_lower: Sequence[float] | None = None
    joint_upper: Sequence[float] | None = None
    language_tokenizer: LanguageTokenizer = field(default_factory=Utf8Tokenizer)
    language_encoder: LanguageEncoder = field(
        default_factory=DeterministicLanguageEncoder
    )
    _prompt_cache_key: tuple[str, str, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _prompt_encoding: PromptEncoding | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _future_latent_telemetry: FutureLatentTelemetry | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.servo_hz) or self.servo_hz <= 0:
            raise PolicyError("servo_hz must be finite and positive")
        if not isinstance(self.telemetry_mode, FutureLatentTelemetryMode):
            raise PolicyError("telemetry_mode must be a FutureLatentTelemetryMode")
        _validate_range_profile(
            self.model,
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
        )
        if self.device is not None:
            self.device = torch.device(self.device)
            if self.move_model:
                self.model.to(self.device)
            elif _module_device(self.model) != self.device:
                raise PolicyError(
                    "requested device differs from model device; move the model first or set move_model=True"
                )

    @property
    def dt_s(self) -> float:
        return 1.0 / float(self.servo_hz)

    @property
    def required_history_steps(self) -> int:
        return int(self.model.action_history_steps)

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return _module_device(self.model)

    def _chunk(self, actions: Tensor, *, now_s: float) -> ActionChunk:
        targets = _validate_actions(
            actions,
            expected_horizon=self.model.action_horizon,
        )
        try:
            return ActionChunk(
                target_joint_position=targets,
                dt_s=self.dt_s,
                created_at_s=now_s,
            )
        except ContractError as error:
            raise PolicyError(str(error)) from error

    def clear_prompt_cache(self) -> None:
        """Discard prompt tokens after model weights or task prompts change."""

        self._prompt_cache_key = None
        self._prompt_encoding = None
        self._future_latent_telemetry = None

    def take_future_latent_telemetry(self) -> FutureLatentTelemetry | None:
        """Consume the latest opt-in telemetry record exactly once."""

        telemetry = self._future_latent_telemetry
        self._future_latent_telemetry = None
        return telemetry

    def _infer_cached(
        self,
        batch: CompactWAMBatch,
        *,
        prompt_key: str,
        task_conditioning: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if not isinstance(self.model, CompactWAM):
            with torch.inference_mode():
                if task_conditioning is None:
                    actions = self.model.infer_action(**batch.as_kwargs())
                else:
                    actions = self.model.infer_action(
                        **batch.as_kwargs(),
                        task_conditioning=task_conditioning,
                    )
            return actions, None

        device = self._device()
        cache_key = (prompt_key, str(device), id(self.model))
        if self._prompt_cache_key != cache_key or self._prompt_encoding is None:
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.inference_mode():
                    self._prompt_encoding = self.model.encode_prompt(
                        batch.prompt_images,
                        batch.prompt_proprio,
                        batch.prompt_actions,
                        prompt_mask=batch.prompt_mask,
                    )
            finally:
                self.model.train(was_training)
            self._prompt_cache_key = cache_key

        with torch.inference_mode():
            outputs = self.model.infer_outputs_cached(
                self._prompt_encoding,
                batch.live_images,
                batch.live_proprio,
                batch.live_actions,
                task_conditioning=task_conditioning,
            )
        return outputs["actions"], outputs["future_latents"]  # type: ignore[return-value]

    def _capture_future_latents(
        self,
        batch: CompactWAMBatch,
        future_latents: Tensor | None,
        *,
        observation_timestamp_s: float,
    ) -> None:
        if self.telemetry_mode is FutureLatentTelemetryMode.DISABLED:
            return
        if future_latents is None:
            raise PolicyError("telemetry capture requires future latent model output")
        if not isinstance(future_latents, Tensor):
            raise PolicyError("future latent model output must be a torch.Tensor")
        expected = (
            1,
            self.model.future_steps,
            PRIMARY_CAMERA_COUNT,
            self.model.latent_dim,
        )
        if future_latents.shape != expected:
            raise PolicyError(
                f"future latent model output must have shape {expected}, "
                f"got {tuple(future_latents.shape)}"
            )
        if future_latents.dtype != torch.float32:
            raise PolicyError(
                "future latent model output must be float32, "
                f"got {future_latents.dtype}"
            )
        if not bool(torch.isfinite(future_latents).all()):
            raise PolicyError("future latent model output contains NaN or infinity")

        observed = self.model.encode_future_images(batch.live_images[:, -1:, ...])
        expected_observed = (
            1,
            1,
            PRIMARY_CAMERA_COUNT,
            self.model.latent_dim,
        )
        if observed.shape != expected_observed:
            raise PolicyError(
                f"future latent observation must have shape {expected_observed}, "
                f"got {tuple(observed.shape)}"
            )
        self._future_latent_telemetry = FutureLatentTelemetry(
            observation_timestamp_s=observation_timestamp_s,
            future_latents=future_latents[0].detach().cpu().numpy(),
            observed_latent=observed[0, 0].detach().cpu().numpy(),
        )

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        self._future_latent_telemetry = None
        if not np.isfinite(now_s) or now_s < 0:
            raise PolicyError("now_s must be finite and non-negative")
        if len(snapshot.live_frames) < self.model.action_history_steps:
            raise ModelContractError(
                f"snapshot live history must contain at least {self.model.action_history_steps} frames, "
                f"got {len(snapshot.live_frames)}"
            )

        batch = _move_batch(tensorize_context(snapshot), self._device())
        actions, future_latents = self._infer_cached(
            batch,
            prompt_key=f"robot:{snapshot.prompt_fingerprint}",
        )
        chunk = self._chunk(actions, now_s=now_s)
        self._capture_future_latents(
            batch,
            future_latents,
            observation_timestamp_s=snapshot.live_frames[-1].timestamp_s,
        )
        return chunk

    def predict_task(
        self,
        task_spec: TaskSpec,
        live_frames: tuple[SensorimotorFrame, ...],
        *,
        now_s: float,
    ) -> ActionChunk:
        """Infer from a task spec without changing the real deployment runtime."""

        self._future_latent_telemetry = None
        if not np.isfinite(now_s) or now_s < 0:
            raise PolicyError("now_s must be finite and non-negative")
        if len(live_frames) < self.model.action_history_steps:
            raise ModelContractError(
                "task-spec live history must contain at least "
                f"{self.model.action_history_steps} frames, got {len(live_frames)}"
            )

        device = self._device()
        neutral = self.model.denormalize_axes(
            torch.zeros(ACTION_DIM, dtype=torch.float32, device=device)
        )
        task_batch = tensorize_task_spec(
            task_spec,
            tuple(live_frames),
            neutral_axes=neutral.detach().cpu().numpy(),
        )
        batch = _move_batch(task_batch.batch, device)
        task_conditioning: Tensor | None = None
        if task_batch.language_text is not None:
            token_ids = self.language_tokenizer.tokenize(
                task_batch.language_text
            )
            encoded = self.language_encoder.encode(
                token_ids,
                width=self.model.latent_dim,
            )
            task_conditioning = torch.from_numpy(
                np.asarray(encoded, dtype=np.float32).copy()
            ).view(1, -1)
            task_conditioning = task_conditioning.to(device=device)

        actions, future_latents = self._infer_cached(
            batch,
            prompt_key=f"{task_batch.kind.value}:{task_batch.fingerprint}",
            task_conditioning=task_conditioning,
        )
        chunk = self._chunk(actions, now_s=now_s)
        self._capture_future_latents(
            batch,
            future_latents,
            observation_timestamp_s=live_frames[-1].timestamp_s,
        )
        return chunk

    def __call__(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        return self.predict(snapshot, now_s=now_s)


__all__ = [
    "CompactWAMPolicy",
    "FUTURE_LATENT_TELEMETRY_CONTRACT",
    "FutureLatentTelemetry",
    "FutureLatentTelemetryMode",
    "PolicyError",
]
