"""Two-stage offline training for a non-deployable CompactWAM candidate."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from itertools import chain
import json
from math import isfinite
import os
from pathlib import Path
from re import fullmatch
from tempfile import NamedTemporaryFile
from time import perf_counter
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .checkpoint import (
    compact_wam_architecture,
    load_compact_wam_bundle,
    save_compact_wam_checkpoint,
)
from .constants import ACTION_DIM
from .decoder_evaluation import (
    DECODER_AUDIT_SCHEMA,
    FutureSource,
    evaluate_decoder,
    named_tensor_hashes,
)
from .model import (
    ActionDecoder,
    ActionRangeConstraint,
    CompactWAM,
    FusedIFP,
    InverseDynamicsActionHead,
    WAMLossWeights,
)
from .training_data import (
    EpisodeRecord,
    TrainingWindowSpec,
    action_source_summary,
    build_training_windows,
    draw_task_balanced_windows,
    materialize_training_window,
    task_draw_counts,
    training_axis_statistics,
    validate_task_disjoint_split,
)
from .vision import COMPACT_IMAGE_MAX_SIDE


TRAINING_REPORT_SCHEMA = "so101_wam.offline_training.v1"
PAPER_IFP_STEPS = 4
PAPER_IFP_STRIDE = 2
PAPER_IFP_LOSS_WEIGHTS = (0.5, 0.25, 0.15, 0.15)


class TrainingError(RuntimeError):
    """Raised when candidate training or evidence publication fails."""


class TrainingArtifactExistsError(TrainingError):
    """Raised when immutable candidate artifacts already exist."""


class IFPArchitecture(StrEnum):
    """Select the legacy checkpoint head or removable fused modules."""

    COMPACT_LINEAR = "compact_linear"
    FUSED_MODULES = "fused_modules"


class SamplingStrategy(StrEnum):
    """Select the training-window sampling policy."""

    WINDOW_SHUFFLE = "window_shuffle"
    TASK_BALANCED = "task_balanced"


@dataclass(frozen=True, slots=True)
class CompactWAMTrainingConfig:
    """Small paper-inspired training protocol for the local compact model."""

    policy_hz: float = 10.0
    servo_hz: float = 50.0
    latent_dim: int = 32
    transformer_layers: int = 1
    transformer_heads: int = 4
    future_steps: int = 3
    action_horizon: int = 10
    action_history_steps: int = 4
    ifp_steps: int = 2
    ifp_stride: int = 2
    ifp_architecture: IFPArchitecture = IFPArchitecture.COMPACT_LINEAR
    ifp_window_steps: int | None = None
    max_context_steps: int = 300
    stage1_steps: int = 100
    stage2_steps: int = 100
    sampling_strategy: SamplingStrategy = SamplingStrategy.TASK_BALANCED
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    future_latent_weight: float = 1.0
    action_weight: float = 1.0
    ifp_weight: float = 0.25
    seed: int = 7
    action_decoder: ActionDecoder = ActionDecoder.LEGACY_MEAN

    def __post_init__(self) -> None:
        for value, name in (
            (self.policy_hz, "policy_hz"),
            (self.servo_hz, "servo_hz"),
            (self.learning_rate, "learning_rate"),
            (self.max_grad_norm, "max_grad_norm"),
        ):
            if not isfinite(value) or value <= 0:
                raise TrainingError(f"{name} must be finite and positive")
        if not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise TrainingError("weight_decay must be finite and non-negative")
        for value, name in (
            (self.future_latent_weight, "future_latent_weight"),
            (self.action_weight, "action_weight"),
            (self.ifp_weight, "ifp_weight"),
        ):
            if not isfinite(value) or value < 0:
                raise TrainingError(f"{name} must be finite and non-negative")
        if self.future_latent_weight <= 0 or self.action_weight <= 0:
            raise TrainingError(
                "future_latent_weight and action_weight must both be positive"
            )
        for value, name in (
            (self.latent_dim, "latent_dim"),
            (self.transformer_layers, "transformer_layers"),
            (self.transformer_heads, "transformer_heads"),
            (self.future_steps, "future_steps"),
            (self.action_horizon, "action_horizon"),
            (self.action_history_steps, "action_history_steps"),
            (self.ifp_stride, "ifp_stride"),
            (self.max_context_steps, "max_context_steps"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise TrainingError(f"{name} must be a positive integer")
        if self.max_context_steps < 300:
            raise TrainingError("max_context_steps must be at least 300")
        if self.latent_dim % self.transformer_heads:
            raise TrainingError("latent_dim must be divisible by transformer_heads")
        if not isinstance(self.ifp_steps, int) or isinstance(self.ifp_steps, bool) or self.ifp_steps < 0:
            raise TrainingError("ifp_steps must be a non-negative integer")
        if not isinstance(self.ifp_architecture, IFPArchitecture):
            raise TrainingError("ifp_architecture must be an IFPArchitecture")
        if self.ifp_window_steps is not None and (
            not isinstance(self.ifp_window_steps, int)
            or isinstance(self.ifp_window_steps, bool)
            or self.ifp_window_steps < self.ifp_steps
        ):
            raise TrainingError(
                "ifp_window_steps must be an integer greater than or equal to ifp_steps"
            )
        if self.ifp_steps > 0 and self.ifp_weight <= 0:
            raise TrainingError("ifp_weight must be positive when ifp_steps > 0")
        if not isinstance(self.stage1_steps, int) or isinstance(self.stage1_steps, bool) or self.stage1_steps < 0:
            raise TrainingError("stage1_steps must be a non-negative integer")
        if not isinstance(self.stage2_steps, int) or isinstance(self.stage2_steps, bool) or self.stage2_steps < 1:
            raise TrainingError("stage2_steps must be a positive integer")
        if not isinstance(self.sampling_strategy, SamplingStrategy):
            raise TrainingError("sampling_strategy must be a SamplingStrategy")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise TrainingError("seed must be a non-negative integer")
        if not isinstance(self.action_decoder, ActionDecoder):
            raise TrainingError("action_decoder must be an ActionDecoder")

    @property
    def effective_ifp_window_steps(self) -> int:
        if self.ifp_window_steps is None:
            return self.ifp_steps
        return self.ifp_window_steps


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Published offline candidate paths and scalar validation evidence."""

    checkpoint_path: str
    report_path: str
    checkpoint_id: str
    training_evidence_sha256: str
    train_windows: int
    validation_windows: int
    validation_action_mse_normalized: float
    validation_action_mae_native: float
    validation_future_latent_mse: float
    trained: bool = False
    deployment_ready: bool = False


def _task_inventory(
    records: Sequence[EpisodeRecord],
) -> list[dict[str, str | int]]:
    identities = sorted(
        {(record.data.task_index, record.data.task) for record in records}
    )
    return [{"task_index": task_index, "task": task} for task_index, task in identities]


def train_offline_candidate(
    train_records: Sequence[EpisodeRecord],
    validation_records: Sequence[EpisodeRecord],
    *,
    checkpoint_path: str | Path,
    report_path: str | Path,
    checkpoint_id: str,
    config: CompactWAMTrainingConfig = CompactWAMTrainingConfig(),
    device: str | torch.device = "cpu",
    study_sha256: str | None = None,
) -> TrainingArtifacts:
    """Train, validate, and immutably publish a candidate-only checkpoint.

    This function intentionally stores ``trained=false``. Offline MSE on
    measured-position proxy labels is not evidence for real motor output.
    """

    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise TrainingError("checkpoint_id must be a non-empty string")
    checkpoint_id = checkpoint_id.strip()
    checkpoint_target = Path(checkpoint_path).resolve()
    report_target = Path(report_path).resolve()
    if checkpoint_target == report_target:
        raise TrainingError("checkpoint and report paths must be different")
    if checkpoint_target.exists() or report_target.exists():
        raise TrainingArtifactExistsError(
            "training artifacts are immutable; choose new checkpoint/report paths"
        )

    try:
        target_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingError(f"invalid training device: {device}") from error
    audit_requested = study_sha256 is not None
    if audit_requested:
        _require_decoder_audit(config, target_device, study_sha256)
    if target_device.type not in {"cpu", "cuda"}:
        raise TrainingError("training device must be cpu or cuda")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError(f"requested CUDA device is unavailable: {target_device}")
    if (
        target_device.type == "cuda"
        and target_device.index is not None
        and target_device.index >= torch.cuda.device_count()
    ):
        raise TrainingError(f"requested CUDA device is unavailable: {target_device}")

    split = validate_task_disjoint_split(train_records, validation_records)
    train_windows = build_training_windows(
        split.train,
        policy_hz=config.policy_hz,
        servo_hz=config.servo_hz,
        action_history_steps=config.action_history_steps,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        ifp_steps=config.effective_ifp_window_steps,
        ifp_stride=config.ifp_stride,
        max_context_steps=config.max_context_steps,
    )
    validation_windows = build_training_windows(
        split.validation,
        policy_hz=config.policy_hz,
        servo_hz=config.servo_hz,
        action_history_steps=config.action_history_steps,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        ifp_steps=config.effective_ifp_window_steps,
        ifp_stride=config.ifp_stride,
        max_context_steps=config.max_context_steps,
    )

    _seed_everything(config.seed)
    checkpoint_ifp_steps = (
        config.ifp_steps
        if config.ifp_architecture is IFPArchitecture.COMPACT_LINEAR
        else 0
    )
    model = CompactWAM(
        latent_dim=config.latent_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        action_history_steps=config.action_history_steps,
        ifp_steps=checkpoint_ifp_steps,
        max_context_steps=config.max_context_steps,
    ).to(target_device)
    # Build legacy first so shared modules and following IFP init keep the
    # historical RNG path; fork the replacement so wide heads compare fairly.
    if config.action_decoder is not ActionDecoder.LEGACY_MEAN:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model.action_head = InverseDynamicsActionHead(
                config.latent_dim,
                config.action_horizon,
                config.action_history_steps,
                future_steps=config.future_steps,
                action_decoder=config.action_decoder,
            ).to(target_device)
    axis_mean, axis_scale = training_axis_statistics(split.train)
    model.set_axis_normalization(
        torch.from_numpy(axis_mean),
        torch.from_numpy(axis_scale),
    )
    initial_model_state_sha256 = _state_dict_sha256(model)
    initial_tensors = named_tensor_hashes(model) if audit_requested else None
    fused_ifp = (
        FusedIFP(model, ifp_steps=config.ifp_steps).to(target_device)
        if config.ifp_architecture is IFPArchitecture.FUSED_MODULES
        and config.ifp_steps > 0
        else None
    )
    # Keep subsequent stochastic training identical across K variants.
    _seed_everything(config.seed)

    rng = np.random.default_rng(config.seed)
    total_steps = config.stage1_steps + config.stage2_steps
    stage1_schedule, stage2_schedule = _draw_stage_schedules(
        train_windows,
        stage1_steps=config.stage1_steps,
        stage2_steps=config.stage2_steps,
        strategy=config.sampling_strategy,
        rng=rng,
    )
    full_schedule = stage1_schedule + stage2_schedule
    training_schedule_sha256 = _schedule_sha256(
        stage1_schedule,
        stage2_schedule,
    )
    stage1_start = perf_counter()
    stage1_loss = _train_inverse_dynamics_stage(
        model,
        stage1_schedule,
        config=config,
        device=target_device,
    )
    stage1_train_seconds = perf_counter() - stage1_start
    decoder_audit: dict[str, Any] | None = None
    if audit_requested:
        assert initial_tensors is not None
        stage1_tensors = named_tensor_hashes(model)
        _require_head_only_stage(initial_tensors, stage1_tensors)
        stage1_eval_start = perf_counter()
        stage1_audit = evaluate_decoder(
            model,
            validation_windows,
            source=FutureSource.ORACLE,
        )
        stage1_eval_seconds = perf_counter() - stage1_eval_start
        assert study_sha256 is not None
        decoder_audit = {
            "schema_version": DECODER_AUDIT_SCHEMA,
            "study_sha256": study_sha256,
            "initial_tensors": initial_tensors,
            "stage1_tensors": stage1_tensors,
            "stage_schedule_sha256": {
                "stage1": _schedule_sha256(stage1_schedule, ()),
                "stage2": _schedule_sha256((), stage2_schedule),
            },
            "head_parameters": _parameter_count(model.action_head.parameters()),
            "model_parameters": _parameter_count(model.parameters()),
            "stage1": stage1_audit,
            "seconds": {
                "stage1_train": stage1_train_seconds,
                "stage1_evaluation": stage1_eval_seconds,
            },
        }
    stage2_start = perf_counter()
    stage2_metrics = _train_end_to_end_stage(
        model,
        stage2_schedule,
        config=config,
        device=target_device,
        fused_ifp=fused_ifp,
    )
    stage2_train_seconds = perf_counter() - stage2_start
    if decoder_audit is not None:
        stage2_eval_start = perf_counter()
        decoder_audit["stage2"] = evaluate_decoder(
            model,
            validation_windows,
            source=FutureSource.PREDICTED,
        )
        decoder_audit["final_tensors"] = named_tensor_hashes(model)
        seconds = decoder_audit["seconds"]
        assert isinstance(seconds, dict)
        seconds["stage2_train"] = stage2_train_seconds
        seconds["stage2_evaluation"] = perf_counter() - stage2_eval_start
    validation_metrics = _evaluate(
        model,
        validation_windows,
        config=config,
        device=target_device,
    )

    train_action_source = action_source_summary(split.train)
    validation_action_source = action_source_summary(split.validation)
    train_tasks = _task_inventory(split.train)
    validation_tasks = _task_inventory(split.validation)
    action_target_source = (
        "servo_rate_linear_interpolation_of_episode_action;"
        f"train={train_action_source};validation={validation_action_source}"
    )
    report_core: dict[str, Any] = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "evidence_level": "offline",
        "artifact_kind": "compact_wam_candidate",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": checkpoint_id,
        "protocol": {
            "prompt_pairing": "same_task_different_episode",
            "split": "task_disjoint",
            "sampling_strategy": config.sampling_strategy.value,
            "stage1": "ground_truth_future_latent_inverse_dynamics",
            "stage2": "predicted_future_latent_end_to_end_with_ifp",
            "ifp_architecture": config.ifp_architecture.value,
            "ifp_stride": config.ifp_stride,
            "ifp_loss_weights": list(_ifp_loss_weights(config.ifp_steps)),
            "ifp_module_removed_from_inference": model.ifp_head is None,
            "wrist_rgb_preprocess": (
                f"aspect_preserving_nearest_max_side_{COMPACT_IMAGE_MAX_SIDE}"
            ),
            "action_target_source": action_target_source,
            "action_row_zero_timing": "policy_anchor_immediate",
            "real_output_authorized": False,
        },
        "data": {
            "train_episode_count": len(split.train),
            "validation_episode_count": len(split.validation),
            "train_task_count": len(train_tasks),
            "validation_task_count": len(validation_tasks),
            "train_tasks": train_tasks,
            "validation_tasks": validation_tasks,
            "train_window_count": len(train_windows),
            "validation_window_count": len(validation_windows),
            "train_task_window_counts": task_draw_counts(train_windows),
            "sampling_audit": {
                "schema_version": "so101_wam.task_sampling.v1",
                "strategy": config.sampling_strategy.value,
                "seed": config.seed,
                "start_draw": 0,
                "next_draw": total_steps,
                "resume_semantics": (
                    "sampler_cursor_only"
                    if config.sampling_strategy is SamplingStrategy.TASK_BALANCED
                    else "legacy_stage_boundary_replay"
                ),
                "task_draw_counts": task_draw_counts(full_schedule),
                "schedule_sha256": training_schedule_sha256,
            },
            "train_split_sha256": split.train_digest,
            "validation_split_sha256": split.validation_digest,
            "train_action_source": train_action_source,
            "validation_action_source": validation_action_source,
        },
        "model": compact_wam_architecture(model),
        "initial_model_state_sha256": initial_model_state_sha256,
        "training_schedule_sha256": training_schedule_sha256,
        "normalization": {
            "axis_mean": [float(value) for value in axis_mean],
            "axis_scale": [float(value) for value in axis_scale],
        },
        "optimization": {
            **asdict(config),
            "device": str(target_device),
            "optimizer": "AdamW",
            "stage1_final_action_mse_normalized": stage1_loss,
            **stage2_metrics,
        },
        "validation": validation_metrics,
        "limitations": [
            "offline metrics do not certify real SO-101 actuation",
            "G9 measured-position proxy labels are not sent Goal_Position commands",
            "task rollout success and real collision margin remain unmeasured",
        ],
    }
    if decoder_audit is not None:
        report_core["decoder_audit"] = decoder_audit
    evidence_sha256 = _json_sha256(report_core)
    if config.ifp_steps == 0:
        training_objective = "compact_future_latent_action_mse"
    elif config.ifp_architecture is IFPArchitecture.FUSED_MODULES:
        training_objective = "compact_future_latent_action_fused_ifp_mse"
    else:
        training_objective = "compact_future_latent_action_ifp_mse"
    metadata: dict[str, str | int | float | bool | None] = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "checkpoint_id": checkpoint_id,
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "training_evidence_sha256": evidence_sha256,
        "training_objective": training_objective,
        "training_ifp_steps": config.ifp_steps,
        "ifp_architecture": config.ifp_architecture.value,
        "inference_ifp_module_present": model.ifp_head is not None,
        "action_target_source": action_target_source,
        "train_task_count": report_core["data"]["train_task_count"],
        "validation_task_count": report_core["data"]["validation_task_count"],
        "optimizer_steps": config.stage1_steps + config.stage2_steps,
        "seed": config.seed,
        "sampling_strategy": config.sampling_strategy.value,
        "sampling_schedule_sha256": training_schedule_sha256,
        "validation_action_mse_normalized": validation_metrics[
            "action_mse_normalized"
        ],
        "action_decoder": model.action_decoder.value,
    }
    if study_sha256 is not None:
        metadata["study_sha256"] = study_sha256
    report = {
        **report_core,
        "training_evidence_sha256": evidence_sha256,
    }
    _publish_candidate_pair(
        model=model,
        metadata=metadata,
        report=report,
        checkpoint_path=checkpoint_target,
        report_path=report_target,
    )

    return TrainingArtifacts(
        checkpoint_path=str(checkpoint_target),
        report_path=str(report_target),
        checkpoint_id=checkpoint_id,
        training_evidence_sha256=evidence_sha256,
        train_windows=len(train_windows),
        validation_windows=len(validation_windows),
        validation_action_mse_normalized=float(
            validation_metrics["action_mse_normalized"]
        ),
        validation_action_mae_native=float(validation_metrics["action_mae_native"]),
        validation_future_latent_mse=float(
            validation_metrics["future_latent_mse"]
        ),
    )


def _train_inverse_dynamics_stage(
    model: CompactWAM,
    schedule: Sequence[TrainingWindowSpec],
    *,
    config: CompactWAMTrainingConfig,
    device: torch.device,
) -> float | None:
    if not schedule:
        return None
    optimizer = torch.optim.AdamW(
        model.action_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    final_loss = 0.0
    for spec in schedule:
        batch = materialize_training_window(
            spec, image_max_side=COMPACT_IMAGE_MAX_SIDE
        ).to(device)
        with torch.no_grad():
            target_future_latents = model.encoder(
                batch.target_future_images,
                segment_id=1,
            )
        current = model.normalize_axes(batch.live_proprio[:, -1, :])
        history = model.normalize_axes(
            batch.live_actions[:, -model.action_history_steps :, :]
        )
        target = model.normalize_axes(batch.target_actions)
        raw_predicted = model.action_head(target_future_latents, current, history)
        predicted = raw_predicted
        if model.action_range_constraint is not ActionRangeConstraint.UNBOUNDED:
            predicted = model.normalize_axes(model.decode_actions(raw_predicted))
        loss = F.mse_loss(predicted, target)
        _optimizer_step(
            loss,
            optimizer=optimizer,
            parameters=model.action_head.parameters(),
            max_grad_norm=config.max_grad_norm,
        )
        final_loss = float(loss.detach().cpu())
    return final_loss


def _require_decoder_audit(
    config: CompactWAMTrainingConfig,
    device: torch.device,
    study_sha256: str | None,
) -> None:
    if not isinstance(study_sha256, str) or fullmatch(r"[0-9a-f]{64}", study_sha256) is None:
        raise TrainingError("study_sha256 must be 64 lowercase hex characters")
    if device.type != "cpu":
        raise TrainingError("decoder audit requires cpu training")
    if config.future_steps < 2:
        raise TrainingError("decoder audit requires future_steps >= 2")
    if config.stage1_steps < 1:
        raise TrainingError("decoder audit requires stage1_steps >= 1")


def _require_head_only_stage(
    initial: Mapping[str, str],
    stage1: Mapping[str, str],
) -> None:
    for name, digest in initial.items():
        if name.startswith("action_head."):
            continue
        if stage1.get(name) != digest:
            raise TrainingError(f"stage1 modified non-action-head tensor: {name}")


def _parameter_count(parameters: Iterator[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _train_end_to_end_stage(
    model: CompactWAM,
    schedule: Sequence[TrainingWindowSpec],
    *,
    config: CompactWAMTrainingConfig,
    device: torch.device,
    fused_ifp: FusedIFP | None,
) -> dict[str, float]:
    parameters = tuple(
        chain(
            model.parameters(),
            () if fused_ifp is None else fused_ifp.parameters(),
        )
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weights = WAMLossWeights(
        future_latent=config.future_latent_weight,
        action=config.action_weight,
        ifp=config.ifp_weight,
    )
    model.train()
    if fused_ifp is not None:
        fused_ifp.train()
    final: dict[str, float] = {}
    for spec in schedule:
        batch = materialize_training_window(
            spec, image_max_side=COMPACT_IMAGE_MAX_SIDE
        ).to(device)
        if fused_ifp is None:
            outputs = model(**batch.model_kwargs(), compute_ifp=True)
            layer_features = None
        else:
            outputs, layer_features = model.forward_with_context_features(
                **batch.model_kwargs()
            )
        with torch.no_grad():
            target_future_latents = model.encoder(
                batch.target_future_images,
                segment_id=1,
            )
            target_ifp_latents = (
                None
                if config.ifp_steps == 0 or batch.target_ifp_images is None
                else model.encoder(
                    batch.target_ifp_images[:, : config.ifp_steps],
                    segment_id=1,
                )
            )
        losses = model.loss(
            outputs,
            target_future_latents=target_future_latents,
            target_actions=batch.target_actions,
            target_ifp_latents=(None if fused_ifp is not None else target_ifp_latents),
            weights=weights,
            action_scale=cast(Tensor, model.axis_scale),
        )
        if fused_ifp is not None:
            if layer_features is None or target_ifp_latents is None:
                raise TrainingError("fused IFP training requires future targets")
            predicted_ifp = fused_ifp(layer_features)
            ifp_loss = _weighted_ifp_loss(
                predicted_ifp,
                target_ifp_latents,
                weights=_ifp_loss_weights(config.ifp_steps),
            )
            losses["ifp"] = ifp_loss
            losses["total"] = losses["total"] + config.ifp_weight * ifp_loss
        _optimizer_step(
            losses["total"],
            optimizer=optimizer,
            parameters=parameters,
            max_grad_norm=config.max_grad_norm,
        )
        final = {
            f"stage2_final_{name}_loss": float(value.detach().cpu())
            for name, value in losses.items()
        }
    return final


def _ifp_loss_weights(ifp_steps: int) -> tuple[float, ...]:
    if ifp_steps == 0:
        return ()
    if ifp_steps == PAPER_IFP_STEPS:
        return PAPER_IFP_LOSS_WEIGHTS
    weight = 1.0 / ifp_steps
    return (weight,) * ifp_steps


def _weighted_ifp_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    weights: Sequence[float],
) -> Tensor:
    if predicted.shape != target.shape:
        raise TrainingError(
            "fused IFP target shape mismatch: "
            f"{tuple(predicted.shape)} != {tuple(target.shape)}"
        )
    if predicted.ndim != 4 or predicted.shape[1] != len(weights):
        raise TrainingError("fused IFP weights must match the prediction steps")
    per_step = torch.square(predicted - target).mean(dim=(0, 2, 3))
    weight_tensor = predicted.new_tensor(tuple(weights))
    return torch.sum(per_step * weight_tensor)


@torch.no_grad()
def _evaluate(
    model: CompactWAM,
    windows: Sequence[TrainingWindowSpec],
    *,
    config: CompactWAMTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    action_squared_sum = 0.0
    action_absolute_sum = 0.0
    action_count = 0
    future_squared_sum = 0.0
    future_count = 0
    null_action_delta_sum = 0.0
    null_future_delta_sum = 0.0
    temporal_action_delta_sum = 0.0
    temporal_future_delta_sum = 0.0
    axis_absolute_sum = torch.zeros(ACTION_DIM, dtype=torch.float64)
    action_out_of_bounds_count = 0
    action_endpoint_target_count = 0
    action_endpoint_absolute_sum = 0.0
    for spec in windows:
        batch = materialize_training_window(
            spec, image_max_side=COMPACT_IMAGE_MAX_SIDE
        ).to(device)
        model_kwargs = dict(batch.model_kwargs())
        outputs = model(**model_kwargs, compute_ifp=False)
        target_future = model.encoder(batch.target_future_images, segment_id=1)
        predicted_future = outputs["future_latents"]
        predicted_actions = outputs["actions"]
        assert predicted_future is not None and predicted_actions is not None

        neutral = cast(Tensor, model.axis_mean).view(1, 1, -1)
        null_kwargs = {
            **model_kwargs,
            "prompt_images": torch.zeros_like(batch.prompt_images),
            "prompt_proprio": neutral.expand_as(batch.prompt_proprio),
            "prompt_actions": neutral.expand_as(batch.prompt_actions),
        }
        temporal_kwargs = {
            **model_kwargs,
            "prompt_images": torch.flip(batch.prompt_images, dims=(1,)),
            "prompt_proprio": torch.flip(batch.prompt_proprio, dims=(1,)),
            "prompt_actions": torch.flip(batch.prompt_actions, dims=(1,)),
        }
        null_outputs = model(**null_kwargs, compute_ifp=False)
        temporal_outputs = model(**temporal_kwargs, compute_ifp=False)
        null_future = null_outputs["future_latents"]
        null_actions = null_outputs["actions"]
        temporal_future = temporal_outputs["future_latents"]
        temporal_actions = temporal_outputs["actions"]
        assert null_future is not None and null_actions is not None
        assert temporal_future is not None and temporal_actions is not None

        normalized_error = (predicted_actions - batch.target_actions) / cast(
            Tensor, model.axis_scale
        )
        action_squared_sum += float(torch.square(normalized_error).sum().cpu())
        action_absolute_sum += float(
            torch.abs(predicted_actions - batch.target_actions).sum().cpu()
        )
        axis_absolute_sum += (
            torch.abs(predicted_actions - batch.target_actions)
            .sum(dim=(0, 1))
            .double()
            .cpu()
        )
        if model.action_range_constraint is not ActionRangeConstraint.UNBOUNDED:
            lower = cast(Tensor, model.action_lower).to(device=device, dtype=torch.float64)
            upper = cast(Tensor, model.action_upper).to(device=device, dtype=torch.float64)
            predicted64 = predicted_actions.to(dtype=torch.float64)
            target64 = batch.target_actions.to(dtype=torch.float64)
            out_of_bounds = (predicted64 < lower) | (predicted64 > upper)
            endpoint_targets = (target64 == lower) | (target64 == upper)
            action_out_of_bounds_count += int(out_of_bounds.sum().cpu())
            action_endpoint_target_count += int(endpoint_targets.sum().cpu())
            action_endpoint_absolute_sum += float(
                torch.abs(predicted64 - target64)[endpoint_targets].sum().cpu()
            )
        action_count += int(predicted_actions.numel())
        future_squared_sum += float(
            torch.square(predicted_future - target_future).sum().cpu()
        )
        future_count += int(predicted_future.numel())
        null_action_delta_sum += float(
            torch.abs(predicted_actions - null_actions).sum().cpu()
        )
        null_future_delta_sum += float(
            torch.abs(predicted_future - null_future).sum().cpu()
        )
        temporal_action_delta_sum += float(
            torch.abs(predicted_actions - temporal_actions).sum().cpu()
        )
        temporal_future_delta_sum += float(
            torch.abs(predicted_future - temporal_future).sum().cpu()
        )

    metrics = {
        "action_mse_normalized": action_squared_sum / action_count,
        "action_mae_native": action_absolute_sum / action_count,
        "future_latent_mse": future_squared_sum / future_count,
        "null_prompt_action_mean_abs_delta": (null_action_delta_sum / action_count),
        "null_prompt_future_mean_abs_delta": (null_future_delta_sum / future_count),
        "temporal_prompt_action_mean_abs_delta": (
            temporal_action_delta_sum / action_count
        ),
        "temporal_prompt_future_mean_abs_delta": (
            temporal_future_delta_sum / future_count
        ),
    }
    if model.action_range_constraint is not ActionRangeConstraint.UNBOUNDED:
        per_axis_count = action_count // ACTION_DIM
        metrics.update(
            {
                f"action_mae_native_axis_{axis}": float(
                    axis_absolute_sum[axis] / per_axis_count
                )
                for axis in range(ACTION_DIM)
            }
        )
        metrics["action_out_of_bounds_count"] = float(action_out_of_bounds_count)
        metrics["action_endpoint_target_count"] = float(action_endpoint_target_count)
        metrics["action_endpoint_mae_native"] = (
            action_endpoint_absolute_sum / action_endpoint_target_count
            if action_endpoint_target_count > 0
            else 0.0
        )
    if not all(isfinite(value) for value in metrics.values()):
        raise TrainingError(f"validation produced non-finite metrics: {metrics}")
    return metrics


def _optimizer_step(
    loss: Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    parameters: Iterator[torch.nn.Parameter] | Sequence[torch.nn.Parameter],
    max_grad_norm: float,
) -> None:
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise TrainingError("training loss must be one finite scalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(tuple(parameters), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise TrainingError("training gradients are non-finite")
    optimizer.step()


def _shuffled_window_stream(
    windows: Sequence[TrainingWindowSpec],
    *,
    rng: np.random.Generator,
) -> Iterator[TrainingWindowSpec]:
    if not windows:
        raise TrainingError("training windows must be non-empty")
    while True:
        for index in rng.permutation(len(windows)):
            yield windows[int(index)]


def _draw_window_schedule(
    windows: Sequence[TrainingWindowSpec],
    *,
    steps: int,
    rng: np.random.Generator,
) -> tuple[TrainingWindowSpec, ...]:
    if steps == 0:
        return ()
    stream = _shuffled_window_stream(windows, rng=rng)
    return tuple(next(stream) for _ in range(steps))


def _draw_stage_schedules(
    windows: Sequence[TrainingWindowSpec],
    *,
    stage1_steps: int,
    stage2_steps: int,
    strategy: SamplingStrategy,
    rng: np.random.Generator,
) -> tuple[tuple[TrainingWindowSpec, ...], tuple[TrainingWindowSpec, ...]]:
    if strategy is SamplingStrategy.TASK_BALANCED:
        full_schedule = draw_task_balanced_windows(
            windows,
            steps=stage1_steps + stage2_steps,
            rng=rng,
        )
        return (
            full_schedule[:stage1_steps],
            full_schedule[stage1_steps:],
        )

    return (
        _draw_window_schedule(windows, steps=stage1_steps, rng=rng),
        _draw_window_schedule(windows, steps=stage2_steps, rng=rng),
    )


def _window_identity(spec: TrainingWindowSpec) -> dict[str, Any]:
    return {
        "prompt": spec.pair.prompt.fingerprint,
        "target": spec.pair.target.fingerprint,
        "anchor": spec.anchor_policy_position,
        "action_history_steps": spec.action_history_steps,
        "future_steps": spec.future_steps,
        "action_horizon": spec.action_horizon,
        "ifp_steps": spec.ifp_steps,
        "ifp_stride": spec.ifp_stride,
    }


def _schedule_sha256(
    stage1: Sequence[TrainingWindowSpec],
    stage2: Sequence[TrainingWindowSpec],
) -> str:
    return _json_sha256(
        {
            "stage1": [_window_identity(spec) for spec in stage1],
            "stage2": [_window_identity(spec) for spec in stage2],
        }
    )


def _state_dict_sha256(model: CompactWAM) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _publish_candidate_pair(
    *,
    model: CompactWAM,
    metadata: Mapping[str, Any],
    report: Mapping[str, Any],
    checkpoint_path: Path,
    report_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() or report_path.exists():
        raise TrainingArtifactExistsError(
            "training artifacts are immutable; choose new checkpoint/report paths"
        )

    with NamedTemporaryFile(
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as checkpoint_temp_file:
        temp_checkpoint = Path(checkpoint_temp_file.name)
    with NamedTemporaryFile(
        dir=report_path.parent,
        prefix=f".{report_path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as report_temp_file:
        temp_report = Path(report_temp_file.name)
    try:
        save_compact_wam_checkpoint(model, temp_checkpoint, metadata=metadata)
        bundle = load_compact_wam_bundle(temp_checkpoint, device="cpu")
        if bundle.metadata.get("trained") is not False:
            raise TrainingError("offline candidate checkpoint must store trained=false")
        checkpoint_sha256 = _file_sha256(temp_checkpoint)
        complete_report = {
            **dict(report),
            "artifacts": {
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_filename": checkpoint_path.name,
                "report_filename": report_path.name,
            },
        }
        with temp_report.open("w", encoding="utf-8") as stream:
            json.dump(
                complete_report,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with temp_report.open("r", encoding="utf-8") as stream:
            loaded_report = json.load(stream)
        if loaded_report != complete_report:
            raise TrainingError("training report round-trip validation failed")

        _link_no_overwrite(temp_checkpoint, checkpoint_path)
        try:
            _link_no_overwrite(temp_report, report_path)
        except BaseException:
            _unlink_if_same_inode(checkpoint_path, temp_checkpoint)
            raise
    finally:
        temp_checkpoint.unlink(missing_ok=True)
        temp_report.unlink(missing_ok=True)

    published = load_compact_wam_bundle(checkpoint_path, device="cpu")
    if published.metadata.get("training_evidence_sha256") != report.get(
        "training_evidence_sha256"
    ):
        raise TrainingError("published checkpoint/report evidence digest mismatch")
    with report_path.open("r", encoding="utf-8") as stream:
        published_report = json.load(stream)
    if published_report["artifacts"]["checkpoint_sha256"] != _file_sha256(
        checkpoint_path
    ):
        raise TrainingError("published checkpoint checksum mismatch")


def _link_no_overwrite(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except FileExistsError as error:
        raise TrainingArtifactExistsError(
            f"training artifact already exists: {target}"
        ) from error


def _unlink_if_same_inode(target: Path, source: Path) -> None:
    try:
        if target.stat().st_ino == source.stat().st_ino:
            target.unlink()
    except FileNotFoundError:
        pass


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_artifacts_as_json(artifacts: TrainingArtifacts) -> Mapping[str, Any]:
    """Return an immutable CLI-safe result mapping."""

    return MappingProxyType(asdict(artifacts))


__all__ = [
    "CompactWAMTrainingConfig",
    "IFPArchitecture",
    "PAPER_IFP_LOSS_WEIGHTS",
    "PAPER_IFP_STRIDE",
    "PAPER_IFP_STEPS",
    "SamplingStrategy",
    "TRAINING_REPORT_SCHEMA",
    "TrainingArtifactExistsError",
    "TrainingArtifacts",
    "TrainingError",
    "train_offline_candidate",
    "training_artifacts_as_json",
]
