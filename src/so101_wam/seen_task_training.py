"""Same-task simulator reconstruction diagnostic training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import compact_wam_architecture
from .constants import ACTION_DIM, JOINT_KEYS
from .deployment import canonical_json_sha256
from .model import ActionRangeConstraint, CompactWAM
from .training import (
    TRAINING_REPORT_SCHEMA,
    CompactWAMTrainingConfig,
    IFPArchitecture,
    TrainingError,
    _draw_stage_schedules,
    _evaluate,
    _publish_candidate_pair,
    _schedule_sha256,
    _seed_everything,
    _state_dict_sha256,
    _train_end_to_end_stage,
    _train_inverse_dynamics_stage,
)
from .training_data import (
    ACTION_TIMING_KEY,
    OBSERVATION_THEN_COMMAND,
    EpisodeRecord,
    build_training_windows,
    episode_split_digest,
    task_draw_counts,
    training_axis_statistics,
)


SEEN_TASK_SCHEMA = "so101_wam.seen_task_diagnostic.v2"
SEEN_TASK_KIND = "compact_wam_seen_task_diagnostic"
SOURCE_KIND = "mujoco_reference"


class SeenTaskTrainingError(TrainingError):
    """Raised when seen-task diagnostic training cannot be trusted."""


@dataclass(frozen=True, slots=True)
class SeenTaskArtifacts:
    checkpoint_path: str
    report_path: str
    checkpoint_id: str
    training_evidence_sha256: str
    episode_count: int
    window_count: int
    optimizer_steps: int
    trained: bool = False
    deployment_ready: bool = False


def train_seen_task_candidate(
    records: Sequence[EpisodeRecord],
    *,
    checkpoint_path: str | Path,
    report_path: str | Path,
    checkpoint_id: str,
    config: CompactWAMTrainingConfig,
    device: str | torch.device = "cpu",
    action_range_constraint: ActionRangeConstraint = ActionRangeConstraint.UNBOUNDED,
    joint_lower: Sequence[float] | None = None,
    joint_upper: Sequence[float] | None = None,
) -> SeenTaskArtifacts:
    """Train one non-deployable same-task reconstruction diagnostic."""

    checkpoint_target = Path(checkpoint_path).resolve()
    report_target = Path(report_path).resolve()
    _require_artifacts(checkpoint_target, report_target)
    checkpoint_id = _checkpoint_id(checkpoint_id)
    target_device = _cpu_device(device)
    _require_config(config)
    source = _require_records(records)
    output_contract = _action_output_contract(action_range_constraint, joint_lower, joint_upper)
    if action_range_constraint is not ActionRangeConstraint.UNBOUNDED:
        lower = np.asarray(joint_lower, dtype=np.float64)
        upper = np.asarray(joint_upper, dtype=np.float64)
        for record in source:
            if np.any(record.data.action < lower) or np.any(record.data.action > upper):
                raise SeenTaskTrainingError("reference actions exceed output bounds")

    windows = build_training_windows(
        source,
        policy_hz=config.policy_hz,
        servo_hz=config.servo_hz,
        action_history_steps=config.action_history_steps,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        ifp_steps=config.effective_ifp_window_steps,
        ifp_stride=config.ifp_stride,
        max_context_steps=config.max_context_steps,
    )
    axis_mean, axis_scale = training_axis_statistics(source)
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        _seed_everything(config.seed)
        model = CompactWAM(
            latent_dim=config.latent_dim,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
            future_steps=config.future_steps,
            action_horizon=config.action_horizon,
            action_history_steps=config.action_history_steps,
            ifp_steps=config.ifp_steps,
            max_context_steps=config.max_context_steps,
            action_decoder=config.action_decoder,
            action_range_constraint=action_range_constraint,
            action_lower=joint_lower,
            action_upper=joint_upper,
        ).to(target_device)
        model.set_axis_normalization(
            torch.from_numpy(axis_mean),
            torch.from_numpy(axis_scale),
        )
        initial_sha = _state_dict_sha256(model)
        pre_metrics = _evaluate(model, windows, config=config, device=target_device)

        _seed_everything(config.seed)
        rng = np.random.default_rng(config.seed)
        stage1, stage2 = _draw_stage_schedules(
            windows,
            stage1_steps=config.stage1_steps,
            stage2_steps=config.stage2_steps,
            strategy=config.sampling_strategy,
            rng=rng,
        )
        schedule = stage1 + stage2
        schedule_sha = _schedule_sha256(stage1, stage2)
        stage1_loss = _train_inverse_dynamics_stage(
            model,
            stage1,
            config=config,
            device=target_device,
        )
        stage2_metrics = _train_end_to_end_stage(
            model,
            stage2,
            config=config,
            device=target_device,
            fused_ifp=None,
        )
        post_metrics = _evaluate(model, windows, config=config, device=target_device)
        report_core = _report_core(
            checkpoint_id=checkpoint_id,
            config=config,
            model=model,
            records=source,
            windows=len(windows),
            schedule=schedule,
            schedule_sha=schedule_sha,
            initial_sha=initial_sha,
            axis_mean=axis_mean,
            axis_scale=axis_scale,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            stage1_loss=stage1_loss,
            stage2_metrics=stage2_metrics,
            device=target_device,
            output_contract=output_contract,
        )
        evidence_sha = canonical_json_sha256(report_core)
        _publish_candidate_pair(
            model=model,
            metadata=_metadata(
                checkpoint_id=checkpoint_id,
                evidence_sha=evidence_sha,
                config=config,
                schedule_sha=schedule_sha,
                post_metrics=post_metrics,
                output_contract=output_contract,
            ),
            report={**report_core, "training_evidence_sha256": evidence_sha},
            checkpoint_path=checkpoint_target,
            report_path=report_target,
        )
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)

    return SeenTaskArtifacts(
        checkpoint_path=str(checkpoint_target),
        report_path=str(report_target),
        checkpoint_id=checkpoint_id,
        training_evidence_sha256=evidence_sha,
        episode_count=len(source),
        window_count=len(windows),
        optimizer_steps=config.stage1_steps + config.stage2_steps,
    )


def _report_core(
    *,
    checkpoint_id: str,
    config: CompactWAMTrainingConfig,
    model: CompactWAM,
    records: Sequence[EpisodeRecord],
    windows: int,
    schedule: Sequence[Any],
    schedule_sha: str,
    initial_sha: str,
    axis_mean: np.ndarray,
    axis_scale: np.ndarray,
    pre_metrics: Mapping[str, float],
    post_metrics: Mapping[str, float],
    stage1_loss: float | None,
    stage2_metrics: Mapping[str, float],
    device: torch.device,
    output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    task = records[0].data
    return {
        "schema_version": SEEN_TASK_SCHEMA,
        "training_report_schema": TRAINING_REPORT_SCHEMA,
        "evidence_level": "offline_seen_task_diagnostic",
        "artifact_kind": SEEN_TASK_KIND,
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "zero_shot_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "semantic_heldout_success_claimed": False,
        "checkpoint_id": checkpoint_id,
        "protocol": {
            "split": "same_task_reconstruction",
            "prompt_pairing": "same_task_different_episode",
            "stage1": "ground_truth_future_latent_inverse_dynamics",
            "stage2": "seen_task_future_latent_action_mse",
            "source_kind_required": SOURCE_KIND,
            "reference_success_required": True,
            "action_timing_required": OBSERVATION_THEN_COMMAND,
            "real_output_authorized": False,
        },
        "data": {
            "episode_count": len(records),
            "window_count": windows,
            "task": task.task,
            "task_index": task.task_index,
            "source_kind": SOURCE_KIND,
            "action_timing": OBSERVATION_THEN_COMMAND,
            "episode_digest": episode_split_digest(records),
            "content_fingerprint_algorithm": "blake2b-128",
            "content_fingerprints": sorted(
                record.data.content_fingerprint for record in records
            ),
            "task_window_counts": task_draw_counts(schedule),
        },
        "model": compact_wam_architecture(model),
        "action_output": dict(output_contract),
        "initial_model_state_sha256": initial_sha,
        "training_schedule_sha256": schedule_sha,
        "normalization": {
            "axis_mean": [float(value) for value in axis_mean],
            "axis_scale": [float(value) for value in axis_scale],
        },
        "optimization": {
            **asdict(config),
            "device": str(device),
            "optimizer": "AdamW",
            "optimizer_steps": config.stage1_steps + config.stage2_steps,
            "stage1_final_action_mse_normalized": stage1_loss,
            **stage2_metrics,
        },
        "reconstruction": {
            "scope": "same_task_same_records_not_validation",
            "pre": dict(pre_metrics),
            "post": dict(post_metrics),
        },
        "limitations": [
            "same-task reconstruction is not task-disjoint validation",
            "simulator reference_success is a source declaration",
            "offline reconstruction does not authorize real robot output",
        ],
    }


def _metadata(
    *,
    checkpoint_id: str,
    evidence_sha: str,
    config: CompactWAMTrainingConfig,
    schedule_sha: str,
    post_metrics: Mapping[str, float],
    output_contract: Mapping[str, Any],
) -> dict[str, str | int | float | bool | None]:
    return {
        "artifact_kind": SEEN_TASK_KIND,
        "evidence_level": "offline_seen_task_diagnostic",
        "checkpoint_id": checkpoint_id,
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "zero_shot_claimed": False,
        "training_evidence_sha256": evidence_sha,
        "training_objective": "seen_task_reconstruction_diagnostic",
        "training_ifp_steps": config.ifp_steps,
        "action_history_steps": config.action_history_steps,
        "optimizer_steps": config.stage1_steps + config.stage2_steps,
        "seed": config.seed,
        "sampling_strategy": config.sampling_strategy.value,
        "sampling_schedule_sha256": schedule_sha,
        "reconstruction_action_mse_normalized": post_metrics[
            "action_mse_normalized"
        ],
        "action_decoder": config.action_decoder.value,
        "action_range_constraint": str(output_contract["mode"]),
        "action_output_sha256": canonical_json_sha256(output_contract),
    }


def _action_output_contract(
    mode: ActionRangeConstraint,
    lower: Sequence[float] | None,
    upper: Sequence[float] | None,
) -> dict[str, Any]:
    if not isinstance(mode, ActionRangeConstraint):
        raise SeenTaskTrainingError("action_range_constraint must be ActionRangeConstraint")
    if mode is ActionRangeConstraint.UNBOUNDED:
        if lower is not None or upper is not None:
            raise SeenTaskTrainingError("unbounded output cannot specify joint bounds")
        return {"mode": mode.value}
    if lower is None or upper is None:
        raise SeenTaskTrainingError("bounded output requires joint bounds")
    low, high = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if (low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,)
            or not np.isfinite(low).all() or not np.isfinite(high).all()
            or not np.all(low < high)):
        raise SeenTaskTrainingError("joint bounds must be finite ordered 12-axis limits")
    contract = {
        "mode": mode.value,
        "joint_order": list(JOINT_KEYS),
        "joint_lower": low.tolist(),
        "joint_upper": high.tolist(),
        "head_space": "logits",
        "endpoint_limitation": "finite mathematical logits cannot reach exact endpoints",
        "safety_supervisor": "unchanged",
    }
    if mode is ActionRangeConstraint.NORMALIZED_CLAMP:
        # Projection preserves normalized coordinates, but stops clipped gradients.
        contract.update({
            "head_space": "normalized_actions",
            "endpoint_limitation": "only float32-representable native endpoints are exact",
            "gradient_limitation": "zero outside the projected interval; no straight-through estimator",
        })
    return contract


def _require_records(
    records: Sequence[EpisodeRecord],
) -> tuple[EpisodeRecord, ...]:
    source = tuple(records)
    if len(source) < 2:
        raise SeenTaskTrainingError("seen-task diagnostic requires at least two records")
    tasks = {(record.data.task_index, record.data.task) for record in source}
    if len(tasks) != 1:
        raise SeenTaskTrainingError("seen-task diagnostic requires one same task")
    contents = {record.data.content_fingerprint for record in source}
    if len(contents) != len(source):
        raise SeenTaskTrainingError("seen-task diagnostic requires distinct contents")
    for record in source:
        _require_metadata(record)
    return source


def _require_metadata(record: EpisodeRecord) -> None:
    metadata = dict(record.data.metadata or {})
    if metadata.get("source_kind") != SOURCE_KIND:
        raise SeenTaskTrainingError("source_kind must be mujoco_reference")
    if metadata.get("reference_success") is not True:
        raise SeenTaskTrainingError("reference_success must be true")
    if metadata.get(ACTION_TIMING_KEY) != OBSERVATION_THEN_COMMAND:
        raise SeenTaskTrainingError(
            "action_timing must be observation_then_command"
        )


def _require_config(config: CompactWAMTrainingConfig) -> None:
    if not isinstance(config, CompactWAMTrainingConfig):
        raise SeenTaskTrainingError("config must be CompactWAMTrainingConfig")
    if config.ifp_architecture is not IFPArchitecture.COMPACT_LINEAR:
        raise SeenTaskTrainingError("seen-task diagnostic requires compact IFP")


def _require_artifacts(checkpoint_path: Path, report_path: Path) -> None:
    if checkpoint_path == report_path:
        raise SeenTaskTrainingError("checkpoint and report paths must be different")
    if checkpoint_path.exists() or report_path.exists():
        raise SeenTaskTrainingError(
            "training artifacts are immutable; choose new checkpoint/report paths"
        )


def _checkpoint_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SeenTaskTrainingError("checkpoint_id must be a non-empty string")
    return value.strip()


def _cpu_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise SeenTaskTrainingError(f"invalid training device: {value}") from error
    if device.type != "cpu":
        raise SeenTaskTrainingError("seen-task diagnostic requires cpu training")
    return device


__all__ = [
    "SeenTaskArtifacts",
    "SeenTaskTrainingError",
    "train_seen_task_candidate",
]
