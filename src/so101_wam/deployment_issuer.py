"""Fail-closed issuer for evidence-bound G10 deployment artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch

from .checkpoint import (
    CheckpointError,
    load_compact_wam_bundle,
    save_compact_wam_checkpoint,
)
from .config import ProjectConfig
from .constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from .dataset import (
    FPS_RELATIVE_TOLERANCE,
    DatasetError,
    load_episode,
    physical_prompt_from_episode,
)
from .deployment import (
    DEPLOYMENT_AUTHORIZATION_SCOPE,
    DEPLOYMENT_CERTIFICATION_KIND,
    DEPLOYMENT_CERTIFICATION_SCHEMA,
    DEPLOYMENT_SOURCE_EVIDENCE_KEYS,
    canonical_json_sha256,
    deployment_evidence_sha256,
    file_sha256,
    project_config_fingerprint,
    project_config_sha256,
    verify_deployment_certification,
)
from .prompt_recorder import ACTION_SOURCE, CAPTURE_MODE
from .training import TRAINING_REPORT_SCHEMA
from .mujoco_cli import run_mujoco_checkpoint_session


MANUAL_SIGNOFF_SCHEMA = "so101_wam.manual_hardware_signoff.v1"
MANUAL_SIGNOFF_KIND = "so101_wam.manual_hardware_signoff"
REQUIRED_MANUAL_CHECKS = (
    "left_right_wrist_camera_identity",
    "arms_down_pose_confirmed",
    "joint_direction_labels_confirmed",
    "per_tick_delta_limits_confirmed",
    "operator_halt_stops_commands",
    "cable_strain_relief_confirmed",
    "collision_clearance_confirmed",
    "real_slow_jog_no_contact",
)
_MANUAL_SOURCE_KEYS = tuple(
    key for key in DEPLOYMENT_SOURCE_EVIDENCE_KEYS if key != "manual_signoff"
)
_MANUAL_SIGNOFF_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "evidence_level",
        "result",
        "operator_id",
        "signed_at_utc",
        "hardware_id",
        "calibration_id",
        "config_sha256",
        "source_evidence_sha256",
        "checks",
    }
)


class DeploymentIssuanceError(RuntimeError):
    """Raised when source evidence cannot authorize a dry deployment artifact."""


class DeploymentArtifactExistsError(DeploymentIssuanceError):
    """Raised when immutable deployment output paths already exist."""


@dataclass(frozen=True, slots=True)
class IssuedDeploymentArtifacts:
    checkpoint_path: str
    certification_path: str
    checkpoint_id: str
    checkpoint_sha256: str
    deployment_evidence_sha256: str
    source_candidate_sha256: str
    max_policy_steps: int
    deployment_ready: bool = True


@dataclass(frozen=True, slots=True)
class ValidatedDeploymentSourceEvidence:
    """Fully revalidated source artifacts required by issuer and runtime."""

    candidate: Any
    candidate_sha256: str
    candidate_checkpoint_id: str
    training_evidence_sha256: str
    prompt_fingerprint: str
    source_evidence_sha256: Mapping[str, str]
    gate_evidence_sha256: Mapping[str, str]


def validate_deployment_source_evidence(
    config: ProjectConfig,
    *,
    candidate_checkpoint_path: str | Path,
    training_report_path: str | Path,
    preflight_report_path: str | Path,
    mujoco_config_path: str | Path,
    mujoco_report_path: str | Path,
    prompt_npz_path: str | Path,
    prompt_manifest_path: str | Path | None,
    manual_signoff_path: str | Path,
) -> ValidatedDeploymentSourceEvidence:
    """Recompute and validate the complete G6-G9 source-evidence chain."""

    _validate_source_config(config)
    candidate_path = Path(candidate_checkpoint_path).resolve()
    training_report = Path(training_report_path).resolve()
    preflight_report = Path(preflight_report_path).resolve()
    mujoco_config = ProjectConfig.load(mujoco_config_path)
    mujoco_report = Path(mujoco_report_path).resolve()
    prompt_npz = Path(prompt_npz_path).resolve()
    prompt_manifest = (
        Path(prompt_manifest_path).resolve()
        if prompt_manifest_path is not None
        else prompt_npz.with_suffix(".json")
    )
    manual_signoff = Path(manual_signoff_path).resolve()

    candidate_sha256 = file_sha256(candidate_path)
    training_report_sha256 = file_sha256(training_report)
    preflight_report_sha256 = file_sha256(preflight_report)
    mujoco_report_sha256 = file_sha256(mujoco_report)
    prompt_npz_sha256 = file_sha256(prompt_npz)
    prompt_manifest_sha256 = file_sha256(prompt_manifest)

    candidate, training_evidence = _validate_candidate_and_training_report(
        candidate_path,
        training_report,
        candidate_sha256=candidate_sha256,
    )
    if candidate.model.action_horizon != config.runtime.action_horizon:
        raise DeploymentIssuanceError(
            "candidate action_horizon does not match deployment config"
        )
    candidate_checkpoint_id = str(candidate.metadata["checkpoint_id"])
    prompt_fingerprint = _validate_g9_prompt(
        config,
        prompt_npz,
        prompt_manifest,
        expected_npz_sha256=prompt_npz_sha256,
    )
    _validate_preflight_report(config, _read_strict_json(preflight_report))
    mujoco_report_payload = _read_strict_json(mujoco_report)
    _validate_mujoco_report(
        mujoco_report_payload,
        candidate_sha256=candidate_sha256,
        candidate_checkpoint_id=candidate_checkpoint_id,
        prompt_fingerprint=prompt_fingerprint,
        mujoco_config_sha256=project_config_sha256(mujoco_config),
    )
    _rerun_mujoco_evidence(
        config=mujoco_config,
        report=mujoco_report_payload,
        candidate_path=candidate_path,
        prompt_npz_path=prompt_npz,
        prompt_manifest_path=prompt_manifest,
    )

    unsigned_sources = {
        "candidate_checkpoint": candidate_sha256,
        "training_report": training_report_sha256,
        "preflight_report": preflight_report_sha256,
        "mujoco_report": mujoco_report_sha256,
        "prompt_npz": prompt_npz_sha256,
        "prompt_manifest": prompt_manifest_sha256,
    }
    _validate_manual_signoff(
        config,
        _read_strict_json(manual_signoff),
        expected_sources=unsigned_sources,
    )
    manual_signoff_sha256 = file_sha256(manual_signoff)
    source_evidence_sha256 = {
        **unsigned_sources,
        "manual_signoff": manual_signoff_sha256,
    }
    if set(source_evidence_sha256) != set(DEPLOYMENT_SOURCE_EVIDENCE_KEYS):
        raise DeploymentIssuanceError(
            "internal deployment source-evidence mapping is incomplete"
        )
    _assert_source_artifacts_unchanged(
        {
            candidate_path: candidate_sha256,
            training_report: training_report_sha256,
            preflight_report: preflight_report_sha256,
            mujoco_report: mujoco_report_sha256,
            prompt_npz: prompt_npz_sha256,
            prompt_manifest: prompt_manifest_sha256,
            manual_signoff: manual_signoff_sha256,
        }
    )
    gate_evidence_sha256 = {
        "G6": _gate_digest(
            "G6", preflight_report_sha256, manual_signoff_sha256
        ),
        "G7": _gate_digest(
            "G7", preflight_report_sha256, manual_signoff_sha256
        ),
        "G8": _gate_digest(
            "G8", mujoco_report_sha256, manual_signoff_sha256
        ),
        "G9": _gate_digest(
            "G9", prompt_npz_sha256, prompt_manifest_sha256
        ),
    }
    return ValidatedDeploymentSourceEvidence(
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        candidate_checkpoint_id=candidate_checkpoint_id,
        training_evidence_sha256=training_evidence,
        prompt_fingerprint=prompt_fingerprint,
        source_evidence_sha256=source_evidence_sha256,
        gate_evidence_sha256=gate_evidence_sha256,
    )


def issue_deployment_artifacts(
    config: ProjectConfig,
    *,
    candidate_checkpoint_path: str | Path,
    training_report_path: str | Path,
    preflight_report_path: str | Path,
    mujoco_config_path: str | Path,
    mujoco_report_path: str | Path,
    prompt_npz_path: str | Path,
    prompt_manifest_path: str | Path | None,
    manual_signoff_path: str | Path,
    output_checkpoint_path: str | Path,
    output_certification_path: str | Path,
    checkpoint_id: str,
    max_policy_steps: int,
) -> IssuedDeploymentArtifacts:
    """Validate G6-G9 evidence and immutably publish a G10-scoped pair."""

    _validate_issuance_request(
        config,
        checkpoint_id=checkpoint_id,
        max_policy_steps=max_policy_steps,
    )
    output_checkpoint = Path(output_checkpoint_path).resolve()
    output_certification = Path(output_certification_path).resolve()
    if output_checkpoint == output_certification:
        raise DeploymentIssuanceError(
            "deployment checkpoint and certification paths must be different"
        )
    if output_checkpoint.exists() or output_certification.exists():
        raise DeploymentArtifactExistsError(
            "deployment artifacts are immutable; choose new output paths"
        )

    validated = validate_deployment_source_evidence(
        config,
        candidate_checkpoint_path=candidate_checkpoint_path,
        training_report_path=training_report_path,
        preflight_report_path=preflight_report_path,
        mujoco_config_path=mujoco_config_path,
        mujoco_report_path=mujoco_report_path,
        prompt_npz_path=prompt_npz_path,
        prompt_manifest_path=prompt_manifest_path,
        manual_signoff_path=manual_signoff_path,
    )
    candidate = validated.candidate
    candidate_sha256 = validated.candidate_sha256
    training_evidence = validated.training_evidence_sha256
    source_evidence_sha256 = dict(validated.source_evidence_sha256)
    gate_evidence_sha256 = dict(validated.gate_evidence_sha256)
    certification_core: dict[str, Any] = {
        "schema_version": DEPLOYMENT_CERTIFICATION_SCHEMA,
        "artifact_kind": DEPLOYMENT_CERTIFICATION_KIND,
        "evidence_level": "real",
        "result": "pass",
        "authorization_scope": DEPLOYMENT_AUTHORIZATION_SCOPE,
        "checkpoint_id": checkpoint_id,
        "training_evidence_sha256": training_evidence,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "config_sha256": project_config_sha256(config),
        "max_policy_steps": max_policy_steps,
        "gate_results": {gate: "pass" for gate in ("G6", "G7", "G8", "G9")},
        "gate_evidence_sha256": gate_evidence_sha256,
        "source_evidence_sha256": source_evidence_sha256,
    }
    deployment_digest = deployment_evidence_sha256(certification_core)
    metadata: dict[str, str | int | float | bool | None] = {
        "artifact_kind": "compact_wam_deployment",
        "evidence_level": "real",
        "trained": True,
        "deployment_ready": True,
        "checkpoint_id": checkpoint_id,
        "training_evidence_sha256": training_evidence,
        "deployment_evidence_sha256": deployment_digest,
        "source_candidate_checkpoint_sha256": candidate_sha256,
        "source_candidate_checkpoint_id": validated.candidate_checkpoint_id,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "authorization_scope": DEPLOYMENT_AUTHORIZATION_SCOPE,
        "max_policy_steps": max_policy_steps,
        "certification_schema": DEPLOYMENT_CERTIFICATION_SCHEMA,
    }
    _publish_deployment_pair(
        model=candidate.model,
        metadata=metadata,
        certification_core=certification_core,
        config=config,
        checkpoint_path=output_checkpoint,
        certification_path=output_certification,
        source_model=candidate.model,
    )
    checkpoint_sha256 = file_sha256(output_checkpoint)
    return IssuedDeploymentArtifacts(
        checkpoint_path=str(output_checkpoint),
        certification_path=str(output_certification),
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        deployment_evidence_sha256=deployment_digest,
        source_candidate_sha256=candidate_sha256,
        max_policy_steps=max_policy_steps,
    )


def _validate_issuance_request(
    config: ProjectConfig,
    *,
    checkpoint_id: str,
    max_policy_steps: int,
) -> None:
    _validate_source_config(config)
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise DeploymentIssuanceError("checkpoint_id must be a non-empty string")
    if (
        not isinstance(max_policy_steps, int)
        or isinstance(max_policy_steps, bool)
        or max_policy_steps < 1
    ):
        raise DeploymentIssuanceError("max_policy_steps must be a positive integer")


def _validate_source_config(config: ProjectConfig) -> None:
    if config.runtime.backend != "lerobot":
        raise DeploymentIssuanceError(
            "deployment issuance requires runtime.backend='lerobot'"
        )
    if not config.runtime.actuation_enabled:
        raise DeploymentIssuanceError(
            "deployment issuance requires runtime.actuation_enabled=true"
        )
    config.lerobot.require_actuation_identity()


def _validate_candidate_and_training_report(
    candidate_path: Path,
    report_path: Path,
    *,
    candidate_sha256: str,
) -> tuple[Any, str]:
    try:
        candidate = load_compact_wam_bundle(candidate_path, device="cpu")
    except CheckpointError as error:
        raise DeploymentIssuanceError(f"candidate checkpoint is invalid: {error}") from error
    metadata = candidate.metadata
    expected_markers = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "offline_trained": True,
        "trained": False,
        "deployment_ready": False,
    }
    for key, expected in expected_markers.items():
        if metadata.get(key) != expected:
            raise DeploymentIssuanceError(
                f"source checkpoint requires {key}={expected!r}"
            )
    candidate_checkpoint_id = metadata.get("checkpoint_id")
    if (
        not isinstance(candidate_checkpoint_id, str)
        or not candidate_checkpoint_id.strip()
    ):
        raise DeploymentIssuanceError(
            "source checkpoint requires a non-empty string checkpoint_id"
        )

    report = _read_strict_json(report_path)
    required_report_markers = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
    }
    for key, expected in required_report_markers.items():
        if report.get(key) != expected:
            raise DeploymentIssuanceError(
                f"training report requires {key}={expected!r}"
            )
    if report.get("checkpoint_id") != metadata.get("checkpoint_id"):
        raise DeploymentIssuanceError(
            "training report checkpoint_id does not match candidate"
        )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get(
        "checkpoint_sha256"
    ) != candidate_sha256:
        raise DeploymentIssuanceError(
            "training report does not bind the exact candidate checkpoint"
        )
    training_evidence = report.get("training_evidence_sha256")
    if training_evidence != metadata.get("training_evidence_sha256"):
        raise DeploymentIssuanceError(
            "training evidence digest does not match candidate metadata"
        )
    report_core = {
        key: value
        for key, value in report.items()
        if key not in {"training_evidence_sha256", "artifacts"}
    }
    if training_evidence != canonical_json_sha256(report_core):
        raise DeploymentIssuanceError("training report core digest is invalid")
    if report.get("model") != dict(candidate.architecture):
        raise DeploymentIssuanceError(
            "training report architecture does not match candidate"
        )
    protocol = report.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("split") != "task_disjoint"
        or protocol.get("real_output_authorized") is not False
    ):
        raise DeploymentIssuanceError(
            "training report protocol is not a task-disjoint offline candidate"
        )
    data = report.get("data")
    if not isinstance(data, Mapping) or any(
        not isinstance(data.get(key), int)
        or isinstance(data.get(key), bool)
        or int(data[key]) < 1
        for key in ("train_task_count", "validation_task_count")
    ):
        raise DeploymentIssuanceError(
            "training report requires non-empty train and held-out validation tasks"
        )
    validation = report.get("validation")
    if not isinstance(validation, Mapping):
        raise DeploymentIssuanceError("training report validation metrics are missing")
    for key in (
        "action_mse_normalized",
        "action_mae_native",
        "future_latent_mse",
    ):
        value = validation.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or float(value) < 0
        ):
            raise DeploymentIssuanceError(
                f"training report validation metric {key} is invalid"
            )
    if not _is_sha256(training_evidence):
        raise DeploymentIssuanceError("training_evidence_sha256 is invalid")
    return candidate, str(training_evidence)


def _validate_preflight_report(
    config: ProjectConfig,
    report: Mapping[str, Any],
) -> None:
    required = {
        "schema_version": 1,
        "evidence_level": "real",
        "gate": "G6/G7-preflight",
        "mode": "observation_only",
        "result": "partial",
        "automated_checks_passed": True,
        "goal_position_commands_sent": 0,
        "effective_actuation_enabled": False,
        "lerobot_calibrated": True,
        "home_within_tolerance": True,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise DeploymentIssuanceError(
                f"G6/G7 preflight requires {key}={expected!r}"
            )
    if report.get("failures") != []:
        raise DeploymentIssuanceError("G6/G7 preflight failures must be empty")
    if report.get("hardware_id") != config.lerobot.hardware_id:
        raise DeploymentIssuanceError("preflight hardware_id does not match config")
    if report.get("calibration_id") != config.lerobot.calibration_id:
        raise DeploymentIssuanceError(
            "preflight calibration_id does not match config"
        )
    configured_actuation = report.get("configured_actuation_enabled")
    if not isinstance(configured_actuation, bool):
        raise DeploymentIssuanceError(
            "preflight configured_actuation_enabled must be a boolean"
        )
    preflight_config = replace(
        config,
        runtime=replace(
            config.runtime,
            actuation_enabled=configured_actuation,
        ),
    )
    if report.get("config_fingerprint") != project_config_fingerprint(
        preflight_config
    ):
        raise DeploymentIssuanceError(
            "preflight config fingerprint does not match deployment config"
        )
    if report.get("camera_keys") != list(PRIMARY_CAMERA_KEYS):
        raise DeploymentIssuanceError("preflight camera keys do not match core views")
    expected_resolution = [
        config.lerobot.camera_height,
        config.lerobot.camera_width,
    ]
    if report.get("camera_resolution") != expected_resolution or report.get(
        "camera_resolution_expected"
    ) != expected_resolution:
        raise DeploymentIssuanceError(
            "preflight camera resolution does not match deployment config"
        )
    frame_count = _positive_int_field(report, "frame_count", source="preflight")
    if report.get("camera_timestamp_samples") != frame_count:
        raise DeploymentIssuanceError(
            "preflight camera timestamp samples must cover every frame"
        )
    fps_floor = config.lerobot.camera_fps * 0.95
    sample_hz = _finite_number_field(report, "sample_hz", source="preflight")
    observation_fps = _finite_number_field(
        report,
        "observation_sampling_fps",
        source="preflight",
    )
    if sample_hz < fps_floor or observation_fps < sample_hz * 0.95:
        raise DeploymentIssuanceError(
            "preflight sampling rates are below the configured camera rate"
        )
    if report.get("observation_latency_limit_s") != (
        config.safety.max_observation_age_s
    ) or report.get("camera_frame_age_limit_s") != (
        config.safety.max_observation_age_s
    ):
        raise DeploymentIssuanceError(
            "preflight observation/camera-age limits do not match config"
        )
    if report.get("camera_skew_limit_s") != config.safety.max_camera_skew_s:
        raise DeploymentIssuanceError(
            "preflight camera-skew limit does not match config"
        )
    for key, limit in (
        ("observation_latency_p95_s", config.safety.max_observation_age_s),
        ("camera_frame_age_p95_s", config.safety.max_observation_age_s),
        ("camera_skew_p95_s", config.safety.max_camera_skew_s),
    ):
        if _finite_number_field(report, key, source="preflight") > limit:
            raise DeploymentIssuanceError(
                f"preflight {key} exceeds the deployment config limit"
            )
    camera_fps = report.get("camera_fps_estimate_hz")
    if not isinstance(camera_fps, Mapping) or set(camera_fps) != set(
        PRIMARY_CAMERA_KEYS
    ):
        raise DeploymentIssuanceError(
            "preflight camera FPS evidence must cover both wrist cameras"
        )
    for key in PRIMARY_CAMERA_KEYS:
        if _finite_number_field(
            camera_fps,
            key,
            source="preflight camera FPS",
        ) < fps_floor:
            raise DeploymentIssuanceError(
                f"preflight {key} FPS is below the configured camera rate"
            )
    if report.get("joint_order_width") != ACTION_DIM or report.get(
        "home_reference_configured"
    ) is not True:
        raise DeploymentIssuanceError(
            "preflight requires the complete measured home reference"
        )
    home_error = _finite_vector_field(
        report,
        "home_error_max_abs",
        source="preflight",
        size=ACTION_DIM,
    )
    tolerance = config.lerobot.home_joint_tolerance
    assert tolerance is not None
    if any(error > limit for error, limit in zip(home_error, tolerance, strict=True)):
        raise DeploymentIssuanceError(
            "preflight home errors exceed deployment config tolerance"
        )


def _validate_mujoco_report(
    report: Mapping[str, Any],
    *,
    candidate_sha256: str,
    candidate_checkpoint_id: str,
    prompt_fingerprint: str,
    mujoco_config_sha256: str,
) -> None:
    required = {
        "schema_version": 1,
        "gate": "G8-simulation",
        "result": "pass",
        "mode": "mujoco",
        "evidence_level": "simulation",
        "config_sha256": mujoco_config_sha256,
        "policy": "compact_wam",
        "collision_gate": True,
        "checkpoint_sha256": candidate_sha256,
        "checkpoint_id": candidate_checkpoint_id,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise DeploymentIssuanceError(
                f"G8 MuJoCo report requires {key}={expected!r}"
            )
    if report.get("primary_cameras") != list(PRIMARY_CAMERA_KEYS):
        raise DeploymentIssuanceError("G8 MuJoCo report camera keys are invalid")
    rollout = report.get("rollout")
    if not isinstance(rollout, Mapping):
        raise DeploymentIssuanceError(
            "G8 MuJoCo report does not prove a matching collision-gated rollout"
        )
    policy_steps = _positive_int_field(rollout, "policy_steps", source="G8 rollout")
    servo_steps = _positive_int_field(rollout, "servo_steps", source="G8 rollout")
    sent_actions = _positive_int_field(rollout, "sent_actions", source="G8 rollout")
    if (
        rollout.get("final_state") != "rollout_ready"
        or rollout.get("prompt_fingerprint") != prompt_fingerprint
        or rollout.get("shadow_steps") != 0
        or policy_steps > servo_steps
        or sent_actions != servo_steps
    ):
        raise DeploymentIssuanceError(
            "G8 MuJoCo report does not prove a matching actuated rollout"
        )
    physics_steps_per_tick = _positive_int_field(
        report,
        "physics_steps_per_servo_tick",
        source="G8 report",
    )
    physics_steps = _positive_int_field(
        report,
        "physics_steps",
        source="G8 report",
    )
    if physics_steps != sent_actions * physics_steps_per_tick:
        raise DeploymentIssuanceError(
            "G8 physics-step count does not match the simulated actions"
        )


def _rerun_mujoco_evidence(
    *,
    config: ProjectConfig,
    report: Mapping[str, Any],
    candidate_path: Path,
    prompt_npz_path: Path,
    prompt_manifest_path: Path,
) -> None:
    rollout = report.get("rollout")
    if not isinstance(rollout, Mapping):
        raise DeploymentIssuanceError("G8 rollout evidence is missing")
    policy_steps = _positive_int_field(
        rollout,
        "policy_steps",
        source="G8 rollout",
    )
    try:
        reproduced = run_mujoco_checkpoint_session(
            config,
            checkpoint_path=candidate_path,
            prompt_path=prompt_npz_path,
            manifest_path=prompt_manifest_path,
            policy_steps=policy_steps,
            device="cpu",
        )
    except Exception as error:
        raise DeploymentIssuanceError(
            f"G8 MuJoCo evidence could not be reproduced: {error}"
        ) from error
    if reproduced != dict(report):
        raise DeploymentIssuanceError(
            "G8 stored report does not match an exact MuJoCo reproduction"
        )


def _validate_g9_prompt(
    config: ProjectConfig,
    npz_path: Path,
    manifest_path: Path,
    *,
    expected_npz_sha256: str,
) -> str:
    try:
        episode = load_episode(npz_path, manifest_path)
        prompt = physical_prompt_from_episode(
            episode,
            policy_hz=config.runtime.policy_hz,
        )
    except DatasetError as error:
        raise DeploymentIssuanceError(f"G9 prompt is invalid: {error}") from error
    metadata = episode.metadata or {}
    required = {
        "gate": "G9-prompt-record",
        "evidence_level": "real",
        "capture_mode": CAPTURE_MODE,
        "action_source": ACTION_SOURCE,
        "goal_position_commands_sent": 0,
        "torque_disabled_for_capture": True,
        "head_camera_included": False,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "prompt_fingerprint": prompt.fingerprint,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise DeploymentIssuanceError(
                f"G9 prompt metadata requires {key}={expected!r}"
            )
    if not np.array_equal(episode.action, episode.joint_state):
        raise DeploymentIssuanceError(
            "G9 prompt action must equal measured joint state for no-write capture"
        )
    if episode.fps != config.runtime.camera_hz or metadata.get("fps") != episode.fps:
        raise DeploymentIssuanceError(
            "G9 prompt FPS does not match the deployment camera rate"
        )
    if not (
        config.runtime.prompt_min_seconds
        <= prompt.duration_s
        <= config.runtime.prompt_max_seconds
    ):
        raise DeploymentIssuanceError(
            "G9 prompt duration is outside the deployment config bounds"
        )
    if (
        metadata.get("duration_recorded_s") != prompt.duration_s
        or metadata.get("frame_count") != episode.frame_count
        or metadata.get("camera_resolution")
        != [config.lerobot.camera_height, config.lerobot.camera_width]
        or metadata.get("joint_width") != ACTION_DIM
        or metadata.get("policy_hz_verified") != config.runtime.policy_hz
        or metadata.get("policy_prompt_frame_count") != len(prompt.frames)
        or metadata.get("fps_tolerance") != FPS_RELATIVE_TOLERANCE
    ):
        raise DeploymentIssuanceError(
            "G9 prompt capture contract does not match the episode/config"
        )
    if _finite_number_field(
        metadata,
        "capture_interval_max_relative_error",
        source="G9 prompt",
    ) > FPS_RELATIVE_TOLERANCE:
        raise DeploymentIssuanceError(
            "G9 prompt capture interval exceeds the allowed drift"
        )
    for key, limit in (
        ("observation_latency_p95_s", config.safety.max_observation_age_s),
        ("camera_frame_age_p95_s", config.safety.max_observation_age_s),
        ("camera_skew_p95_s", config.safety.max_camera_skew_s),
    ):
        if _finite_number_field(metadata, key, source="G9 prompt") > limit:
            raise DeploymentIssuanceError(f"G9 prompt {key} exceeds config limit")
    camera_fps = metadata.get("camera_fps_estimate_hz")
    camera_interval_error = metadata.get("camera_interval_max_relative_error")
    if (
        not isinstance(camera_fps, Mapping)
        or set(camera_fps) != set(PRIMARY_CAMERA_KEYS)
        or not isinstance(camera_interval_error, Mapping)
        or set(camera_interval_error) != set(PRIMARY_CAMERA_KEYS)
    ):
        raise DeploymentIssuanceError(
            "G9 prompt timing evidence must cover both wrist cameras"
        )
    fps_floor = config.runtime.camera_hz * (1.0 - FPS_RELATIVE_TOLERANCE)
    for key in PRIMARY_CAMERA_KEYS:
        if _finite_number_field(
            camera_fps,
            key,
            source="G9 camera FPS",
        ) < fps_floor:
            raise DeploymentIssuanceError(f"G9 prompt {key} FPS is too low")
        if _finite_number_field(
            camera_interval_error,
            key,
            source="G9 camera interval",
        ) > FPS_RELATIVE_TOLERANCE:
            raise DeploymentIssuanceError(
                f"G9 prompt {key} camera interval drift is too high"
            )
    manifest = _read_strict_json(manifest_path)
    if manifest.get("npz_sha256") != expected_npz_sha256:
        raise DeploymentIssuanceError("G9 manifest does not bind prompt NPZ bytes")
    return prompt.fingerprint


def _validate_manual_signoff(
    config: ProjectConfig,
    signoff: Mapping[str, Any],
    *,
    expected_sources: Mapping[str, str],
) -> None:
    if set(signoff) != _MANUAL_SIGNOFF_KEYS:
        raise DeploymentIssuanceError("manual signoff keys do not match schema")
    required = {
        "schema_version": MANUAL_SIGNOFF_SCHEMA,
        "artifact_kind": MANUAL_SIGNOFF_KIND,
        "evidence_level": "real",
        "result": "pass",
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "config_sha256": project_config_sha256(config),
    }
    for key, expected in required.items():
        if signoff.get(key) != expected:
            raise DeploymentIssuanceError(
                f"manual signoff requires {key}={expected!r}"
            )
    operator_id = signoff.get("operator_id")
    if not isinstance(operator_id, str) or not operator_id.strip():
        raise DeploymentIssuanceError(
            "manual signoff operator_id must be a non-empty string"
        )
    signed_at = signoff.get("signed_at_utc")
    if not isinstance(signed_at, str):
        raise DeploymentIssuanceError("manual signoff signed_at_utc is invalid")
    try:
        parsed = datetime.fromisoformat(signed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise DeploymentIssuanceError(
            "manual signoff signed_at_utc is not ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DeploymentIssuanceError(
            "manual signoff signed_at_utc must include a timezone"
        )
    sources = signoff.get("source_evidence_sha256")
    if not isinstance(sources, Mapping) or set(sources) != set(
        _MANUAL_SOURCE_KEYS
    ):
        raise DeploymentIssuanceError(
            "manual signoff source evidence keys do not match schema"
        )
    if dict(sources) != dict(expected_sources):
        raise DeploymentIssuanceError(
            "manual signoff is not bound to the supplied evidence artifacts"
        )
    checks = signoff.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(
        REQUIRED_MANUAL_CHECKS
    ):
        raise DeploymentIssuanceError(
            "manual signoff checks do not match the required hardware checks"
        )
    if any(value is not True for value in checks.values()):
        raise DeploymentIssuanceError(
            "manual signoff requires every hardware check to be true"
        )


def _publish_deployment_pair(
    *,
    model: Any,
    metadata: Mapping[str, Any],
    certification_core: Mapping[str, Any],
    config: ProjectConfig,
    checkpoint_path: Path,
    certification_path: Path,
    source_model: Any,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    certification_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() or certification_path.exists():
        raise DeploymentArtifactExistsError(
            "deployment artifacts are immutable; choose new output paths"
        )
    with NamedTemporaryFile(
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as checkpoint_file:
        temp_checkpoint = Path(checkpoint_file.name)
    with NamedTemporaryFile(
        dir=certification_path.parent,
        prefix=f".{certification_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as certification_file:
        temp_certification = Path(certification_file.name)
    try:
        save_compact_wam_checkpoint(model, temp_checkpoint, metadata=metadata)
        checkpoint_sha256 = file_sha256(temp_checkpoint)
        certification = {
            **dict(certification_core),
            "checkpoint_sha256": checkpoint_sha256,
        }
        _write_json(temp_certification, certification)
        published_bundle = load_compact_wam_bundle(temp_checkpoint, device="cpu")
        verify_exact_model_state(source_model, published_bundle.model)
        verify_deployment_certification(
            temp_certification,
            checkpoint_sha256=checkpoint_sha256,
            source_evidence_sha256=dict(
                certification_core["source_evidence_sha256"]
            ),
            checkpoint_metadata=published_bundle.metadata,
            config=config,
            policy_steps=int(certification_core["max_policy_steps"]),
        )
        _link_no_overwrite(temp_checkpoint, checkpoint_path)
        try:
            _link_no_overwrite(temp_certification, certification_path)
        except BaseException:
            _unlink_if_same_inode(checkpoint_path, temp_checkpoint)
            raise
        try:
            final_bundle = load_compact_wam_bundle(checkpoint_path, device="cpu")
            verify_exact_model_state(source_model, final_bundle.model)
            verify_deployment_certification(
                certification_path,
                checkpoint_sha256=file_sha256(checkpoint_path),
                source_evidence_sha256=dict(
                    certification_core["source_evidence_sha256"]
                ),
                checkpoint_metadata=final_bundle.metadata,
                config=config,
                policy_steps=int(certification_core["max_policy_steps"]),
            )
        except BaseException:
            _unlink_if_same_inode(certification_path, temp_certification)
            _unlink_if_same_inode(checkpoint_path, temp_checkpoint)
            raise
    finally:
        temp_checkpoint.unlink(missing_ok=True)
        temp_certification.unlink(missing_ok=True)


def verify_exact_model_state(source: Any, target: Any) -> None:
    """Fail unless two CompactWAM instances have byte-identical state tensors."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise DeploymentIssuanceError(
            "deployment checkpoint changed the candidate state keys"
        )
    if any(
        not torch.equal(source_state[key].cpu(), target_state[key].cpu())
        for key in source_state
    ):
        raise DeploymentIssuanceError(
            "deployment checkpoint changed candidate model weights"
        )


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DeploymentIssuanceError(f"cannot read JSON artifact {path}: {error}") from error
    if len(raw) > 16 * 1024 * 1024:
        raise DeploymentIssuanceError(f"JSON artifact exceeds 16 MiB: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DeploymentIssuanceError(
                    f"JSON artifact contains duplicate key {key!r}: {path}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_raise_invalid_json_constant(value, path)),
        )
    except DeploymentIssuanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentIssuanceError(
            f"JSON artifact is invalid {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise DeploymentIssuanceError(f"JSON artifact root must be an object: {path}")
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _raise_invalid_json_constant(value: str, path: Path) -> None:
    raise DeploymentIssuanceError(
        f"JSON artifact contains non-standard numeric constant {value!r}: {path}"
    )


def _assert_source_artifacts_unchanged(expected: Mapping[Path, str]) -> None:
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise DeploymentIssuanceError(
                f"source evidence changed during deployment issuance: {path}"
            )


def _link_no_overwrite(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except FileExistsError as error:
        raise DeploymentArtifactExistsError(
            f"deployment artifact already exists: {target}"
        ) from error


def _unlink_if_same_inode(target: Path, source: Path) -> None:
    try:
        target_stat = target.stat()
        source_stat = source.stat()
    except FileNotFoundError:
        return
    if (target_stat.st_dev, target_stat.st_ino) == (
        source_stat.st_dev,
        source_stat.st_ino,
    ):
        target.unlink()


def _gate_digest(gate: str, *artifact_digests: str) -> str:
    return canonical_json_sha256(
        {
            "gate": gate,
            "artifact_sha256": list(artifact_digests),
        }
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number_field(
    value: Mapping[str, Any],
    key: str,
    *,
    source: str,
) -> float:
    number = value.get(key)
    if (
        not isinstance(number, (int, float))
        or isinstance(number, bool)
        or not isfinite(float(number))
        or float(number) < 0
    ):
        raise DeploymentIssuanceError(
            f"{source} {key} must be a finite non-negative number"
        )
    return float(number)


def _positive_int_field(
    value: Mapping[str, Any],
    key: str,
    *,
    source: str,
) -> int:
    number = value.get(key)
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise DeploymentIssuanceError(
            f"{source} {key} must be a positive integer"
        )
    return number


def _finite_vector_field(
    value: Mapping[str, Any],
    key: str,
    *,
    source: str,
    size: int,
) -> tuple[float, ...]:
    vector = value.get(key)
    if not isinstance(vector, list) or len(vector) != size:
        raise DeploymentIssuanceError(
            f"{source} {key} must contain exactly {size} values"
        )
    return tuple(
        _finite_number_field({key: item}, key, source=source) for item in vector
    )


def issued_artifacts_as_json(
    artifacts: IssuedDeploymentArtifacts,
) -> Mapping[str, Any]:
    return asdict(artifacts)


__all__ = [
    "DeploymentArtifactExistsError",
    "DeploymentIssuanceError",
    "IssuedDeploymentArtifacts",
    "MANUAL_SIGNOFF_KIND",
    "MANUAL_SIGNOFF_SCHEMA",
    "REQUIRED_MANUAL_CHECKS",
    "ValidatedDeploymentSourceEvidence",
    "issue_deployment_artifacts",
    "issued_artifacts_as_json",
    "validate_deployment_source_evidence",
    "verify_exact_model_state",
]
