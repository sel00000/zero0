from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import so101_wam.deployment_issuer as deployment_issuer_module
from so101_wam.certify_cli import main as certify_main
from so101_wam.checkpoint import (
    compact_wam_architecture,
    load_compact_wam_bundle,
    save_compact_wam_checkpoint,
)
from so101_wam.config import LeRobotConfig, ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer, physical_prompt_from_episode
from so101_wam.deployment import (
    DEPLOYMENT_AUTHORIZATION_SCOPE,
    DEPLOYMENT_CERTIFICATION_SCHEMA,
    DeploymentCertificationError,
    canonical_json_sha256,
    file_sha256,
    project_config_fingerprint,
    project_config_sha256,
    verify_deployment_certification,
)
from so101_wam.deployment_issuer import (
    MANUAL_SIGNOFF_KIND,
    MANUAL_SIGNOFF_SCHEMA,
    REQUIRED_MANUAL_CHECKS,
    DeploymentArtifactExistsError,
    DeploymentIssuanceError,
    issue_deployment_artifacts,
)
from so101_wam.hardware_cli import _require_deployment_checkpoint
from so101_wam.model import CompactWAM
from so101_wam.prompt_recorder import ACTION_SOURCE, CAPTURE_MODE
from so101_wam.training import TRAINING_REPORT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
_REAL_RERUN_MUJOCO_EVIDENCE = deployment_issuer_module._rerun_mujoco_evidence
requires_mujoco = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="optional mujoco dependency is absent",
)


@pytest.fixture(autouse=True)
def _stub_mujoco_reproduction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment_issuer_module,
        "_rerun_mujoco_evidence",
        lambda **kwargs: None,
    )


@dataclass(frozen=True)
class _Evidence:
    config: ProjectConfig
    config_path: Path
    candidate: Path
    training_report: Path
    preflight_report: Path
    mujoco_config: Path
    mujoco_report: Path
    prompt_npz: Path
    prompt_manifest: Path
    manual_signoff: Path


def _config() -> ProjectConfig:
    return ProjectConfig(
        runtime=RuntimeConfig(
            backend="lerobot",
            actuation_enabled=True,
            action_horizon=10,
        ),
        safety=SafetyConfig(
            joint_lower=(-10.0,) * ACTION_DIM,
            joint_upper=(10.0,) * ACTION_DIM,
            max_delta_per_servo_tick=(0.1,) * ACTION_DIM,
            calibrated=True,
        ),
        lerobot=LeRobotConfig(
            left_port="/dev/left-test",
            right_port="/dev/right-test",
            left_wrist_camera=0,
            right_wrist_camera=1,
            camera_width=8,
            camera_height=8,
            calibration_dir="/calibration/test",
            hardware_id="dual-so101-test-rig",
            calibration_id="calibration-test-v1",
            home_joint_position=(0.0,) * ACTION_DIM,
            home_joint_tolerance=(0.2,) * ACTION_DIM,
        ),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_config(path: Path) -> None:
    path.write_text(
        """
[runtime]
backend = "lerobot"
camera_hz = 30.0
policy_hz = 10.0
servo_hz = 50.0
context_seconds = 30.0
prompt_min_seconds = 3.0
prompt_max_seconds = 12.0
action_horizon = 10
primary_cameras = ["left_wrist", "right_wrist"]
use_head_camera = false
actuation_enabled = true

[safety]
calibrated = true
joint_lower = [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]
joint_upper = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
max_delta_per_servo_tick = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
max_observation_age_s = 0.10
max_camera_skew_s = 0.017
watchdog_timeout_s = 0.10

[lerobot]
version = "0.6.1"
robot_type = "bi_so_follower"
robot_id = "so101_wam"
left_port = "/dev/left-test"
right_port = "/dev/right-test"
left_wrist_camera = 0
right_wrist_camera = 1
camera_width = 8
camera_height = 8
camera_fps = 30
calibration_dir = "/calibration/test"
hardware_id = "dual-so101-test-rig"
calibration_id = "calibration-test-v1"
home_joint_position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
home_joint_tolerance = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _candidate_and_report(directory: Path) -> tuple[Path, Path, str]:
    model = CompactWAM(
        latent_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        future_steps=1,
        action_horizon=10,
        action_history_steps=1,
        ifp_steps=0,
        max_context_steps=300,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint_id = "offline-candidate-test"
    report_core = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": checkpoint_id,
        "protocol": {
            "split": "task_disjoint",
            "real_output_authorized": False,
        },
        "data": {"train_task_count": 1, "validation_task_count": 1},
        "model": compact_wam_architecture(model),
        "validation": {
            "action_mse_normalized": 0.1,
            "action_mae_native": 0.2,
            "future_latent_mse": 0.3,
        },
    }
    training_digest = canonical_json_sha256(report_core)
    candidate = directory / "candidate.pt"
    save_compact_wam_checkpoint(
        model,
        candidate,
        metadata={
            "artifact_kind": "compact_wam_candidate",
            "evidence_level": "offline",
            "checkpoint_id": checkpoint_id,
            "trained": False,
            "offline_trained": True,
            "deployment_ready": False,
            "training_evidence_sha256": training_digest,
        },
    )
    report = directory / "candidate.training.json"
    _write_json(
        report,
        {
            **report_core,
            "training_evidence_sha256": training_digest,
            "artifacts": {"checkpoint_sha256": file_sha256(candidate)},
        },
    )
    return candidate, report, checkpoint_id


def _prompt(
    directory: Path,
    config: ProjectConfig,
    *,
    metadata_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path, str]:
    buffer = EpisodeBuffer(
        fps=30.0,
        task="test physical prompt",
        task_index=1,
        episode_index=1,
        metadata={
            "gate": "G9-prompt-record",
            "evidence_level": "real",
            "capture_mode": CAPTURE_MODE,
            "action_source": ACTION_SOURCE,
            "goal_position_commands_sent": 0,
            "torque_disabled_for_capture": True,
            "head_camera_included": False,
            "hardware_id": config.lerobot.hardware_id,
            "calibration_id": config.lerobot.calibration_id,
            "duration_requested_s": 3.0,
            "duration_recorded_s": 3.0,
            "fps": 30.0,
            "fps_tolerance": 0.05,
            "frame_count": 91,
            "camera_resolution": [8, 8],
            "camera_fps_estimate_hz": {
                "left_wrist": 30.0,
                "right_wrist": 30.0,
            },
            "camera_interval_max_relative_error": {
                "left_wrist": 0.0,
                "right_wrist": 0.0,
            },
            "capture_interval_max_relative_error": 0.0,
            "observation_latency_p95_s": 0.001,
            "camera_frame_age_p95_s": 0.001,
            "camera_skew_p95_s": 0.001,
            "joint_width": ACTION_DIM,
        },
    )
    for frame_index in range(91):
        timestamp_s = frame_index / 30.0
        joint_position = np.full(
            ACTION_DIM,
            frame_index / 1000.0,
            dtype=np.float32,
        )
        buffer.append(
            SensorimotorFrame(
                timestamp_s=timestamp_s,
                images={
                    "left_wrist": np.full(
                        (8, 8, 3), frame_index % 255, dtype=np.uint8
                    ),
                    "right_wrist": np.full(
                        (8, 8, 3), (frame_index + 1) % 255, dtype=np.uint8
                    ),
                },
                joint_position=joint_position,
                executed_action=joint_position,
            )
        )
    prompt = physical_prompt_from_episode(
        buffer.to_episode_data(),
        policy_hz=config.runtime.policy_hz,
    )
    buffer.metadata["prompt_fingerprint"] = prompt.fingerprint
    buffer.metadata["policy_hz_verified"] = config.runtime.policy_hz
    buffer.metadata["policy_prompt_frame_count"] = len(prompt.frames)
    buffer.metadata.update(metadata_overrides or {})
    npz_path, manifest_path = buffer.save(directory, stem="prompt_000001")
    return npz_path, manifest_path, prompt.fingerprint


def _build_evidence(
    directory: Path,
    *,
    prompt_metadata_overrides: dict[str, object] | None = None,
) -> _Evidence:
    config = _config()
    config_path = directory / "hardware.toml"
    _write_config(config_path)
    candidate, training_report, candidate_id = _candidate_and_report(directory)
    prompt_npz, prompt_manifest, prompt_fingerprint = _prompt(
        directory,
        config,
        metadata_overrides=prompt_metadata_overrides,
    )

    preflight_report = directory / "preflight.json"
    _write_json(
        preflight_report,
        {
            "schema_version": 1,
            "evidence_level": "real",
            "gate": "G6/G7-preflight",
            "mode": "observation_only",
            "result": "partial",
            "automated_checks_passed": True,
            "failures": [],
            "hardware_id": config.lerobot.hardware_id,
            "calibration_id": config.lerobot.calibration_id,
            "config_fingerprint": project_config_fingerprint(config),
            "configured_actuation_enabled": True,
            "effective_actuation_enabled": False,
            "goal_position_commands_sent": 0,
            "lerobot_calibrated": True,
            "home_within_tolerance": True,
            "camera_keys": list(PRIMARY_CAMERA_KEYS),
            "camera_resolution": [8, 8],
            "camera_resolution_expected": [8, 8],
            "sample_hz": 30.0,
            "frame_count": 300,
            "observation_sampling_fps": 30.0,
            "observation_latency_p95_s": 0.001,
            "observation_latency_limit_s": 0.1,
            "camera_timestamp_samples": 300,
            "camera_fps_estimate_hz": {
                "left_wrist": 30.0,
                "right_wrist": 30.0,
            },
            "camera_frame_age_p95_s": 0.001,
            "camera_frame_age_limit_s": 0.1,
            "camera_skew_p95_s": 0.001,
            "camera_skew_limit_s": 0.017,
            "joint_order_width": ACTION_DIM,
            "home_reference_configured": True,
            "home_error_max_abs": [0.0] * ACTION_DIM,
        },
    )
    mujoco_config = ROOT / "configs" / "mujoco.toml"
    loaded_mujoco_config = ProjectConfig.load(mujoco_config)
    mujoco_report = directory / "mujoco.g8.json"
    _write_json(
        mujoco_report,
        {
            "schema_version": 1,
            "gate": "G8-simulation",
            "result": "pass",
            "mode": "mujoco",
            "evidence_level": "simulation",
            "config_sha256": project_config_sha256(loaded_mujoco_config),
            "policy": "compact_wam",
            "checkpoint_sha256": file_sha256(candidate),
            "checkpoint_id": candidate_id,
            "primary_cameras": list(PRIMARY_CAMERA_KEYS),
            "physics_steps_per_servo_tick": 4,
            "physics_steps": 40,
            "collision_gate": True,
            "rollout": {
                "final_state": "rollout_ready",
                "prompt_fingerprint": prompt_fingerprint,
                "policy_steps": 2,
                "servo_steps": 10,
                "sent_actions": 10,
                "shadow_steps": 0,
            },
        },
    )
    unsigned_sources = {
        "candidate_checkpoint": file_sha256(candidate),
        "training_report": file_sha256(training_report),
        "preflight_report": file_sha256(preflight_report),
        "mujoco_report": file_sha256(mujoco_report),
        "prompt_npz": file_sha256(prompt_npz),
        "prompt_manifest": file_sha256(prompt_manifest),
    }
    manual_signoff = directory / "manual-signoff.json"
    _write_json(
        manual_signoff,
        {
            "schema_version": MANUAL_SIGNOFF_SCHEMA,
            "artifact_kind": MANUAL_SIGNOFF_KIND,
            "evidence_level": "real",
            "result": "pass",
            "operator_id": "test-operator",
            "signed_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_id": config.lerobot.hardware_id,
            "calibration_id": config.lerobot.calibration_id,
            "config_sha256": project_config_sha256(config),
            "source_evidence_sha256": unsigned_sources,
            "checks": {key: True for key in REQUIRED_MANUAL_CHECKS},
        },
    )
    return _Evidence(
        config=config,
        config_path=config_path,
        candidate=candidate,
        training_report=training_report,
        preflight_report=preflight_report,
        mujoco_config=mujoco_config,
        mujoco_report=mujoco_report,
        prompt_npz=prompt_npz,
        prompt_manifest=prompt_manifest,
        manual_signoff=manual_signoff,
    )


def _issue(
    evidence: _Evidence,
    directory: Path,
    *,
    checkpoint_name: str = "deployment.pt",
    certification_name: str = "deployment.certification.json",
) -> tuple[Path, Path]:
    checkpoint = directory / checkpoint_name
    certification = directory / certification_name
    issue_deployment_artifacts(
        evidence.config,
        candidate_checkpoint_path=evidence.candidate,
        training_report_path=evidence.training_report,
        preflight_report_path=evidence.preflight_report,
        mujoco_config_path=evidence.mujoco_config,
        mujoco_report_path=evidence.mujoco_report,
        prompt_npz_path=evidence.prompt_npz,
        prompt_manifest_path=evidence.prompt_manifest,
        manual_signoff_path=evidence.manual_signoff,
        output_checkpoint_path=checkpoint,
        output_certification_path=certification,
        checkpoint_id="deployment-test-v1",
        max_policy_steps=2,
    )
    return checkpoint, certification


def test_issuer_promotes_exact_weights_and_binds_all_source_evidence(
    tmp_path: Path,
) -> None:
    evidence = _build_evidence(tmp_path)
    deployment, certification = _issue(evidence, tmp_path)

    candidate = load_compact_wam_bundle(evidence.candidate)
    deployed = load_compact_wam_bundle(deployment)
    for key, value in candidate.model.state_dict().items():
        assert torch.equal(value, deployed.model.state_dict()[key])
    assert deployed.metadata["artifact_kind"] == "compact_wam_deployment"
    assert deployed.metadata["evidence_level"] == "real"
    assert deployed.metadata["trained"] is True
    assert deployed.metadata["deployment_ready"] is True
    assert "offline_trained" not in deployed.metadata
    _require_deployment_checkpoint(deployed.metadata)

    certificate = json.loads(certification.read_text(encoding="utf-8"))
    assert certificate["schema_version"] == DEPLOYMENT_CERTIFICATION_SCHEMA
    assert certificate["authorization_scope"] == DEPLOYMENT_AUTHORIZATION_SCOPE
    assert certificate["source_evidence_sha256"] == {
        "candidate_checkpoint": file_sha256(evidence.candidate),
        "training_report": file_sha256(evidence.training_report),
        "preflight_report": file_sha256(evidence.preflight_report),
        "mujoco_report": file_sha256(evidence.mujoco_report),
        "prompt_npz": file_sha256(evidence.prompt_npz),
        "prompt_manifest": file_sha256(evidence.prompt_manifest),
        "manual_signoff": file_sha256(evidence.manual_signoff),
    }
    verified = verify_deployment_certification(
        certification,
        checkpoint_sha256=file_sha256(deployment),
        source_evidence_sha256=certificate["source_evidence_sha256"],
        checkpoint_metadata=deployed.metadata,
        config=evidence.config,
        policy_steps=2,
    )
    assert verified.max_policy_steps == 2


@pytest.mark.parametrize(
    ("artifact_name", "key", "replacement", "message"),
    [
        ("preflight_report", "automated_checks_passed", False, "automated_checks"),
        ("mujoco_report", "policy", "wrist_roll_smoke", "policy"),
        ("manual_signoff", "evidence_level", "simulation", "evidence_level"),
    ],
)
def test_issuer_rejects_partial_or_sim_only_evidence_without_outputs(
    tmp_path: Path,
    artifact_name: str,
    key: str,
    replacement: object,
    message: str,
) -> None:
    evidence = _build_evidence(tmp_path)
    artifact = getattr(evidence, artifact_name)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload[key] = replacement
    _write_json(artifact, payload)
    deployment = tmp_path / "rejected.pt"
    certification = tmp_path / "rejected.json"

    with pytest.raises(DeploymentIssuanceError, match=message):
        issue_deployment_artifacts(
            evidence.config,
            candidate_checkpoint_path=evidence.candidate,
            training_report_path=evidence.training_report,
            preflight_report_path=evidence.preflight_report,
            mujoco_config_path=evidence.mujoco_config,
            mujoco_report_path=evidence.mujoco_report,
            prompt_npz_path=evidence.prompt_npz,
            prompt_manifest_path=evidence.prompt_manifest,
            manual_signoff_path=evidence.manual_signoff,
            output_checkpoint_path=deployment,
            output_certification_path=certification,
            checkpoint_id="must-not-exist",
            max_policy_steps=1,
        )

    assert not deployment.exists()
    assert not certification.exists()


def test_issuer_rejects_missing_or_false_manual_signoff(tmp_path: Path) -> None:
    evidence = _build_evidence(tmp_path)
    signoff = json.loads(evidence.manual_signoff.read_text(encoding="utf-8"))
    signoff["checks"]["real_slow_jog_no_contact"] = False
    _write_json(evidence.manual_signoff, signoff)

    with pytest.raises(DeploymentIssuanceError, match="every hardware check"):
        _issue(evidence, tmp_path)
    assert not (tmp_path / "deployment.pt").exists()
    assert not (tmp_path / "deployment.certification.json").exists()

    evidence.manual_signoff.unlink()
    with pytest.raises(DeploymentIssuanceError, match="cannot read JSON"):
        _issue(
            evidence,
            tmp_path,
            checkpoint_name="missing-signoff.pt",
            certification_name="missing-signoff.json",
        )
    assert not (tmp_path / "missing-signoff.pt").exists()
    assert not (tmp_path / "missing-signoff.json").exists()


def test_issuer_rejects_evidence_tampering_after_signoff(tmp_path: Path) -> None:
    evidence = _build_evidence(tmp_path)
    with evidence.mujoco_report.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(DeploymentIssuanceError, match="not bound"):
        _issue(evidence, tmp_path)
    assert not (tmp_path / "deployment.pt").exists()
    assert not (tmp_path / "deployment.certification.json").exists()


@pytest.mark.parametrize(
    ("artifact_name", "key", "replacement", "message"),
    [
        (
            "preflight_report",
            "observation_latency_p95_s",
            0.2,
            "observation_latency_p95_s exceeds",
        ),
        ("mujoco_report", "physics_steps", 39, "physics-step count"),
    ],
)
def test_issuer_recomputes_raw_preflight_and_mujoco_metrics(
    tmp_path: Path,
    artifact_name: str,
    key: str,
    replacement: object,
    message: str,
) -> None:
    evidence = _build_evidence(tmp_path)
    artifact = getattr(evidence, artifact_name)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload[key] = replacement
    _write_json(artifact, payload)

    with pytest.raises(DeploymentIssuanceError, match=message):
        _issue(evidence, tmp_path)
    assert not (tmp_path / "deployment.pt").exists()
    assert not (tmp_path / "deployment.certification.json").exists()


def test_issuer_recomputes_g9_capture_metrics(tmp_path: Path) -> None:
    evidence = _build_evidence(
        tmp_path,
        prompt_metadata_overrides={"camera_skew_p95_s": 0.1},
    )

    with pytest.raises(DeploymentIssuanceError, match="camera_skew_p95_s exceeds"):
        _issue(evidence, tmp_path)
    assert not (tmp_path / "deployment.pt").exists()
    assert not (tmp_path / "deployment.certification.json").exists()


def test_g8_report_must_match_fresh_mujoco_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build_evidence(tmp_path)
    report = json.loads(evidence.mujoco_report.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        deployment_issuer_module,
        "run_mujoco_checkpoint_session",
        lambda *args, **kwargs: dict(report),
    )

    _REAL_RERUN_MUJOCO_EVIDENCE(
        config=ProjectConfig.load(evidence.mujoco_config),
        report=report,
        candidate_path=evidence.candidate,
        prompt_npz_path=evidence.prompt_npz,
        prompt_manifest_path=evidence.prompt_manifest,
    )

    contradictory = dict(report)
    contradictory["physics_steps"] = int(report["physics_steps"]) + 4
    monkeypatch.setattr(
        deployment_issuer_module,
        "run_mujoco_checkpoint_session",
        lambda *args, **kwargs: contradictory,
    )
    with pytest.raises(DeploymentIssuanceError, match="exact MuJoCo reproduction"):
        _REAL_RERUN_MUJOCO_EVIDENCE(
            config=ProjectConfig.load(evidence.mujoco_config),
            report=report,
            candidate_path=evidence.candidate,
            prompt_npz_path=evidence.prompt_npz,
            prompt_manifest_path=evidence.prompt_manifest,
        )


@requires_mujoco
def test_g8_report_is_actually_reproducible_in_mujoco(tmp_path: Path) -> None:
    evidence = _build_evidence(tmp_path)
    base_mujoco_config = ProjectConfig.load(evidence.mujoco_config)
    mujoco_config = replace(
        base_mujoco_config,
        mujoco=replace(
            base_mujoco_config.mujoco,
            camera_width=8,
            camera_height=8,
        ),
    )
    report = deployment_issuer_module.run_mujoco_checkpoint_session(
        mujoco_config,
        checkpoint_path=evidence.candidate,
        prompt_path=evidence.prompt_npz,
        manifest_path=evidence.prompt_manifest,
        policy_steps=1,
        device="cpu",
    )

    _REAL_RERUN_MUJOCO_EVIDENCE(
        config=mujoco_config,
        report=report,
        candidate_path=evidence.candidate,
        prompt_npz_path=evidence.prompt_npz,
        prompt_manifest_path=evidence.prompt_manifest,
    )


def test_issuer_is_no_overwrite_and_rolls_back_first_half_on_pair_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build_evidence(tmp_path)
    deployment, certification = _issue(evidence, tmp_path)
    before = (file_sha256(deployment), file_sha256(certification))

    with pytest.raises(DeploymentArtifactExistsError, match="immutable"):
        _issue(evidence, tmp_path)
    assert (file_sha256(deployment), file_sha256(certification)) == before

    from so101_wam import deployment_issuer

    original_link = deployment_issuer._link_no_overwrite

    def fail_certification_link(source: Path, target: Path) -> None:
        if target.name == "pair-failure.json":
            raise DeploymentArtifactExistsError("injected second-link failure")
        original_link(source, target)

    monkeypatch.setattr(
        deployment_issuer,
        "_link_no_overwrite",
        fail_certification_link,
    )
    with pytest.raises(DeploymentArtifactExistsError, match="injected"):
        _issue(
            evidence,
            tmp_path,
            checkpoint_name="pair-failure.pt",
            certification_name="pair-failure.json",
        )
    assert not (tmp_path / "pair-failure.pt").exists()
    assert not (tmp_path / "pair-failure.json").exists()


def test_certify_cli_issues_only_from_artifact_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = _build_evidence(tmp_path)
    output = tmp_path / "cli-deployment.pt"
    certification = tmp_path / "cli-deployment.json"

    result = certify_main(
        [
            "--config",
            str(evidence.config_path),
            "--candidate",
            str(evidence.candidate),
            "--training-report",
            str(evidence.training_report),
            "--preflight-report",
            str(evidence.preflight_report),
            "--mujoco-report",
            str(evidence.mujoco_report),
            "--mujoco-config",
            str(evidence.mujoco_config),
            "--prompt",
            str(evidence.prompt_npz),
            "--manifest",
            str(evidence.prompt_manifest),
            "--manual-signoff",
            str(evidence.manual_signoff),
            "--output",
            str(output),
            "--certification",
            str(certification),
            "--checkpoint-id",
            "cli-deployment-v1",
            "--max-policy-steps",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["deployment_ready"] is True
    assert payload["checkpoint_id"] == "cli-deployment-v1"
    assert output.exists() and certification.exists()


def test_missing_source_artifact_fails_before_any_output(tmp_path: Path) -> None:
    evidence = _build_evidence(tmp_path)
    evidence.candidate.unlink()

    with pytest.raises(DeploymentCertificationError, match="cannot read"):
        _issue(evidence, tmp_path)
    assert not (tmp_path / "deployment.pt").exists()
    assert not (tmp_path / "deployment.certification.json").exists()
