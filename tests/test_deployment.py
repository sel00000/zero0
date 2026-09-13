from __future__ import annotations

import json
from pathlib import Path

import pytest

from so101_wam.config import LeRobotConfig, ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.constants import ACTION_DIM
from so101_wam.deployment import (
    DEPLOYMENT_AUTHORIZATION_SCOPE,
    DEPLOYMENT_CERTIFICATION_KIND,
    DEPLOYMENT_CERTIFICATION_SCHEMA,
    DEPLOYMENT_SOURCE_EVIDENCE_KEYS,
    DeploymentCertificationError,
    deployment_evidence_sha256,
    project_config_sha256,
    verify_deployment_certification,
)


def _config(*, hardware_id: str = "bench-a") -> ProjectConfig:
    return ProjectConfig(
        runtime=RuntimeConfig(backend="lerobot", actuation_enabled=True),
        safety=SafetyConfig(
            joint_lower=(-10.0,) * ACTION_DIM,
            joint_upper=(10.0,) * ACTION_DIM,
            max_delta_per_servo_tick=(1.0,) * ACTION_DIM,
            calibrated=True,
        ),
        lerobot=LeRobotConfig(
            left_port="left",
            right_port="right",
            left_wrist_camera=0,
            right_wrist_camera=1,
            calibration_dir="calibration",
            hardware_id=hardware_id,
            calibration_id="cal-a",
            home_joint_position=(0.0,) * ACTION_DIM,
            home_joint_tolerance=(0.1,) * ACTION_DIM,
        ),
    )


def _core(config: ProjectConfig) -> dict[str, object]:
    return {
        "schema_version": DEPLOYMENT_CERTIFICATION_SCHEMA,
        "artifact_kind": DEPLOYMENT_CERTIFICATION_KIND,
        "evidence_level": "real",
        "result": "pass",
        "authorization_scope": DEPLOYMENT_AUTHORIZATION_SCOPE,
        "checkpoint_id": "deployment-1",
        "training_evidence_sha256": "a" * 64,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "config_sha256": project_config_sha256(config),
        "max_policy_steps": 2,
        "gate_results": {gate: "pass" for gate in ("G6", "G7", "G8", "G9")},
        "gate_evidence_sha256": {
            gate: str(index) * 64
            for index, gate in enumerate(("G6", "G7", "G8", "G9"), start=1)
        },
        "source_evidence_sha256": {
            source: "a" * 64 for source in DEPLOYMENT_SOURCE_EVIDENCE_KEYS
        },
    }


def _write_certification(
    path: Path,
    core: dict[str, object],
    *,
    checkpoint_sha256: str,
) -> None:
    path.write_text(
        json.dumps({**core, "checkpoint_sha256": checkpoint_sha256}, sort_keys=True),
        encoding="utf-8",
    )


def _metadata(core: dict[str, object]) -> dict[str, str | int | float | bool | None]:
    return {
        "checkpoint_id": str(core["checkpoint_id"]),
        "training_evidence_sha256": str(core["training_evidence_sha256"]),
        "deployment_evidence_sha256": deployment_evidence_sha256(core),
    }


def test_external_certification_binds_checkpoint_config_hardware_and_scope(
    tmp_path: Path,
) -> None:
    config = _config()
    core = _core(config)
    certificate = tmp_path / "deployment.json"
    checkpoint_sha256 = "b" * 64
    _write_certification(certificate, core, checkpoint_sha256=checkpoint_sha256)

    verified = verify_deployment_certification(
        certificate,
        checkpoint_sha256=checkpoint_sha256,
        source_evidence_sha256=core["source_evidence_sha256"],  # type: ignore[arg-type]
        checkpoint_metadata=_metadata(core),
        config=config,
        policy_steps=2,
    )

    assert verified.checkpoint_sha256 == checkpoint_sha256
    assert verified.evidence_sha256 == deployment_evidence_sha256(core)
    assert verified.hardware_id == "bench-a"


def test_external_certification_rejects_checkpoint_or_config_substitution(
    tmp_path: Path,
) -> None:
    config = _config()
    core = _core(config)
    certificate = tmp_path / "deployment.json"
    _write_certification(certificate, core, checkpoint_sha256="b" * 64)

    with pytest.raises(DeploymentCertificationError, match="checkpoint_sha256"):
        verify_deployment_certification(
            certificate,
            checkpoint_sha256="c" * 64,
            source_evidence_sha256=core["source_evidence_sha256"],  # type: ignore[arg-type]
            checkpoint_metadata=_metadata(core),
            config=config,
            policy_steps=1,
        )
    with pytest.raises(DeploymentCertificationError, match="hardware_id"):
        verify_deployment_certification(
            certificate,
            checkpoint_sha256="b" * 64,
            source_evidence_sha256=core["source_evidence_sha256"],  # type: ignore[arg-type]
            checkpoint_metadata=_metadata(core),
            config=_config(hardware_id="bench-b"),
            policy_steps=1,
        )


def test_external_certification_rejects_tampered_gate_results_and_scope(
    tmp_path: Path,
) -> None:
    config = _config()
    core = _core(config)
    certificate = tmp_path / "deployment.json"
    tampered = dict(core)
    tampered["gate_results"] = {
        "G6": "pass",
        "G7": "pass",
        "G8": "pass",
        "G9": "fail",
    }
    _write_certification(certificate, tampered, checkpoint_sha256="b" * 64)

    with pytest.raises(DeploymentCertificationError, match="every G6-G9"):
        verify_deployment_certification(
            certificate,
            checkpoint_sha256="b" * 64,
            source_evidence_sha256=core["source_evidence_sha256"],  # type: ignore[arg-type]
            checkpoint_metadata=_metadata(core),
            config=config,
            policy_steps=1,
        )

    _write_certification(certificate, core, checkpoint_sha256="b" * 64)
    with pytest.raises(DeploymentCertificationError, match="exceed"):
        verify_deployment_certification(
            certificate,
            checkpoint_sha256="b" * 64,
            source_evidence_sha256=core["source_evidence_sha256"],  # type: ignore[arg-type]
            checkpoint_metadata=_metadata(core),
            config=config,
            policy_steps=3,
        )


def test_external_certification_rejects_float_schema_version(tmp_path: Path) -> None:
    config = _config()
    core = _core(config)
    core["schema_version"] = float(DEPLOYMENT_CERTIFICATION_SCHEMA)
    certificate = tmp_path / "deployment.json"
    _write_certification(certificate, core, checkpoint_sha256="b" * 64)

    with pytest.raises(DeploymentCertificationError, match="must be an integer"):
        verify_deployment_certification(
            certificate,
            checkpoint_sha256="b" * 64,
            source_evidence_sha256=core["source_evidence_sha256"],  # type: ignore[arg-type]
            checkpoint_metadata=_metadata(core),
            config=config,
            policy_steps=1,
        )
