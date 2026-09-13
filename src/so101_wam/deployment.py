"""External deployment-certification verification for real SO-101 output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import blake2b, sha256
import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig


DEPLOYMENT_CERTIFICATION_SCHEMA = 2
DEPLOYMENT_CERTIFICATION_KIND = "so101_wam.deployment_certification"
DEPLOYMENT_AUTHORIZATION_SCOPE = "G10_dry_rollout"
REQUIRED_PREACTUATION_GATES = ("G6", "G7", "G8", "G9")
DEPLOYMENT_SOURCE_EVIDENCE_KEYS = (
    "candidate_checkpoint",
    "training_report",
    "preflight_report",
    "mujoco_report",
    "prompt_npz",
    "prompt_manifest",
    "manual_signoff",
)

_CORE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "evidence_level",
        "result",
        "authorization_scope",
        "checkpoint_id",
        "training_evidence_sha256",
        "hardware_id",
        "calibration_id",
        "config_sha256",
        "max_policy_steps",
        "gate_results",
        "gate_evidence_sha256",
        "source_evidence_sha256",
    }
)
_CERTIFICATION_KEYS = _CORE_KEYS | {"checkpoint_sha256"}


class DeploymentCertificationError(ValueError):
    """Raised when external real-output evidence is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class DeploymentCertification:
    checkpoint_sha256: str
    evidence_sha256: str
    checkpoint_id: str
    hardware_id: str
    calibration_id: str
    config_sha256: str
    max_policy_steps: int


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of an immutable artifact without loading it into memory."""

    digest = sha256()
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DeploymentCertificationError(
            f"cannot read deployment artifact {Path(path)}: {error}"
        ) from error
    return digest.hexdigest()


def project_config_sha256(config: ProjectConfig) -> str:
    """Bind certification to the complete validated runtime/safety/device config."""

    return _canonical_json_sha256(_project_config_payload(config))


def canonical_json_sha256(value: object) -> str:
    """Return the canonical JSON digest used by deployment evidence contracts."""

    return _canonical_json_sha256(value)


def project_config_fingerprint(config: ProjectConfig) -> str:
    """Return the short fingerprint stored by observation-only preflight reports."""

    encoded = json.dumps(
        _project_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return blake2b(encoded, digest_size=16).hexdigest()


def deployment_evidence_sha256(core: Mapping[str, Any]) -> str:
    """Hash the certification core, excluding only the cyclic checkpoint hash."""

    if set(core) != _CORE_KEYS:
        raise DeploymentCertificationError(
            "deployment certification core keys do not match schema"
        )
    return _canonical_json_sha256(core)


def verify_deployment_certification(
    path: str | Path,
    *,
    checkpoint_sha256: str,
    source_evidence_sha256: Mapping[str, str],
    checkpoint_metadata: Mapping[str, str | int | float | bool | None],
    config: ProjectConfig,
    policy_steps: int,
) -> DeploymentCertification:
    """Verify external real-output evidence before any robot object is constructed."""

    payload = _load_strict_json(path)
    if set(payload) != _CERTIFICATION_KEYS:
        raise DeploymentCertificationError(
            "deployment certification keys do not match schema"
        )
    core = {key: payload[key] for key in _CORE_KEYS}
    _validate_core(core)

    evidence_sha256 = deployment_evidence_sha256(core)
    if checkpoint_metadata.get("deployment_evidence_sha256") != evidence_sha256:
        raise DeploymentCertificationError(
            "deployment certification digest does not match checkpoint metadata"
        )
    if payload["checkpoint_sha256"] != checkpoint_sha256:
        raise DeploymentCertificationError(
            "deployment certification checkpoint_sha256 does not match checkpoint bytes"
        )
    certified_sources = core["source_evidence_sha256"]
    assert isinstance(certified_sources, Mapping)
    if dict(certified_sources) != dict(source_evidence_sha256):
        raise DeploymentCertificationError(
            "deployment certification source evidence does not match revalidated artifacts"
        )
    if core["checkpoint_id"] != checkpoint_metadata.get("checkpoint_id"):
        raise DeploymentCertificationError(
            "deployment certification checkpoint_id does not match checkpoint metadata"
        )
    if core["training_evidence_sha256"] != checkpoint_metadata.get(
        "training_evidence_sha256"
    ):
        raise DeploymentCertificationError(
            "deployment certification training evidence does not match checkpoint metadata"
        )
    if core["hardware_id"] != config.lerobot.hardware_id:
        raise DeploymentCertificationError(
            "deployment certification hardware_id does not match runtime config"
        )
    if core["calibration_id"] != config.lerobot.calibration_id:
        raise DeploymentCertificationError(
            "deployment certification calibration_id does not match runtime config"
        )
    config_sha256 = project_config_sha256(config)
    if core["config_sha256"] != config_sha256:
        raise DeploymentCertificationError(
            "deployment certification config_sha256 does not match runtime config"
        )
    max_policy_steps = int(core["max_policy_steps"])
    if policy_steps > max_policy_steps:
        raise DeploymentCertificationError(
            "requested policy steps exceed deployment certification scope"
        )

    return DeploymentCertification(
        checkpoint_sha256=checkpoint_sha256,
        evidence_sha256=evidence_sha256,
        checkpoint_id=str(core["checkpoint_id"]),
        hardware_id=str(core["hardware_id"]),
        calibration_id=str(core["calibration_id"]),
        config_sha256=config_sha256,
        max_policy_steps=max_policy_steps,
    )


def _load_strict_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as error:
        raise DeploymentCertificationError(
            f"cannot read deployment certification {target}: {error}"
        ) from error
    if len(raw) > 1024 * 1024:
        raise DeploymentCertificationError(
            "deployment certification must not exceed 1 MiB"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DeploymentCertificationError(
                    f"deployment certification contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except DeploymentCertificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentCertificationError(
            f"deployment certification is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise DeploymentCertificationError(
            "deployment certification root must be an object"
        )
    return payload


def _validate_core(core: Mapping[str, Any]) -> None:
    expected_scalars = {
        "schema_version": DEPLOYMENT_CERTIFICATION_SCHEMA,
        "artifact_kind": DEPLOYMENT_CERTIFICATION_KIND,
        "evidence_level": "real",
        "result": "pass",
        "authorization_scope": DEPLOYMENT_AUTHORIZATION_SCOPE,
    }
    for key, expected in expected_scalars.items():
        value = core.get(key)
        if value != expected or isinstance(value, bool):
            raise DeploymentCertificationError(
                f"deployment certification requires {key}={expected!r}"
            )
        if key == "schema_version" and not isinstance(value, int):
            raise DeploymentCertificationError(
                "deployment certification schema_version must be an integer"
            )
    for key in ("checkpoint_id", "hardware_id", "calibration_id"):
        value = core.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DeploymentCertificationError(
                f"deployment certification {key} must be a non-empty string"
            )
    for key in (
        "training_evidence_sha256",
        "config_sha256",
    ):
        _require_sha256(core.get(key), key)
    max_policy_steps = core.get("max_policy_steps")
    if (
        not isinstance(max_policy_steps, int)
        or isinstance(max_policy_steps, bool)
        or max_policy_steps < 1
    ):
        raise DeploymentCertificationError(
            "deployment certification max_policy_steps must be a positive integer"
        )

    expected_gates = set(REQUIRED_PREACTUATION_GATES)
    gate_results = core.get("gate_results")
    if not isinstance(gate_results, Mapping) or set(gate_results) != expected_gates:
        raise DeploymentCertificationError(
            "deployment certification gate_results must cover exactly G6-G9"
        )
    if any(value != "pass" for value in gate_results.values()):
        raise DeploymentCertificationError(
            "deployment certification requires every G6-G9 result to pass"
        )
    gate_evidence = core.get("gate_evidence_sha256")
    if not isinstance(gate_evidence, Mapping) or set(gate_evidence) != expected_gates:
        raise DeploymentCertificationError(
            "deployment certification gate evidence must cover exactly G6-G9"
        )
    for gate, digest in gate_evidence.items():
        _require_sha256(digest, f"gate_evidence_sha256.{gate}")
    source_evidence = core.get("source_evidence_sha256")
    expected_sources = set(DEPLOYMENT_SOURCE_EVIDENCE_KEYS)
    if (
        not isinstance(source_evidence, Mapping)
        or set(source_evidence) != expected_sources
    ):
        raise DeploymentCertificationError(
            "deployment certification source evidence keys do not match schema"
        )
    for source, digest in source_evidence.items():
        _require_sha256(digest, f"source_evidence_sha256.{source}")


def _require_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DeploymentCertificationError(
            f"deployment certification {name} must be a 64-character lowercase SHA-256"
        )


def _project_config_payload(config: ProjectConfig) -> dict[str, Any]:
    payload = asdict(config)
    safety = payload["safety"]
    assert isinstance(safety, dict)
    if (
        safety.get("observation_joint_lower") is None
        and safety.get("observation_joint_upper") is None
    ):
        safety.pop("observation_joint_lower", None)
        safety.pop("observation_joint_upper", None)
    return payload


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DeploymentCertificationError(
            f"deployment evidence cannot be canonicalized: {error}"
        ) from error
    return sha256(encoded).hexdigest()
