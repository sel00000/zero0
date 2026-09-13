"""Static integrity verification for a published robot-free result bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn, Sequence, cast

from .checkpoint import CheckpointError, load_compact_wam_bundle
from .config import ConfigError, ProjectConfig
from .dataset import (
    DatasetError,
    EpisodeData,
    load_episode,
    physical_prompt_from_episode,
)
from .deployment import (
    DeploymentCertificationError,
    canonical_json_sha256,
    file_sha256,
    project_config_sha256,
)
from .robot_free_cli import (
    EPISODE_DURATION_S,
    EPISODE_FPS,
    EPISODE_FRAME_COUNT,
    ROBOT_FREE_LIMITATIONS,
    ROBOT_FREE_SCHEMA_VERSION,
    SYNTHETIC_IMAGE_HEIGHT,
    SYNTHETIC_IMAGE_WIDTH,
    RobotFreePipelineError,
    _check_real_output_guard,
    _require_actuated_g8,
    _require_robot_free_config,
    _robot_free_training_config,
)
from .training import TRAINING_REPORT_SCHEMA
from .training_data import EpisodeRecord, episode_split_digest


ROBOT_FREE_VERIFICATION_SCHEMA_VERSION = "so101_wam.robot_free_verification.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FIXED_ARTIFACT_NAMES = {
    "checkpoint",
    "config",
    "g8_report",
    "training_report",
    *(f"train_episode_{index:02d}" for index in range(2)),
    *(f"train_episode_manifest_{index:02d}" for index in range(2)),
    *(f"validation_episode_{index:02d}" for index in range(2)),
    *(f"validation_episode_manifest_{index:02d}" for index in range(2)),
}
_RESULT_KEYS = {
    "artifacts",
    "candidate_checkpoint_sha256",
    "deployment_ready",
    "evidence_inputs",
    "evidence_level",
    "g8",
    "g8_report_sha256",
    "limitations",
    "mode",
    "real_output_authorized",
    "real_output_rejection_reason",
    "result",
    "schema_version",
    "synthetic_data",
    "trained",
    "training",
}


class RobotFreeVerificationError(ValueError):
    """Raised when a robot-free bundle is incomplete, inconsistent, or altered."""


def verify_robot_free_result(
    result_path: str | Path,
    *,
    device: str = "cpu",
) -> Mapping[str, Any]:
    """Verify bundle-local hashes and offline/simulation contracts without rerunning G8.

    This is an integrity and consistency check, not an authenticity signature and
    not new real-hardware evidence.
    """

    result_target = Path(result_path)
    if result_target.is_symlink() or not result_target.is_file():
        raise RobotFreeVerificationError(
            "robot-free result must be an existing non-symlink file"
        )
    result = _read_json_object(result_target, name="robot-free result")
    _require_result_contract(result)
    artifact_values = _require_mapping(result.get("artifacts"), name="artifacts")
    artifact_paths = _verify_artifacts(result_target.parent.resolve(), artifact_values)

    try:
        config = ProjectConfig.load(artifact_paths["config"])
    except (OSError, ValueError) as error:
        raise RobotFreeVerificationError("effective config is invalid") from error
    _require_robot_free_config(config)

    checkpoint_path = artifact_paths["checkpoint"]
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if result.get("candidate_checkpoint_sha256") != checkpoint_sha256:
        raise RobotFreeVerificationError(
            "candidate_checkpoint_sha256 does not match the checkpoint artifact"
        )
    bundle = load_compact_wam_bundle(checkpoint_path, device=device)
    checkpoint_id = _require_nonempty_string(
        bundle.metadata.get("checkpoint_id"),
        name="checkpoint metadata.checkpoint_id",
    )
    _require_offline_candidate_metadata(bundle.metadata)
    training_evidence_sha256 = _require_nonempty_string(
        bundle.metadata.get("training_evidence_sha256"),
        name="checkpoint metadata.training_evidence_sha256",
    )
    if _SHA256_PATTERN.fullmatch(training_evidence_sha256) is None:
        raise RobotFreeVerificationError(
            "checkpoint metadata.training_evidence_sha256 must be lowercase SHA-256"
        )

    training_report = _read_json_object(
        artifact_paths["training_report"],
        name="training report",
    )
    _require_training_report(
        training_report,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        training_evidence_sha256=training_evidence_sha256,
        checkpoint_filename=artifact_paths["checkpoint"].name,
        report_filename=artifact_paths["training_report"].name,
    )
    _require_training_summary(
        result,
        training_report,
        checkpoint_path=_artifact_relative_path(artifact_values, "checkpoint"),
        report_path=_artifact_relative_path(artifact_values, "training_report"),
        checkpoint_id=checkpoint_id,
        training_evidence_sha256=training_evidence_sha256,
    )

    g8_report_path = artifact_paths["g8_report"]
    g8_report = _read_json_object(g8_report_path, name="G8 report")
    _require_actuated_g8(g8_report)
    g8_report_sha256 = file_sha256(g8_report_path)
    _require_g8_contract(
        result,
        g8_report,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        g8_report_sha256=g8_report_sha256,
        config_sha256=project_config_sha256(config),
    )

    episodes = _verify_episodes(artifact_paths)
    _require_training_split_digests(training_report, artifact_paths, episodes)
    rollout = _require_mapping(g8_report.get("rollout"), name="G8 rollout")
    expected_prompt = physical_prompt_from_episode(
        episodes["validation"][0],
        policy_hz=config.runtime.policy_hz,
    )
    if rollout.get("prompt_fingerprint") != expected_prompt.fingerprint:
        raise RobotFreeVerificationError(
            "G8 prompt_fingerprint does not match validation episode 00"
        )

    real_output = _check_real_output_guard(checkpoint_path, device=device)
    if real_output.get("real_output_authorized") is not False:
        raise RobotFreeVerificationError(
            "offline candidate unexpectedly passed the real-output guard"
        )
    if result.get("real_output_rejection_reason") != real_output.get(
        "rejection_reason"
    ):
        raise RobotFreeVerificationError(
            "real-output rejection reason does not match a fresh guard check"
        )

    return {
        "schema_version": ROBOT_FREE_VERIFICATION_SCHEMA_VERSION,
        "result": "pass",
        "mode": "robot_free_integrity_verification",
        "evidence_level": "simulation",
        "verified_result_path": result_target.resolve().as_posix(),
        "verified_artifact_count": len(artifact_paths),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "g8_report_sha256": g8_report_sha256,
        "g8_result": g8_report["result"],
        "real_output_authorized": False,
        "limitations": [
            "hash consistency is not an authenticity signature",
            "verification does not rerun training or MuJoCo G8",
            "verification adds no real-hardware evidence",
        ],
    }


def _require_result_contract(result: Mapping[str, Any]) -> None:
    if set(result) != _RESULT_KEYS:
        missing = sorted(_RESULT_KEYS - set(result))
        unexpected = sorted(set(result) - _RESULT_KEYS)
        raise RobotFreeVerificationError(
            f"robot-free result fields mismatch: missing={missing}, unexpected={unexpected}"
        )
    expected = {
        "schema_version": ROBOT_FREE_SCHEMA_VERSION,
        "result": "pass",
        "mode": "robot_free",
        "evidence_level": "simulation",
        "evidence_inputs": ["offline", "simulation"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RobotFreeVerificationError(
                f"robot-free result requires {key}={value!r}"
            )
    for key in ("trained", "deployment_ready", "real_output_authorized"):
        if result.get(key) is not False:
            raise RobotFreeVerificationError(f"robot-free result requires {key}=false")
    _require_nonempty_string(
        result.get("real_output_rejection_reason"),
        name="real_output_rejection_reason",
    )
    synthetic = _require_mapping(result.get("synthetic_data"), name="synthetic_data")
    expected_synthetic = {
        "fps": EPISODE_FPS,
        "duration_s": EPISODE_DURATION_S,
        "frame_count": EPISODE_FRAME_COUNT,
        "resolution": {
            "height": SYNTHETIC_IMAGE_HEIGHT,
            "width": SYNTHETIC_IMAGE_WIDTH,
        },
        "action_source": "synthetic_deterministic_no_robot_no_goal_write",
        "train_episode_count": 2,
        "validation_episode_count": 2,
        "train_task_count": 1,
        "validation_task_count": 1,
        "task_split": "task_disjoint",
    }
    if dict(synthetic) != expected_synthetic:
        raise RobotFreeVerificationError(
            "synthetic_data does not match the v1 contract"
        )
    if result.get("limitations") != list(ROBOT_FREE_LIMITATIONS):
        raise RobotFreeVerificationError(
            "robot-free result limitations do not match the v1 contract"
        )


def _verify_artifacts(
    output_root: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, Path]:
    if set(artifacts) != _FIXED_ARTIFACT_NAMES:
        missing = sorted(_FIXED_ARTIFACT_NAMES - set(artifacts))
        unexpected = sorted(set(artifacts) - _FIXED_ARTIFACT_NAMES)
        raise RobotFreeVerificationError(
            f"artifact names mismatch: missing={missing}, unexpected={unexpected}"
        )
    verified: dict[str, Path] = {}
    resolved_targets: set[Path] = set()
    for name in sorted(artifacts):
        descriptor = _require_mapping(artifacts[name], name=f"artifacts.{name}")
        if set(descriptor) != {"path", "sha256"}:
            raise RobotFreeVerificationError(
                f"artifacts.{name} must contain exactly path and sha256"
            )
        raw_path = _require_nonempty_string(
            descriptor.get("path"),
            name=f"artifacts.{name}.path",
        )
        relative_path = Path(raw_path)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != raw_path
            or any(part in {".", ".."} for part in relative_path.parts)
        ):
            raise RobotFreeVerificationError(
                f"artifacts.{name}.path must be a normalized bundle-relative path"
            )
        candidate = output_root / relative_path
        if candidate.is_symlink():
            raise RobotFreeVerificationError(
                f"artifacts.{name}.path must not be a symlink"
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise RobotFreeVerificationError(
                f"artifacts.{name} is missing: {relative_path.as_posix()}"
            ) from error
        if not resolved.is_relative_to(output_root) or not resolved.is_file():
            raise RobotFreeVerificationError(
                f"artifacts.{name}.path escapes the result bundle or is not a file"
            )
        if resolved in resolved_targets:
            raise RobotFreeVerificationError(
                f"artifacts.{name}.path aliases another artifact"
            )
        digest = _require_nonempty_string(
            descriptor.get("sha256"),
            name=f"artifacts.{name}.sha256",
        )
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise RobotFreeVerificationError(
                f"artifacts.{name}.sha256 must be lowercase SHA-256"
            )
        if file_sha256(resolved) != digest:
            raise RobotFreeVerificationError(f"artifacts.{name} SHA-256 mismatch")
        verified[name] = resolved
        resolved_targets.add(resolved)
    return verified


def _artifact_relative_path(artifacts: Mapping[str, Any], name: str) -> str:
    descriptor = _require_mapping(artifacts.get(name), name=f"artifacts.{name}")
    return _require_nonempty_string(
        descriptor.get("path"),
        name=f"artifacts.{name}.path",
    )


def _require_offline_candidate_metadata(metadata: Mapping[str, Any]) -> None:
    expected = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value or (
            isinstance(value, bool) and metadata.get(key) is not value
        ):
            raise RobotFreeVerificationError(
                f"checkpoint metadata requires {key}={value!r}"
            )


def _require_training_report(
    report: Mapping[str, Any],
    *,
    checkpoint_id: str,
    checkpoint_sha256: str,
    training_evidence_sha256: str,
    checkpoint_filename: str,
    report_filename: str,
) -> None:
    expected = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "result": "pass",
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "checkpoint_id": checkpoint_id,
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "training_evidence_sha256": training_evidence_sha256,
    }
    for key, value in expected.items():
        if report.get(key) != value or (
            isinstance(value, bool) and report.get(key) is not value
        ):
            raise RobotFreeVerificationError(
                f"training report requires {key}={value!r}"
            )
    artifacts = _require_mapping(report.get("artifacts"), name="training artifacts")
    expected_artifacts = {
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_sha256": checkpoint_sha256,
        "report_filename": report_filename,
    }
    if dict(artifacts) != expected_artifacts:
        raise RobotFreeVerificationError(
            "training report artifacts do not match the result bundle"
        )
    report_core = {
        key: value
        for key, value in report.items()
        if key not in {"training_evidence_sha256", "artifacts"}
    }
    if canonical_json_sha256(report_core) != training_evidence_sha256:
        raise RobotFreeVerificationError("training report core digest is invalid")
    data = _require_mapping(report.get("data"), name="training data summary")
    expected_counts = {
        "train_episode_count": 2,
        "validation_episode_count": 2,
        "train_task_count": 1,
        "validation_task_count": 1,
    }
    for key, value in expected_counts.items():
        if data.get(key) != value:
            raise RobotFreeVerificationError(
                f"training report requires data.{key}={value}"
            )


def _require_training_summary(
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    checkpoint_path: str,
    report_path: str,
    checkpoint_id: str,
    training_evidence_sha256: str,
) -> None:
    optimization = _require_mapping(
        report.get("optimization"),
        name="training optimization",
    )
    seed = optimization.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RobotFreeVerificationError(
            "training optimization seed must be a non-negative integer"
        )
    expected_config = asdict(_robot_free_training_config(seed))
    for key, value in expected_config.items():
        if optimization.get(key) != value:
            raise RobotFreeVerificationError(
                f"training optimization requires {key}={value!r}"
            )

    data = _require_mapping(report.get("data"), name="training data summary")
    validation = _require_mapping(
        report.get("validation"),
        name="training validation summary",
    )
    summary = _require_mapping(result.get("training"), name="training summary")
    expected_summary = {
        "checkpoint_path": checkpoint_path,
        "report_path": report_path,
        "checkpoint_id": checkpoint_id,
        "training_evidence_sha256": training_evidence_sha256,
        "train_windows": data.get("train_window_count"),
        "validation_windows": data.get("validation_window_count"),
        "validation_action_mse_normalized": validation.get("action_mse_normalized"),
        "validation_action_mae_native": validation.get("action_mae_native"),
        "validation_future_latent_mse": validation.get("future_latent_mse"),
        "trained": False,
        "deployment_ready": False,
        "config": expected_config,
    }
    if dict(summary) != expected_summary:
        raise RobotFreeVerificationError(
            "training summary does not match the bundle-local training report"
        )
    if (
        summary.get("trained") is not False
        or summary.get("deployment_ready") is not False
    ):
        raise RobotFreeVerificationError(
            "training summary must keep trained and deployment_ready false"
        )


def _require_training_split_digests(
    report: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    episodes: Mapping[str, list[EpisodeData]],
) -> None:
    data = _require_mapping(report.get("data"), name="training data summary")
    for split in ("train", "validation"):
        records = tuple(
            EpisodeRecord(
                path=artifact_paths[f"{split}_episode_{index:02d}"],
                data=episodes[split][index],
            )
            for index in range(2)
        )
        expected_digest = episode_split_digest(records)
        if data.get(f"{split}_split_sha256") != expected_digest:
            raise RobotFreeVerificationError(
                f"training report {split}_split_sha256 does not match the episodes"
            )


def _require_g8_contract(
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    checkpoint_id: str,
    checkpoint_sha256: str,
    g8_report_sha256: str,
    config_sha256: str,
) -> None:
    expected_report = {
        "gate": "G8-simulation",
        "result": "pass",
        "evidence_level": "simulation",
        "policy": "compact_wam",
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
    }
    for key, value in expected_report.items():
        if report.get(key) != value:
            raise RobotFreeVerificationError(f"G8 report requires {key}={value!r}")
    if result.get("g8_report_sha256") != g8_report_sha256:
        raise RobotFreeVerificationError(
            "g8_report_sha256 does not match the G8 artifact"
        )
    summary = _require_mapping(result.get("g8"), name="g8 summary")
    expected_summary = {
        "result": "pass",
        "gate": "G8-simulation",
        "policy_steps": 1,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "report_sha256": g8_report_sha256,
    }
    if dict(summary) != expected_summary:
        raise RobotFreeVerificationError("g8 summary does not match the G8 artifact")


def _verify_episodes(
    artifact_paths: Mapping[str, Path],
) -> dict[str, list[EpisodeData]]:
    result: dict[str, list[EpisodeData]] = {"train": [], "validation": []}
    for split in result:
        for index in range(2):
            episode = load_episode(
                artifact_paths[f"{split}_episode_{index:02d}"],
                artifact_paths[f"{split}_episode_manifest_{index:02d}"],
            )
            if episode.frame_count != EPISODE_FRAME_COUNT:
                raise RobotFreeVerificationError(
                    f"{split} episode {index} frame count mismatch"
                )
            if episode.fps != EPISODE_FPS:
                raise RobotFreeVerificationError(
                    f"{split} episode {index} FPS mismatch"
                )
            if episode.resolution != (
                SYNTHETIC_IMAGE_HEIGHT,
                SYNTHETIC_IMAGE_WIDTH,
            ):
                raise RobotFreeVerificationError(
                    f"{split} episode {index} resolution mismatch"
                )
            metadata = episode.metadata or {}
            if metadata.get("action_source") != (
                "synthetic_deterministic_no_robot_no_goal_write"
            ):
                raise RobotFreeVerificationError(
                    f"{split} episode {index} action source mismatch"
                )
            if metadata.get("evidence_level") != "offline":
                raise RobotFreeVerificationError(
                    f"{split} episode {index} evidence level mismatch"
                )
            result[split].append(episode)
    train_tasks = {(episode.task_index, episode.task) for episode in result["train"]}
    validation_tasks = {
        (episode.task_index, episode.task) for episode in result["validation"]
    }
    if len(train_tasks) != 1 or len(validation_tasks) != 1:
        raise RobotFreeVerificationError("each split must contain exactly one task")
    if not train_tasks.isdisjoint(validation_tasks):
        raise RobotFreeVerificationError("train and validation tasks must be disjoint")
    return result


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
    except RobotFreeVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobotFreeVerificationError(f"failed to read {name}: {path}") from error
    if not isinstance(value, dict):
        raise RobotFreeVerificationError(f"{name} must be a JSON object")
    _reject_nonfinite_json_numbers(value, name=name)
    return cast(dict[str, Any], value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RobotFreeVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise RobotFreeVerificationError(f"non-finite JSON constant: {value}")


def _reject_nonfinite_json_numbers(value: Any, *, name: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise RobotFreeVerificationError(f"{name} contains a non-finite JSON number")
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_nonfinite_json_numbers(nested, name=name)
    elif isinstance(value, list):
        for nested in value:
            _reject_nonfinite_json_numbers(nested, name=name)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RobotFreeVerificationError(f"{name} must be a string-keyed object")
    return cast(Mapping[str, Any], value)


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobotFreeVerificationError(f"{name} must be a non-empty string")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a robot-free result bundle without rerunning training or G8."
    )
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        summary = verify_robot_free_result(args.result, device=args.device)
    except (
        CheckpointError,
        ConfigError,
        DatasetError,
        DeploymentCertificationError,
        RobotFreePipelineError,
        RobotFreeVerificationError,
        OSError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ROBOT_FREE_VERIFICATION_SCHEMA_VERSION",
    "RobotFreeVerificationError",
    "verify_robot_free_result",
]
