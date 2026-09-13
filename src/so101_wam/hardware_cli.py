"""Fail-closed CompactWAM bring-up CLI for a real LeRobot BiSOFollower."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Sequence

from .adapters.lerobot import LeRobotBiSOAdapter
from .checkpoint import CheckpointError, load_compact_wam_bundle
from .config import ConfigError, ProjectConfig
from .contracts import ContractError
from .dataset import DatasetError, load_episode, physical_prompt_from_episode
from .deployment import (
    DeploymentCertificationError,
    file_sha256,
    verify_deployment_certification,
)
from .deployment_issuer import (
    DeploymentIssuanceError,
    validate_deployment_source_evidence,
    verify_exact_model_state,
)
from .lerobot_factory import LeRobotFactoryError, create_bi_so_follower
from .policy import CompactWAMPolicy, PolicyError
from .rollout import RolloutError, RolloutSummary, run_managed_rollout
from .runtime import RuntimeErrorState, SO101WAMRuntime


class HardwareCLIError(ValueError):
    """Raised when explicit hardware bring-up gates are incomplete."""


def resolve_output_mode(
    config: ProjectConfig,
    *,
    enable_real_output: bool,
    acknowledged_hardware_id: str,
) -> ProjectConfig:
    """Require both config opt-in and an exact command-line hardware acknowledgement."""

    if enable_real_output:
        if not config.runtime.actuation_enabled:
            raise HardwareCLIError(
                "real output requires runtime.actuation_enabled=true in the hardware config"
            )
        if acknowledged_hardware_id != config.lerobot.hardware_id:
            raise HardwareCLIError("--ack-hardware-id must exactly match lerobot.hardware_id")
        return config

    return replace(config, runtime=replace(config.runtime, actuation_enabled=False))


def _require_deployment_checkpoint(
    metadata: Mapping[str, str | int | float | bool | None],
) -> None:
    if (
        metadata.get("offline_trained") is True
        or metadata.get("artifact_kind") == "compact_wam_candidate"
        or metadata.get("evidence_level") == "offline"
    ):
        raise HardwareCLIError(
            "real output rejects offline CompactWAM candidate artifacts"
        )
    if metadata.get("artifact_kind") != "compact_wam_deployment":
        raise HardwareCLIError(
            "real output requires artifact_kind=compact_wam_deployment"
        )
    if metadata.get("trained") is not True:
        raise HardwareCLIError("real output requires checkpoint metadata trained=true")
    if metadata.get("deployment_ready") is not True:
        raise HardwareCLIError(
            "real output requires checkpoint metadata deployment_ready=true"
        )
    checkpoint_id = metadata.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise HardwareCLIError("real output requires a non-empty checkpoint_id")
    evidence = metadata.get("training_evidence_sha256")
    if (
        not isinstance(evidence, str)
        or len(evidence) != 64
        or any(character not in "0123456789abcdef" for character in evidence)
    ):
        raise HardwareCLIError(
            "real output requires a 64-character lowercase training_evidence_sha256"
        )
    deployment_evidence = metadata.get("deployment_evidence_sha256")
    if (
        not isinstance(deployment_evidence, str)
        or len(deployment_evidence) != 64
        or any(
            character not in "0123456789abcdef"
            for character in deployment_evidence
        )
    ):
        raise HardwareCLIError(
            "real output requires a 64-character lowercase "
            "deployment_evidence_sha256"
        )


def run_hardware_session(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    prompt_path: str | Path,
    manifest_path: str | Path | None = None,
    deployment_certification_path: str | Path | None = None,
    source_candidate_checkpoint_path: str | Path | None = None,
    source_training_report_path: str | Path | None = None,
    source_preflight_report_path: str | Path | None = None,
    source_mujoco_config_path: str | Path | None = None,
    source_mujoco_report_path: str | Path | None = None,
    source_manual_signoff_path: str | Path | None = None,
    policy_steps: int,
    device: str = "cpu",
    enable_real_output: bool = False,
    acknowledged_hardware_id: str = "",
    calibrate_on_connect: bool = False,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = sleep,
) -> RolloutSummary:
    """Load immutable artifacts, construct hardware, and run one managed session."""

    effective = resolve_output_mode(
        config,
        enable_real_output=enable_real_output,
        acknowledged_hardware_id=acknowledged_hardware_id,
    )
    checkpoint_sha256_before = (
        file_sha256(checkpoint_path) if enable_real_output else None
    )
    bundle = load_compact_wam_bundle(checkpoint_path, device=device)
    source_paths: dict[str, str | Path] | None = None
    if enable_real_output:
        _require_deployment_checkpoint(bundle.metadata)
        if deployment_certification_path is None:
            raise HardwareCLIError(
                "real output requires an external deployment certification JSON"
            )
        supplied_sources = {
            "candidate checkpoint": source_candidate_checkpoint_path,
            "training report": source_training_report_path,
            "preflight report": source_preflight_report_path,
            "MuJoCo config": source_mujoco_config_path,
            "MuJoCo report": source_mujoco_report_path,
            "manual signoff": source_manual_signoff_path,
        }
        missing_sources = tuple(
            name for name, path in supplied_sources.items() if path is None
        )
        if missing_sources:
            raise HardwareCLIError(
                "real output requires revalidation paths for source evidence: "
                f"{missing_sources}"
            )
        source_paths = {
            name: path
            for name, path in supplied_sources.items()
            if path is not None
        }
        checkpoint_sha256_after = file_sha256(checkpoint_path)
        if checkpoint_sha256_after != checkpoint_sha256_before:
            raise HardwareCLIError(
                "checkpoint changed while it was being loaded for real output"
            )
    if bundle.model.action_horizon != effective.runtime.action_horizon:
        raise HardwareCLIError(
            "checkpoint action_horizon does not match runtime.action_horizon: "
            f"{bundle.model.action_horizon} != {effective.runtime.action_horizon}"
        )

    effective_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else Path(prompt_path).with_suffix(".json")
    )
    prompt_npz_sha256_before = (
        file_sha256(prompt_path) if enable_real_output else None
    )
    prompt_manifest_sha256_before = (
        file_sha256(effective_manifest_path) if enable_real_output else None
    )
    episode = load_episode(prompt_path, manifest_path)
    prompt = physical_prompt_from_episode(episode, policy_hz=effective.runtime.policy_hz)
    if enable_real_output:
        prompt_npz_sha256_after = file_sha256(prompt_path)
        prompt_manifest_sha256_after = file_sha256(effective_manifest_path)
        if prompt_npz_sha256_after != prompt_npz_sha256_before:
            raise HardwareCLIError(
                "prompt NPZ changed while it was being loaded for real output"
            )
        if prompt_manifest_sha256_after != prompt_manifest_sha256_before:
            raise HardwareCLIError(
                "prompt manifest changed while it was being loaded for real output"
            )
        assert deployment_certification_path is not None
        assert checkpoint_sha256_after is not None
        assert source_paths is not None
        validated_sources = validate_deployment_source_evidence(
            effective,
            candidate_checkpoint_path=source_paths["candidate checkpoint"],
            training_report_path=source_paths["training report"],
            preflight_report_path=source_paths["preflight report"],
            mujoco_config_path=source_paths["MuJoCo config"],
            mujoco_report_path=source_paths["MuJoCo report"],
            prompt_npz_path=prompt_path,
            prompt_manifest_path=effective_manifest_path,
            manual_signoff_path=source_paths["manual signoff"],
        )
        if validated_sources.source_evidence_sha256["prompt_npz"] != (
            prompt_npz_sha256_after
        ) or validated_sources.source_evidence_sha256["prompt_manifest"] != (
            prompt_manifest_sha256_after
        ):
            raise HardwareCLIError(
                "runtime prompt changed during source-evidence revalidation"
            )
        verify_exact_model_state(
            validated_sources.candidate.model,
            bundle.model,
        )
        verify_deployment_certification(
            deployment_certification_path,
            checkpoint_sha256=checkpoint_sha256_after,
            source_evidence_sha256=validated_sources.source_evidence_sha256,
            checkpoint_metadata=bundle.metadata,
            config=effective,
            policy_steps=policy_steps,
        )
    raw_robot = create_bi_so_follower(effective.lerobot)
    adapter = LeRobotBiSOAdapter(
        raw_robot,
        config=effective.lerobot,
        actuation_enabled=effective.runtime.actuation_enabled,
        max_camera_age_s=effective.safety.max_observation_age_s,
    )
    policy = CompactWAMPolicy(
        bundle.model,
        servo_hz=effective.runtime.servo_hz,
        device=device,
    )
    runtime = SO101WAMRuntime(
        config=effective,
        prompt=prompt,
        robot=adapter,
        policy=policy,
        clock=clock,
    )
    return run_managed_rollout(
        runtime,
        adapter,
        policy_steps=policy_steps,
        calibrate_on_connect=calibrate_on_connect,
        clock=clock,
        sleeper=sleeper,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a checkpointed SO101-WAM hardware session.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True, help="SO101-WAM episode .npz")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--deployment-certification", type=Path)
    parser.add_argument("--source-candidate", type=Path)
    parser.add_argument("--source-training-report", type=Path)
    parser.add_argument("--source-preflight-report", type=Path)
    parser.add_argument("--source-mujoco-config", type=Path)
    parser.add_argument("--source-mujoco-report", type=Path)
    parser.add_argument("--source-manual-signoff", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibrate-on-connect", action="store_true")
    parser.add_argument("--enable-real-output", action="store_true")
    parser.add_argument("--ack-hardware-id", default="")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")

    try:
        summary = run_hardware_session(
            ProjectConfig.load(args.config),
            checkpoint_path=args.checkpoint,
            prompt_path=args.prompt,
            manifest_path=args.manifest,
            deployment_certification_path=args.deployment_certification,
            source_candidate_checkpoint_path=args.source_candidate,
            source_training_report_path=args.source_training_report,
            source_preflight_report_path=args.source_preflight_report,
            source_mujoco_config_path=args.source_mujoco_config,
            source_mujoco_report_path=args.source_mujoco_report,
            source_manual_signoff_path=args.source_manual_signoff,
            policy_steps=args.steps,
            device=args.device,
            enable_real_output=args.enable_real_output,
            acknowledged_hardware_id=args.ack_hardware_id,
            calibrate_on_connect=args.calibrate_on_connect,
        )
    except (
        CheckpointError,
        ConfigError,
        ContractError,
        DatasetError,
        DeploymentCertificationError,
        DeploymentIssuanceError,
        HardwareCLIError,
        LeRobotFactoryError,
        PolicyError,
        RolloutError,
        RuntimeErrorState,
    ) as error:
        parser.error(str(error))

    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
