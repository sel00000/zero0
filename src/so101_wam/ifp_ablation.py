"""Comparable K=0/2/4 training-only IFP ablation artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

import numpy as np

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import CheckpointError, load_compact_wam_bundle
from .config import (
    ConfigError,
    DEFAULT_ROBOT_FREE_CONFIG_PATH,
    ProjectConfig,
)
from .constants import ACTION_DIM
from .contracts import ContractError
from .dataset import DatasetError
from .deployment import canonical_json_sha256, file_sha256, project_config_sha256
from .ifp_results import (
    ClosedLoopResult,
    IFPAblationError,
    IFP_ABLATION_SCHEMA,
    IFP_PROTOCOL_LIMITATION,
)
from .model import ModelContractError
from .mujoco_identity import MujocoIdentityError, MujocoModelIdentity
from .mujoco_cli import (
    MujocoCLIError,
    run_mujoco_checkpoint_session,
    write_mujoco_report,
)
from .mujoco_semantic_suite import (
    MIN_SEMANTIC_CASES,
    MujocoSemanticSuiteError,
    SEMANTIC_SUITE_REPORT_SCHEMA,
    run_mujoco_semantic_suite,
)
from .policy import PolicyError
from .rollout import RolloutError
from .runtime import RuntimeErrorState
from .training import (
    CompactWAMTrainingConfig,
    IFPArchitecture,
    PAPER_IFP_STRIDE,
    PAPER_IFP_STEPS,
    TrainingArtifactExistsError,
    TrainingError,
    train_offline_candidate,
)
from .training_data import (
    EpisodeRecord,
    TrainingDataError,
    load_episode_records,
)


IFP_ABLATION_STEPS = (0, 2, PAPER_IFP_STEPS)
DEFAULT_POLICY_STEPS = 1
SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
TERMINAL_PROXY_SCOPE = "synthetic_terminal_joint_proxy"
SEMANTIC_SUITE_SCOPE = "mujoco_semantic_object_state_suite"
TERMINAL_ERROR_BASIS = "max_abs_joint_position_error"
SEMANTIC_ERROR_BASIS = "object_position_error_m_scored_trials"
TERMINAL_FAILURE_REASON = "terminal_joint_target_tolerance"


class ClosedLoopMode(StrEnum):
    """Available local closed-loop evidence scopes."""

    TERMINAL_JOINT_PROXY = "terminal-joint-proxy"
    SEMANTIC_SUITE_PROXY = "semantic-suite-proxy"


class ClosedLoopEvaluator(Protocol):
    def __call__(
        self,
        checkpoint_path: Path,
        *,
        ifp_steps: int,
    ) -> ClosedLoopResult: ...


@dataclass(frozen=True, slots=True)
class MujocoTerminalProxy:
    """Score one deterministic CompactWAM rollout against a joint target."""

    config: ProjectConfig
    prompt_path: Path
    target_joint_position: tuple[float, ...]
    tolerance: float
    policy_steps: int = DEFAULT_POLICY_STEPS
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.config.runtime.backend != "mujoco":
            raise IFPAblationError("terminal proxy requires a MuJoCo config")
        if not self.config.runtime.actuation_enabled:
            raise IFPAblationError("terminal proxy requires MuJoCo actuation")
        if not self.prompt_path.is_file():
            raise IFPAblationError(
                f"terminal proxy prompt is missing: {self.prompt_path}"
            )
        target = np.asarray(self.target_joint_position, dtype=np.float64)
        if target.shape != (ACTION_DIM,) or not np.isfinite(target).all():
            raise IFPAblationError(
                f"terminal target must contain {ACTION_DIM} finite values"
            )
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise IFPAblationError("terminal tolerance must be finite and positive")
        if (
            not isinstance(self.policy_steps, int)
            or isinstance(self.policy_steps, bool)
            or self.policy_steps < 1
        ):
            raise IFPAblationError("policy_steps must be a positive integer")

    def __call__(
        self,
        checkpoint_path: Path,
        *,
        ifp_steps: int,
    ) -> ClosedLoopResult:
        protocol = self._protocol()
        report = run_mujoco_checkpoint_session(
            self.config,
            checkpoint_path=checkpoint_path,
            prompt_path=self.prompt_path,
            policy_steps=self.policy_steps,
            device=self.device,
        )
        if self._protocol() != protocol:
            raise IFPAblationError("terminal proxy inputs changed during evaluation")
        protocol["mujoco_model_identity"] = _mujoco_identity(report)
        report["evaluation_protocol"] = protocol
        final = np.asarray(report.get("terminal_joint_position"), dtype=np.float64)
        if final.shape != (ACTION_DIM,) or not np.isfinite(final).all():
            raise IFPAblationError("MuJoCo report has no finite terminal joints")
        target = np.asarray(self.target_joint_position, dtype=np.float64)
        terminal_error = float(np.max(np.abs(final - target)))
        success = terminal_error <= self.tolerance
        report["terminal_proxy"] = {
            "scope": TERMINAL_PROXY_SCOPE,
            "ifp_steps": ifp_steps,
            "target_joint_position": target.tolist(),
            "tolerance": self.tolerance,
            "final_error_max_abs": terminal_error,
            "success": success,
            "object_manipulation_success": False,
        }
        artifact_path = checkpoint_path.with_suffix(".mujoco.json")
        write_mujoco_report(artifact_path, report)
        return ClosedLoopResult(
            scope=TERMINAL_PROXY_SCOPE,
            trial_count=1,
            scored_trial_count=1,
            execution_failure_count=0,
            success_count=int(success),
            success_rate=float(success),
            terminal_error_mean=terminal_error,
            terminal_error_basis=TERMINAL_ERROR_BASIS,
            protocol_sha256=canonical_json_sha256(protocol),
            failure_counts=(
                () if success else ((TERMINAL_FAILURE_REASON, 1),)
            ),
            artifact=artifact_path.name,
            artifact_sha256=file_sha256(artifact_path),
        )

    def _protocol(self) -> dict[str, Any]:
        return {
            "scope": TERMINAL_PROXY_SCOPE,
            "config_sha256": project_config_sha256(self.config),
            "prompt_sha256": file_sha256(self.prompt_path),
            "prompt_manifest_sha256": file_sha256(self.prompt_path.with_suffix(".json")),
            "target_joint_position": list(self.target_joint_position),
            "tolerance": float(self.tolerance),
            "policy_steps": self.policy_steps,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class MujocoSemanticSuiteProxy:
    """Score one checkpoint on the local multi-case semantic suite."""

    config: ProjectConfig
    suite_path: Path
    mapping_path: Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.config.runtime.backend != "mujoco":
            raise IFPAblationError("semantic suite proxy requires a MuJoCo config")
        if not self.config.runtime.actuation_enabled:
            raise IFPAblationError("semantic suite proxy requires MuJoCo actuation")
        if not self.suite_path.is_file():
            raise IFPAblationError(
                f"semantic suite is missing: {self.suite_path}"
            )
        if not self.mapping_path.is_file():
            raise IFPAblationError(
                f"semantic mapping is missing: {self.mapping_path}"
            )

    def __call__(
        self,
        checkpoint_path: Path,
        *,
        ifp_steps: int,
    ) -> ClosedLoopResult:
        del ifp_steps
        training_report_path = checkpoint_path.with_suffix(".training.json")
        artifact_dir = checkpoint_path.with_suffix(".semantic-suite-artifacts")
        report_path = checkpoint_path.with_suffix(".semantic-suite.json")
        report = run_mujoco_semantic_suite(
            self.config,
            suite_path=self.suite_path,
            mapping_path=self.mapping_path,
            checkpoint_path=checkpoint_path,
            training_report_path=training_report_path,
            artifact_dir=artifact_dir,
            report_path=report_path,
            device=self.device,
        )
        # Hash the same bytes whose contents supply the returned metrics.
        try:
            source = report_path.read_bytes()
            persisted = json.loads(source)
        except (OSError, ValueError) as error:
            raise IFPAblationError(f"invalid persisted semantic report: {error}") from error
        if persisted != report:
            raise IFPAblationError("persisted semantic report disagrees with runner")
        report = persisted
        expected = {
            "schema_version": SEMANTIC_SUITE_REPORT_SCHEMA,
            "result": "complete",
            "robot_used": False,
            "semantic_heldout_success_claimed": False,
            "prompt_causality_claimed": False,
            "real_world_success_claimed": False,
            "official_zero_wam_claimed": False,
            "independent_mapping_verified": False,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "training_report_sha256": file_sha256(training_report_path),
            "mujoco_config_sha256": project_config_sha256(self.config),
        }
        if any(
            type(report.get(key)) is not type(value) or report.get(key) != value
            for key, value in expected.items()
        ):
            raise IFPAblationError("semantic suite report identity or semantics mismatch")
        if not artifact_dir.is_dir():
            raise IFPAblationError("semantic suite evidence artifacts are missing")
        summary = report.get("summary")
        if not isinstance(summary, Mapping):
            raise IFPAblationError("semantic suite report is missing summary")
        result = ClosedLoopResult.from_payload({
            "scope": SEMANTIC_SUITE_SCOPE,
            "trial_count": summary.get("total_trial_count"),
            "scored_trial_count": summary.get("scored_trial_count"),
            "execution_failure_count": summary.get("execution_failure_count"),
            "success_count": summary.get("success_count"),
            "success_rate": summary.get("success_rate"),
            "terminal_error_mean": summary.get("object_position_error_mean_m"),
            "terminal_error_basis": SEMANTIC_ERROR_BASIS,
            "failure_counts": summary.get("failure_counts"),
            "protocol_sha256": _semantic_protocol_hash(report, device=self.device),
            "artifact": report_path.name,
            "artifact_sha256": sha256(source).hexdigest(),
        })
        if report.get("semantic_mujoco_object_state_evaluated") is not (
            result.scored_trial_count > 0
        ):
            raise IFPAblationError("semantic suite scored flag disagrees with counts")
        return result


def _mujoco_identity(report: Mapping[str, Any]) -> dict[str, object]:
    identity = report.get("mujoco_model_identity")
    if not isinstance(identity, Mapping):
        raise IFPAblationError("MuJoCo report requires compiled model identity")
    try:
        return MujocoModelIdentity.from_payload(identity).to_payload()
    except MujocoIdentityError as error:
        raise IFPAblationError(f"invalid MuJoCo model identity: {error}") from error


def _semantic_protocol_hash(report: Mapping[str, Any], *, device: str) -> str:
    def digest(value: Mapping[str, Any], key: str) -> str:
        result = value.get(key)
        if not isinstance(result, str) or SHA256_HEX_RE.fullmatch(result) is None:
            raise IFPAblationError(f"semantic protocol requires {key}")
        return result

    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) < MIN_SEMANTIC_CASES:
        raise IFPAblationError("semantic protocol requires multiple cases")
    inputs: dict[str, dict[str, str]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise IFPAblationError("semantic protocol case is invalid")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in inputs:
            raise IFPAblationError("semantic protocol case ID is invalid")
        inputs[case_id] = {
            key: digest(case, key)
            for key in (
                "semantic_manifest_sha256",
                "prompt_npz_sha256",
                "prompt_manifest_sha256",
                "object_task_signature_sha256",
                "object_physical_profile_sha256",
            )
        }
    # Only frozen inputs participate; checkpoint and outcome changes are expected.
    return canonical_json_sha256({
        "scope": SEMANTIC_SUITE_SCOPE,
        "device": device,
        "mujoco_model_identity": _mujoco_identity(report),
        "suite_sha256": digest(report, "suite_sha256"),
        "mapping_sha256": digest(report, "mapping_sha256"),
        "mujoco_config_sha256": digest(report, "mujoco_config_sha256"),
        "cases": inputs,
    })


def _require_empty_output(output_dir: Path, report_path: Path) -> None:
    if report_path.exists():
        raise TrainingArtifactExistsError(
            f"IFP ablation report already exists: {report_path}"
        )
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise IFPAblationError(f"ablation output is not a directory: {output_dir}")
    existing = next(output_dir.iterdir(), None)
    if existing is not None:
        raise TrainingArtifactExistsError(
            f"IFP ablation output must be empty: {output_dir}"
        )


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IFPAblationError(f"failed to read training report: {error}") from error
    if not isinstance(payload, dict):
        raise IFPAblationError("training report must be a JSON object")
    return payload


def _nested(report: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        raise IFPAblationError(f"training report is missing {key}")
    return value


def _variant(
    *,
    ifp_steps: int,
    checkpoint_path: Path,
    training_report_path: Path,
    training_report: Mapping[str, Any],
    closed_loop: ClosedLoopResult,
    output_dir: Path,
) -> dict[str, Any]:
    model = _nested(training_report, "model")
    if model.get("ifp_steps") != 0:
        raise IFPAblationError("fused IFP checkpoint must export ifp_steps=0")
    protocol = _nested(training_report, "protocol")
    if (
        protocol.get("ifp_architecture") != IFPArchitecture.FUSED_MODULES.value
        or protocol.get("ifp_module_removed_from_inference") is not True
    ):
        raise IFPAblationError("training report does not prove IFP removal")
    optimization = _nested(training_report, "optimization")
    if (
        optimization.get("ifp_steps") != ifp_steps
        or optimization.get("ifp_window_steps") != PAPER_IFP_STEPS
    ):
        raise IFPAblationError("training report has the wrong IFP variant")

    bundle = load_compact_wam_bundle(checkpoint_path, device="cpu")
    if bundle.model.ifp_steps != 0 or any(
        "ifp" in key for key in bundle.model.state_dict()
    ):
        raise IFPAblationError("checkpoint contains inference-time IFP state")
    if (
        bundle.metadata.get("training_ifp_steps") != ifp_steps
        or bundle.metadata.get("ifp_architecture")
        != IFPArchitecture.FUSED_MODULES.value
        or bundle.metadata.get("inference_ifp_module_present") is not False
    ):
        raise IFPAblationError("checkpoint IFP metadata is inconsistent")

    if closed_loop.artifact is not None:
        artifact_path = output_dir / closed_loop.artifact
        if not artifact_path.is_file():
            raise IFPAblationError("closed-loop artifact is missing")
        if file_sha256(artifact_path) != closed_loop.artifact_sha256:
            raise IFPAblationError("closed-loop artifact checksum mismatch")

    validation = dict(_nested(training_report, "validation"))
    return {
        "ifp_steps": ifp_steps,
        "checkpoint_id": training_report.get("checkpoint_id"),
        "checkpoint": checkpoint_path.relative_to(output_dir).as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_report": training_report_path.relative_to(output_dir).as_posix(),
        "training_report_sha256": file_sha256(training_report_path),
        "initial_model_state_sha256": training_report.get("initial_model_state_sha256"),
        "training_schedule_sha256": training_report.get("training_schedule_sha256"),
        "data": dict(_nested(training_report, "data")),
        "optimization": dict(optimization),
        "validation": validation,
        "inference": {
            "checkpoint_ifp_steps": 0,
            "ifp_module_present": False,
        },
        "closed_loop": closed_loop.to_payload(),
    }


def _same(variants: Sequence[Mapping[str, Any]], field: str) -> bool:
    return len({json.dumps(item[field], sort_keys=True) for item in variants}) == 1


def _comparability(variants: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    optimizations = [_nested(item, "optimization") for item in variants]
    data = [_nested(item, "data") for item in variants]
    protocols = {
        (
            item["closed_loop"]["scope"],
            item["closed_loop"]["protocol_sha256"],
            item["closed_loop"]["trial_count"],
            item["closed_loop"]["terminal_error_basis"],
        )
        for item in variants
    }
    result = {
        "same_closed_loop_protocol": len(protocols) == 1,
        "same_initial_model_state": _same(variants, "initial_model_state_sha256"),
        "same_optimizer_budget": len(
            {
                (
                    item.get("stage1_steps"),
                    item.get("stage2_steps"),
                )
                for item in optimizations
            }
        )
        == 1,
        "same_seed": len({item.get("seed") for item in optimizations}) == 1,
        "same_train_split": len({item.get("train_split_sha256") for item in data}) == 1,
        "same_training_schedule": _same(variants, "training_schedule_sha256"),
        "same_validation_split": len(
            {item.get("validation_split_sha256") for item in data}
        )
        == 1,
        "same_window_count": len(
            {
                (
                    item.get("train_window_count"),
                    item.get("validation_window_count"),
                )
                for item in data
            }
        )
        == 1,
    }
    if not all(result.values()):
        failed = sorted(key for key, value in result.items() if not value)
        raise IFPAblationError(f"IFP variants are not comparable: {failed}")
    return result


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise TrainingArtifactExistsError(f"IFP ablation report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as error:
            raise TrainingArtifactExistsError(
                f"IFP ablation report already exists: {path}"
            ) from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run_ifp_ablation(
    train_records: Sequence[EpisodeRecord],
    validation_records: Sequence[EpisodeRecord],
    *,
    output_dir: str | Path,
    report_path: str | Path,
    checkpoint_id_prefix: str,
    base_config: CompactWAMTrainingConfig,
    closed_loop_evaluator: ClosedLoopEvaluator,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train comparable K variants and publish descriptive evidence only."""

    if not isinstance(checkpoint_id_prefix, str) or not checkpoint_id_prefix.strip():
        raise IFPAblationError("checkpoint_id_prefix must be non-empty")
    if base_config.ifp_stride != PAPER_IFP_STRIDE:
        raise IFPAblationError(f"IFP ablation requires stride={PAPER_IFP_STRIDE}")
    output_target = Path(output_dir).resolve()
    report_target = Path(report_path).resolve()
    _require_empty_output(output_target, report_target)
    reserved_paths = {
        output_target / f"k{ifp_steps}{suffix}"
        for ifp_steps in IFP_ABLATION_STEPS
        for suffix in (
            ".pt",
            ".training.json",
            ".mujoco.json",
            ".semantic-suite.json",
            ".semantic-suite-artifacts",
        )
    }
    if report_target in reserved_paths:
        raise IFPAblationError("ablation report collides with a variant artifact")
    output_target.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, Any]] = []
    for ifp_steps in IFP_ABLATION_STEPS:
        checkpoint_path = output_target / f"k{ifp_steps}.pt"
        training_report_path = output_target / f"k{ifp_steps}.training.json"
        config = replace(
            base_config,
            ifp_steps=ifp_steps,
            ifp_architecture=IFPArchitecture.FUSED_MODULES,
            ifp_window_steps=PAPER_IFP_STEPS,
        )
        train_offline_candidate(
            train_records,
            validation_records,
            checkpoint_path=checkpoint_path,
            report_path=training_report_path,
            checkpoint_id=f"{checkpoint_id_prefix.strip()}-k{ifp_steps}",
            config=config,
            device=device,
        )
        training_report = _read_report(training_report_path)
        closed_loop = closed_loop_evaluator(
            checkpoint_path,
            ifp_steps=ifp_steps,
        )
        variants.append(
            _variant(
                ifp_steps=ifp_steps,
                checkpoint_path=checkpoint_path,
                training_report_path=training_report_path,
                training_report=training_report,
                closed_loop=closed_loop,
                output_dir=output_target,
            )
        )

    report = {
        "schema_version": IFP_ABLATION_SCHEMA,
        "result": "complete",
        "evidence_level": "offline_and_simulation_diagnostic",
        "robot_used": False,
        "scientific_claim": "not_evaluated",
        "result_semantics": "descriptive K=0/2/4 comparison",
        "comparability": _comparability(variants),
        "variants": variants,
        "limitations": [
            IFP_PROTOCOL_LIMITATION,
            "compact latent MSE is not the paper's flow-matching objective",
            "prompt perturbation deltas do not prove causal task understanding",
            "closed-loop results remain local simulation diagnostics",
            "semantic task mappings are not independently verified",
            "offline and MuJoCo evidence cannot authorize real robot output",
        ],
    }
    _write_json_no_overwrite(report_target, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a comparable K=0/2/4 IFP ablation."
    )
    parser.add_argument("--train-episodes", nargs="+", required=True, type=Path)
    parser.add_argument("--validation-episodes", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--checkpoint-id-prefix", required=True)
    parser.add_argument(
        "--mujoco-config",
        type=Path,
        default=DEFAULT_ROBOT_FREE_CONFIG_PATH,
    )
    parser.add_argument(
        "--closed-loop-mode",
        choices=tuple(ClosedLoopMode),
        default=ClosedLoopMode.TERMINAL_JOINT_PROXY,
        type=ClosedLoopMode,
    )
    parser.add_argument("--terminal-tolerance", type=float)
    parser.add_argument("--semantic-suite", type=Path)
    parser.add_argument("--semantic-mapping", type=Path)
    parser.add_argument("--policy-steps", default=DEFAULT_POLICY_STEPS, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--stage1-steps", default=100, type=int)
    parser.add_argument("--stage2-steps", default=100, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--latent-dim", default=32, type=int)
    parser.add_argument("--transformer-layers", default=1, type=int)
    parser.add_argument("--transformer-heads", default=4, type=int)
    parser.add_argument("--future-steps", default=3, type=int)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--action-history-steps", default=4, type=int)
    parser.add_argument("--max-context-steps", default=300, type=int)
    parser.add_argument("--future-latent-weight", default=1.0, type=float)
    parser.add_argument("--action-weight", default=1.0, type=float)
    parser.add_argument("--ifp-weight", default=0.25, type=float)
    args = parser.parse_args(argv)

    report_path = args.report or args.output_dir / "ifp_ablation.json"
    try:
        config = ProjectConfig.load(args.mujoco_config)
        if (
            args.action_horizon is not None
            and args.action_horizon != config.runtime.action_horizon
        ):
            raise IFPAblationError(
                "action_horizon must match MuJoCo runtime: "
                f"{args.action_horizon} != {config.runtime.action_horizon}"
            )
        train_records = load_episode_records(args.train_episodes)
        validation_records = load_episode_records(args.validation_episodes)
        if not validation_records:
            raise IFPAblationError("validation episodes must not be empty")
        action_horizon = (
            config.runtime.action_horizon
            if args.action_horizon is None
            else args.action_horizon
        )
        base_config = CompactWAMTrainingConfig(
            policy_hz=config.runtime.policy_hz,
            servo_hz=config.runtime.servo_hz,
            latent_dim=args.latent_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            future_steps=args.future_steps,
            action_horizon=action_horizon,
            action_history_steps=args.action_history_steps,
            ifp_steps=0,
            ifp_stride=PAPER_IFP_STRIDE,
            ifp_architecture=IFPArchitecture.FUSED_MODULES,
            ifp_window_steps=PAPER_IFP_STEPS,
            max_context_steps=args.max_context_steps,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            future_latent_weight=args.future_latent_weight,
            action_weight=args.action_weight,
            ifp_weight=args.ifp_weight,
            seed=args.seed,
        )
        if args.closed_loop_mode is ClosedLoopMode.SEMANTIC_SUITE_PROXY:
            if args.semantic_suite is None or args.semantic_mapping is None:
                raise IFPAblationError(
                    "semantic suite proxy requires --semantic-suite and "
                    "--semantic-mapping"
                )
            evaluator: ClosedLoopEvaluator = MujocoSemanticSuiteProxy(
                config=config,
                suite_path=args.semantic_suite,
                mapping_path=args.semantic_mapping,
                device=args.device,
            )
        else:
            if args.terminal_tolerance is None:
                raise IFPAblationError(
                    "terminal joint proxy requires --terminal-tolerance"
                )
            prompt_record = validation_records[0]
            evaluator = MujocoTerminalProxy(
                config=config,
                prompt_path=prompt_record.path,
                target_joint_position=tuple(
                    float(value) for value in prompt_record.data.joint_state[-1]
                ),
                tolerance=args.terminal_tolerance,
                policy_steps=args.policy_steps,
                device=args.device,
            )
        report = run_ifp_ablation(
            train_records,
            validation_records,
            output_dir=args.output_dir,
            report_path=report_path,
            checkpoint_id_prefix=args.checkpoint_id_prefix,
            base_config=base_config,
            closed_loop_evaluator=evaluator,
            device=args.device,
        )
    except (
        CheckpointError,
        ConfigError,
        ContractError,
        DatasetError,
        IFPAblationError,
        ModelContractError,
        MujocoAdapterError,
        MujocoCLIError,
        MujocoSemanticSuiteError,
        OSError,
        PolicyError,
        RolloutError,
        RuntimeErrorState,
        TrainingDataError,
        TrainingError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "ClosedLoopEvaluator",
    "ClosedLoopMode",
    "ClosedLoopResult",
    "IFP_ABLATION_SCHEMA",
    "IFP_ABLATION_STEPS",
    "IFPAblationError",
    "MujocoSemanticSuiteProxy",
    "MujocoTerminalProxy",
    "main",
    "run_ifp_ablation",
]


if __name__ == "__main__":
    raise SystemExit(main())
