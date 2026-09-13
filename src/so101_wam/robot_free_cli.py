"""Robot-free offline training plus MuJoCo G8 simulation demo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoint import CheckpointError, load_compact_wam_bundle
from .config import ConfigError, DEFAULT_ROBOT_FREE_CONFIG_PATH, ProjectConfig
from .constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from .contracts import ContractError, SensorimotorFrame
from .dataset import DatasetArtifactExistsError, DatasetError, EpisodeBuffer
from .deployment import DeploymentCertificationError, file_sha256
from .hardware_cli import HardwareCLIError, _require_deployment_checkpoint
from .model import ModelContractError
from .mujoco_cli import (
    MujocoCLIError,
    run_mujoco_checkpoint_session,
    write_mujoco_report,
)
from .adapters.mujoco import MujocoAdapterError
from .policy import PolicyError
from .rollout import RolloutError
from .runtime import RuntimeErrorState
from .training import (
    CompactWAMTrainingConfig,
    TrainingError,
    train_offline_candidate,
)
from .training_data import TrainingDataError, load_episode_records


ROBOT_FREE_SCHEMA_VERSION = "so101_wam.robot_free_pipeline.v1"
DEFAULT_CONFIG_PATH = DEFAULT_ROBOT_FREE_CONFIG_PATH
EPISODE_FPS = 30.0
EPISODE_DURATION_S = 3.0
EPISODE_FRAME_COUNT = int(EPISODE_FPS * EPISODE_DURATION_S) + 1
SYNTHETIC_IMAGE_HEIGHT = 24
SYNTHETIC_IMAGE_WIDTH = 32
ROBOT_FREE_LIMITATIONS = (
    "synthetic episodes are not real SO-101 demonstrations",
    "offline metrics do not prove task success",
    "MuJoCo G8 simulation is not G6/G7/G9/G10 real hardware evidence",
    "no real Goal_Position commands were sent",
    "real collision margins and calibrated hardware identity remain unmeasured",
)


class RobotFreePipelineError(RuntimeError):
    """Raised when the robot-free demo cannot publish immutable evidence."""


def run_robot_free_demo(
    output_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
    seed: int = 7,
    policy_steps: int = 1,
) -> Mapping[str, Any]:
    """Generate synthetic data, train a candidate, and run G8 in MuJoCo.

    The returned result is explicit that this is offline/simulation evidence only
    and that the generated candidate is rejected for real hardware output.
    """

    if policy_steps != 1:
        raise RobotFreePipelineError("robot-free pipeline requires policy_steps=1")
    output_root = Path(output_dir)
    _require_empty_output_dir(output_root)
    episodes_dir = output_root / "episodes"
    train_dir = episodes_dir / "train"
    validation_dir = episodes_dir / "validation"
    checkpoint_path = output_root / "candidate.pt"
    training_report_path = output_root / "candidate.training.json"
    g8_report_path = output_root / "mujoco.g8.json"
    result_path = output_root / "robot_free_result.json"
    effective_config_path = output_root / "effective_config.toml"
    checkpoint_id = f"robot-free-candidate-seed-{seed}"

    config = ProjectConfig.load(config_path)
    _require_robot_free_config(config)
    training_config = _robot_free_training_config(seed)
    if not output_root.exists():
        output_root.mkdir(parents=True)
    _copy_file_no_overwrite(Path(config_path), effective_config_path)
    effective_config = ProjectConfig.load(effective_config_path)
    _require_robot_free_config(effective_config)
    if effective_config != config:
        raise RobotFreePipelineError(
            "robot-free config changed while the immutable copy was being published"
        )
    config = effective_config
    episode_paths = _write_synthetic_episodes(
        train_dir=train_dir,
        validation_dir=validation_dir,
        seed=seed,
    )
    training = train_offline_candidate(
        load_episode_records([train_dir]),
        load_episode_records([validation_dir]),
        checkpoint_path=checkpoint_path,
        report_path=training_report_path,
        checkpoint_id=checkpoint_id,
        config=training_config,
        device=device,
    )
    g8_report = run_mujoco_checkpoint_session(
        config,
        checkpoint_path=checkpoint_path,
        prompt_path=episode_paths["validation"][0],
        policy_steps=policy_steps,
        device=device,
    )
    _require_actuated_g8(g8_report)
    write_mujoco_report(g8_report_path, g8_report)
    real_output = _check_real_output_guard(checkpoint_path, device=device)
    training_summary = {
        **asdict(training),
        "checkpoint_path": _bundle_relative_path(output_root, checkpoint_path),
        "report_path": _bundle_relative_path(output_root, training_report_path),
        "config": asdict(training_config),
    }
    result = {
        "schema_version": ROBOT_FREE_SCHEMA_VERSION,
        "result": "pass",
        "mode": "robot_free",
        "evidence_level": "simulation",
        "evidence_inputs": ["offline", "simulation"],
        "trained": False,
        "deployment_ready": False,
        "candidate_checkpoint_sha256": file_sha256(checkpoint_path),
        "g8_report_sha256": file_sha256(g8_report_path),
        "synthetic_data": {
            "fps": EPISODE_FPS,
            "duration_s": EPISODE_DURATION_S,
            "frame_count": EPISODE_FRAME_COUNT,
            "resolution": {
                "height": SYNTHETIC_IMAGE_HEIGHT,
                "width": SYNTHETIC_IMAGE_WIDTH,
            },
            "action_source": "synthetic_deterministic_no_robot_no_goal_write",
            "train_episode_count": len(episode_paths["train"]),
            "validation_episode_count": len(episode_paths["validation"]),
            "train_task_count": 1,
            "validation_task_count": 1,
            "task_split": "task_disjoint",
        },
        "training": training_summary,
        "g8": {
            "result": g8_report["result"],
            "gate": g8_report["gate"],
            "policy_steps": policy_steps,
            "checkpoint_id": g8_report["checkpoint_id"],
            "checkpoint_sha256": g8_report["checkpoint_sha256"],
            "report_sha256": file_sha256(g8_report_path),
        },
        "real_output_authorized": real_output["real_output_authorized"],
        "real_output_rejection_reason": real_output["rejection_reason"],
        "artifacts": _artifact_manifest(
            output_root,
            {
                "checkpoint": checkpoint_path,
                "training_report": training_report_path,
                "g8_report": g8_report_path,
                "config": effective_config_path,
                **{
                    f"train_episode_{index:02d}": path
                    for index, path in enumerate(episode_paths["train"])
                },
                **{
                    f"train_episode_manifest_{index:02d}": path.with_suffix(".json")
                    for index, path in enumerate(episode_paths["train"])
                },
                **{
                    f"validation_episode_{index:02d}": path
                    for index, path in enumerate(episode_paths["validation"])
                },
                **{
                    f"validation_episode_manifest_{index:02d}": path.with_suffix(
                        ".json"
                    )
                    for index, path in enumerate(episode_paths["validation"])
                },
            },
        ),
        "limitations": list(ROBOT_FREE_LIMITATIONS),
    }
    _write_json_no_overwrite(result_path, result)
    return result


def _robot_free_training_config(seed: int) -> CompactWAMTrainingConfig:
    return CompactWAMTrainingConfig(
        latent_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        future_steps=1,
        action_horizon=10,
        action_history_steps=1,
        ifp_steps=1,
        ifp_stride=2,
        max_context_steps=300,
        stage1_steps=0,
        stage2_steps=1,
        seed=seed,
    )


def run_robot_free_pipeline(
    output_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
    seed: int = 7,
    policy_steps: int = 1,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the robot-free public API."""

    return run_robot_free_demo(
        output_dir,
        config_path=config_path,
        device=device,
        seed=seed,
        policy_steps=policy_steps,
    )


def _write_synthetic_episodes(
    *,
    train_dir: Path,
    validation_dir: Path,
    seed: int,
) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {"train": [], "validation": []}
    specs = (
        ("train", train_dir, "synthetic-train-reach", 101, 0.0),
        ("validation", validation_dir, "synthetic-validation-place", 201, 0.7),
    )
    episode_index = 0
    for split, directory, task, task_index, base_phase in specs:
        for repeat_index in range(2):
            stem = f"episode_{episode_index:06d}"
            npz_path, _ = _make_episode(
                directory,
                stem=stem,
                task=task,
                task_index=task_index,
                episode_index=episode_index,
                phase=base_phase + repeat_index * 0.17 + seed * 0.001,
            )
            paths[split].append(npz_path)
            episode_index += 1
    return paths


def _make_episode(
    directory: Path,
    *,
    stem: str,
    task: str,
    task_index: int,
    episode_index: int,
    phase: float,
) -> tuple[Path, Path]:
    buffer = EpisodeBuffer(
        fps=EPISODE_FPS,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
        metadata={
            "action_source": "synthetic_deterministic_no_robot_no_goal_write",
            "evidence_level": "offline",
            "source_kind": "synthetic_fixture",
            "generator": ROBOT_FREE_SCHEMA_VERSION,
        },
    )
    for frame_index in range(EPISODE_FRAME_COUNT):
        timestamp_s = frame_index / EPISODE_FPS
        joints = _synthetic_joint_vector(timestamp_s, phase=phase)
        buffer.append(
            SensorimotorFrame(
                timestamp_s=timestamp_s,
                images=_synthetic_images(frame_index, phase=phase),
                joint_position=joints,
                executed_action=joints,
            )
        )
    return buffer.save(directory, stem=stem)


def _synthetic_joint_vector(timestamp_s: float, *, phase: float) -> np.ndarray:
    axes = np.arange(ACTION_DIM, dtype=np.float32)
    signal = np.sin(timestamp_s * 1.7 + phase + axes * 0.23).astype(np.float32)
    joints = signal * np.float32(4.0)
    joints[5] = np.float32(50.0 + 10.0 * np.sin(timestamp_s + phase))
    joints[11] = np.float32(50.0 + 10.0 * np.cos(timestamp_s + phase))
    return joints.astype(np.float32)


def _synthetic_images(frame_index: int, *, phase: float) -> Mapping[str, np.ndarray]:
    row = np.arange(SYNTHETIC_IMAGE_HEIGHT, dtype=np.uint8)[:, None]
    col = np.arange(SYNTHETIC_IMAGE_WIDTH, dtype=np.uint8)[None, :]
    base = (row * 5 + col * 3 + frame_index + int(phase * 100)) % 255
    left = np.stack((base, (base + 37) % 255, (base + 73) % 255), axis=-1)
    right = np.stack(((base + 19) % 255, (base + 61) % 255, (base + 97) % 255), axis=-1)
    return {
        PRIMARY_CAMERA_KEYS[0]: left.astype(np.uint8),
        PRIMARY_CAMERA_KEYS[1]: right.astype(np.uint8),
    }


def _check_real_output_guard(checkpoint_path: Path, *, device: str) -> dict[str, Any]:
    bundle = load_compact_wam_bundle(checkpoint_path, device=device)
    try:
        _require_deployment_checkpoint(bundle.metadata)
    except HardwareCLIError as error:
        return {"real_output_authorized": False, "rejection_reason": str(error)}
    raise RobotFreePipelineError(
        "offline candidate unexpectedly passed real-output guard"
    )


def _require_robot_free_config(config: ProjectConfig) -> None:
    """Require the exact actuated simulation contract used by this fixture."""

    if config.runtime.backend != "mujoco":
        raise RobotFreePipelineError(
            "robot-free pipeline requires runtime.backend='mujoco'"
        )
    if not config.runtime.actuation_enabled:
        raise RobotFreePipelineError(
            "robot-free pipeline requires MuJoCo actuation_enabled=true"
        )
    expected_runtime = {
        "camera_hz": EPISODE_FPS,
        "policy_hz": 10.0,
        "servo_hz": 50.0,
        "action_horizon": 10,
    }
    actual_runtime = {
        "camera_hz": config.runtime.camera_hz,
        "policy_hz": config.runtime.policy_hz,
        "servo_hz": config.runtime.servo_hz,
        "action_horizon": config.runtime.action_horizon,
    }
    if actual_runtime != expected_runtime:
        raise RobotFreePipelineError(
            "robot-free runtime contract mismatch: "
            f"expected {expected_runtime}, got {actual_runtime}"
        )
    resolution = (config.mujoco.camera_height, config.mujoco.camera_width)
    expected_resolution = (SYNTHETIC_IMAGE_HEIGHT, SYNTHETIC_IMAGE_WIDTH)
    if resolution != expected_resolution:
        raise RobotFreePipelineError(
            "robot-free MuJoCo resolution mismatch: "
            f"expected {expected_resolution}, got {resolution}"
        )
    if not config.mujoco.forbid_collisions:
        raise RobotFreePipelineError(
            "robot-free G8 requires mujoco.forbid_collisions=true"
        )


def _require_actuated_g8(report: Mapping[str, Any]) -> None:
    if (
        report.get("result") != "pass"
        or report.get("evidence_level") != "simulation"
        or report.get("policy") != "compact_wam"
    ):
        raise RobotFreePipelineError(
            "robot-free G8 must be a passing compact_wam simulation report"
        )
    rollout = report.get("rollout")
    if not isinstance(rollout, Mapping):
        raise RobotFreePipelineError("MuJoCo G8 report is missing rollout metrics")
    servo_steps = rollout.get("servo_steps")
    sent_actions = rollout.get("sent_actions")
    shadow_steps = rollout.get("shadow_steps")
    if (
        not isinstance(servo_steps, int)
        or isinstance(servo_steps, bool)
        or servo_steps < 1
        or not isinstance(sent_actions, int)
        or isinstance(sent_actions, bool)
        or not isinstance(shadow_steps, int)
        or isinstance(shadow_steps, bool)
        or sent_actions != servo_steps
        or shadow_steps != 0
    ):
        raise RobotFreePipelineError(
            "robot-free G8 must actuate every MuJoCo servo step without shadow steps"
        )


def _artifact_manifest(
    output_root: Path,
    paths: Mapping[str, Path],
) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise RobotFreePipelineError(f"artifact is missing or not a file: {path}")
        manifest[name] = {
            "path": _bundle_relative_path(output_root, resolved),
            "sha256": file_sha256(resolved),
        }
    return manifest


def _bundle_relative_path(output_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(output_root.resolve())
    except ValueError as error:
        raise RobotFreePipelineError(
            f"artifact path escapes the result bundle: {path}"
        ) from error
    return relative.as_posix()


def _require_empty_output_dir(output_root: Path) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise RobotFreePipelineError(
                f"output-dir exists and is not a directory: {output_root}"
            )
        try:
            next(output_root.iterdir())
        except StopIteration:
            return
        raise RobotFreePipelineError(f"output-dir must be empty: {output_root}")


def _write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise RobotFreePipelineError(f"result artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with temp_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != value:
                raise RobotFreePipelineError("result JSON round-trip validation failed")
        try:
            os.link(temp_path, path)
        except FileExistsError as error:
            raise RobotFreePipelineError(
                f"result artifact already exists: {path}"
            ) from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _copy_file_no_overwrite(source: Path, target: Path) -> None:
    if not source.is_file():
        raise RobotFreePipelineError(
            f"config source is missing or not a file: {source}"
        )
    if target.exists():
        raise RobotFreePipelineError(f"config artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = source.read_bytes()
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if temp_path.read_bytes() != source_bytes:
            raise RobotFreePipelineError("config artifact round-trip validation failed")
        try:
            os.link(temp_path, target)
        except FileExistsError as error:
            raise RobotFreePipelineError(
                f"config artifact already exists: {target}"
            ) from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a robot-free synthetic training and MuJoCo G8 session."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--policy-steps", default=1, type=int)
    args = parser.parse_args(argv)

    try:
        result = run_robot_free_demo(
            args.output_dir,
            config_path=args.config,
            device=args.device,
            seed=args.seed,
            policy_steps=args.policy_steps,
        )
    except (
        CheckpointError,
        ConfigError,
        ContractError,
        DatasetArtifactExistsError,
        DatasetError,
        DeploymentCertificationError,
        HardwareCLIError,
        ModelContractError,
        MujocoAdapterError,
        MujocoCLIError,
        PolicyError,
        RobotFreePipelineError,
        RolloutError,
        RuntimeErrorState,
        TrainingDataError,
        TrainingError,
        OSError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ROBOT_FREE_SCHEMA_VERSION",
    "RobotFreePipelineError",
    "run_robot_free_demo",
    "run_robot_free_pipeline",
]
