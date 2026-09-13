"""Robot-free offline training plus a real multi-object MuJoCo suite."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import CheckpointError
from .config import ConfigError, DEFAULT_ROBOT_FREE_CONFIG_PATH, ProjectConfig
from .contracts import ContractError
from .dataset import DatasetArtifactExistsError, DatasetError
from .deployment import file_sha256
from .model import ModelContractError
from .mujoco_benchmark import MujocoBenchmarkError
from .mujoco_cli import MujocoCLIError
from .mujoco_semantic_benchmark import (
    MANIFEST_SCHEMA_VERSION,
    TERMINAL_CRITERION,
    load_semantic_manifest,
)
from .mujoco_semantic_suite import (
    SEMANTIC_MAPPING_SCHEMA,
    SEMANTIC_MAPPING_SCOPE,
    SEMANTIC_SUITE_SCHEMA,
    MujocoSemanticSuiteError,
    object_task_signature,
    run_mujoco_semantic_suite,
)
from .policy import PolicyError
from .robot_free_cli import (
    RobotFreePipelineError,
    _bundle_relative_path,
    _copy_file_no_overwrite,
    _make_episode,
    _require_empty_output_dir,
    _require_robot_free_config,
    _robot_free_training_config,
    _write_json_no_overwrite,
)
from .rollout import RolloutError
from .runtime import RuntimeErrorState
from .training import TrainingError, train_offline_candidate
from .training_data import TrainingDataError, load_episode_records


DEFAULT_CONFIG_PATH = DEFAULT_ROBOT_FREE_CONFIG_PATH
SUITE_ID_PREFIX = "robot-free-semantic-seed"
TRAIN_TASK = "synthetic-train-reach"
TRAIN_TASK_INDEX = 101
TRIAL_SEEDS = (7, 13, 29)
INITIAL_X_POSITIONS = (0.32, 0.34, 0.36)
INITIAL_Y_POSITIONS = (-0.02, 0.0, 0.02)
TARGET_X = 0.34
TARGET_Z = 0.58
POSITION_TOLERANCE_M = 0.03
POLICY_STEPS = 1
EPISODES_PER_TASK = 2
EPISODE_PHASE_STRIDE = 0.17
SEED_PHASE_SCALE = 0.001


@dataclass(frozen=True, slots=True)
class _CaseSpec:
    case_id: str
    task: str
    task_index: int
    object_body: str
    initial_z: float
    target_y: float
    phase: float

    @property
    def object_task_id(self) -> str:
        return f"heldout_{self.case_id}"


CASES = (
    _CaseSpec(
        case_id="block-left",
        task="synthetic-place-left",
        task_index=201,
        object_body="task_block",
        initial_z=0.475,
        target_y=-0.04,
        phase=0.7,
    ),
    _CaseSpec(
        case_id="cylinder-right",
        task="synthetic-place-right",
        task_index=202,
        object_body="task_cylinder",
        initial_z=0.49,
        target_y=0.04,
        phase=1.4,
    ),
)


def run_robot_free_semantic_demo(
    output_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
    seed: int = 7,
) -> Mapping[str, Any]:
    """Train a synthetic candidate and run two named-object suite cases."""

    output_root = Path(output_dir)
    _require_empty_output_dir(output_root)

    effective_config_path = output_root / "effective_config.toml"
    train_dir = output_root / "episodes" / "train"
    validation_dir = output_root / "episodes" / "validation"
    checkpoint_path = output_root / "candidate.pt"
    training_report_path = output_root / "candidate.training.json"
    suite_path = output_root / "semantic-suite.json"
    mapping_path = output_root / "semantic-mapping.json"
    artifact_dir = output_root / "semantic-artifacts"
    report_path = output_root / "semantic-suite-report.json"

    source_config = ProjectConfig.load(config_path)
    _require_robot_free_config(source_config)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_file_no_overwrite(Path(config_path), effective_config_path)
    config = ProjectConfig.load(effective_config_path)
    _require_robot_free_config(config)
    if config != source_config:
        raise RobotFreePipelineError(
            "robot-free config changed while the immutable copy was published"
        )

    episodes = _write_semantic_episodes(
        train_dir=train_dir,
        validation_dir=validation_dir,
        seed=seed,
    )
    train_offline_candidate(
        load_episode_records([train_dir]),
        load_episode_records([validation_dir]),
        checkpoint_path=checkpoint_path,
        report_path=training_report_path,
        checkpoint_id=f"robot-free-semantic-candidate-seed-{seed}",
        config=_robot_free_training_config(seed),
        device=device,
    )

    semantic_paths = _write_semantic_manifests(output_root)
    _write_suite_manifest(
        output_root,
        suite_path=suite_path,
        semantic_paths=semantic_paths,
        prompts=episodes,
        seed=seed,
    )
    _write_mapping_manifest(
        mapping_path,
        semantic_paths=semantic_paths,
        seed=seed,
    )

    return run_mujoco_semantic_suite(
        config,
        suite_path=suite_path,
        mapping_path=mapping_path,
        checkpoint_path=checkpoint_path,
        training_report_path=training_report_path,
        artifact_dir=artifact_dir,
        report_path=report_path,
        device=device,
    )


def _write_semantic_episodes(
    *,
    train_dir: Path,
    validation_dir: Path,
    seed: int,
    cases: Sequence[_CaseSpec] = CASES,
) -> dict[str, tuple[Path, Path]]:
    prompts: dict[str, tuple[Path, Path]] = {}
    episode_index = 0

    for repeat_index in range(EPISODES_PER_TASK):
        _make_episode(
            train_dir,
            stem=f"episode_{episode_index:06d}",
            task=TRAIN_TASK,
            task_index=TRAIN_TASK_INDEX,
            episode_index=episode_index,
            phase=(
                repeat_index * EPISODE_PHASE_STRIDE
                + seed * SEED_PHASE_SCALE
            ),
        )
        episode_index += 1

    for case in cases:
        for repeat_index in range(EPISODES_PER_TASK):
            npz_path, manifest_path = _make_episode(
                validation_dir,
                stem=f"episode_{episode_index:06d}",
                task=case.task,
                task_index=case.task_index,
                episode_index=episode_index,
                phase=(
                    case.phase
                    + repeat_index * EPISODE_PHASE_STRIDE
                    + seed * SEED_PHASE_SCALE
                ),
            )
            if repeat_index == 0:
                prompts[case.case_id] = (npz_path, manifest_path)
            episode_index += 1

    return prompts


def _semantic_payload(
    case: _CaseSpec,
    *,
    policy_steps: int = POLICY_STEPS,
    trial_seeds: Sequence[int] = TRIAL_SEEDS,
) -> dict[str, Any]:
    initial_positions = [
        {
            "seed": trial_seed,
            "position": [x, y, case.initial_z],
        }
        for trial_seed, x, y in zip(
            trial_seeds,
            INITIAL_X_POSITIONS,
            INITIAL_Y_POSITIONS,
            strict=True,
        )
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_id": f"{case.case_id}-object-state-v1",
        "dataset_task": {
            "task_index": case.task_index,
            "task": case.task,
        },
        "train_task_ids": [TRAIN_TASK],
        "heldout_tasks": [
            {
                "task_id": case.object_task_id,
                "label": f"held-out {case.case_id}",
                "policy_steps": policy_steps,
                "seeds": list(trial_seeds),
                "object_body": case.object_body,
                "initial_object_positions": initial_positions,
                "target_object_position": [TARGET_X, case.target_y, TARGET_Z],
                "position_tolerance_m": POSITION_TOLERANCE_M,
            }
        ],
    }


def _write_semantic_manifests(
    output_root: Path,
    *,
    cases: Sequence[_CaseSpec] = CASES,
    policy_steps: int = POLICY_STEPS,
    trial_seeds: Sequence[int] = TRIAL_SEEDS,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for case in cases:
        path = output_root / "semantic" / f"{case.case_id}.json"
        _write_json_no_overwrite(
            path,
            _semantic_payload(
                case,
                policy_steps=policy_steps,
                trial_seeds=trial_seeds,
            ),
        )
        paths[case.case_id] = path
    return paths


def _suite_id(seed: int) -> str:
    return f"{SUITE_ID_PREFIX}-{seed}"


def _write_suite_manifest(
    output_root: Path,
    *,
    suite_path: Path,
    semantic_paths: Mapping[str, Path],
    prompts: Mapping[str, tuple[Path, Path]],
    seed: int,
    cases: Sequence[_CaseSpec] = CASES,
) -> None:
    suite_cases = []
    for case in cases:
        prompt_path, prompt_manifest_path = prompts[case.case_id]
        suite_cases.append(
            {
                "case_id": case.case_id,
                "semantic_manifest": _bundle_relative_path(
                    output_root,
                    semantic_paths[case.case_id],
                ),
                "prompt": _bundle_relative_path(output_root, prompt_path),
                "prompt_manifest": _bundle_relative_path(
                    output_root,
                    prompt_manifest_path,
                ),
            }
        )
    _write_json_no_overwrite(
        suite_path,
        {
            "schema_version": SEMANTIC_SUITE_SCHEMA,
            "suite_id": _suite_id(seed),
            "cases": suite_cases,
        },
    )


def _write_mapping_manifest(
    mapping_path: Path,
    *,
    semantic_paths: Mapping[str, Path],
    seed: int,
    cases: Sequence[_CaseSpec] = CASES,
) -> None:
    mappings = []
    for case in cases:
        semantic_path = semantic_paths[case.case_id]
        manifest = load_semantic_manifest(semantic_path)
        task = manifest.heldout_tasks[0]
        mappings.append(
            {
                "case_id": case.case_id,
                "semantic_match": "unverified",
                "reviewer_id": None,
                "dataset_task": {
                    "task_index": case.task_index,
                    "task": case.task,
                },
                "object_task_id": case.object_task_id,
                "object_task_signature_sha256": object_task_signature(task),
                "semantic_manifest_sha256": file_sha256(semantic_path),
                "criterion": TERMINAL_CRITERION,
                "object_body": case.object_body,
                "mapping_basis": (
                    "synthetic_dataset_label_to_declared_object_state_manifest"
                ),
            }
        )
    _write_json_no_overwrite(
        mapping_path,
        {
            "schema_version": SEMANTIC_MAPPING_SCHEMA,
            "suite_id": _suite_id(seed),
            "scope": SEMANTIC_MAPPING_SCOPE,
            "mappings": mappings,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a robot-free candidate and run the block/cylinder semantic suite."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args(argv)

    try:
        report = run_robot_free_semantic_demo(
            args.output_dir,
            config_path=args.config,
            device=args.device,
            seed=args.seed,
        )
    except (
        CheckpointError,
        ConfigError,
        ContractError,
        DatasetArtifactExistsError,
        DatasetError,
        ModelContractError,
        MujocoAdapterError,
        MujocoBenchmarkError,
        MujocoCLIError,
        MujocoSemanticSuiteError,
        PolicyError,
        RobotFreePipelineError,
        RolloutError,
        RuntimeErrorState,
        TrainingDataError,
        TrainingError,
        OSError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RobotFreePipelineError",
    "main",
    "run_robot_free_semantic_demo",
]
