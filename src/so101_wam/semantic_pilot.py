"""Bounded multi-seed semantic simulation pilot."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from importlib import metadata
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .adapters.mujoco import MujocoAdapterError
from .checkpoint import CheckpointError
from .config import ConfigError, DEFAULT_ROBOT_FREE_CONFIG_PATH, ProjectConfig
from .contracts import ContractError
from .dataset import DatasetArtifactExistsError, DatasetError
from .deployment import canonical_json_sha256, file_sha256
from .model import ModelContractError
from .mujoco_benchmark import MujocoBenchmarkError
from .mujoco_cli import MujocoCLIError
from .mujoco_semantic_suite import MujocoSemanticSuiteError, run_mujoco_semantic_suite
from .policy import PolicyError
from .robot_free_cli import (
    RobotFreePipelineError,
    _copy_file_no_overwrite,
    _require_empty_output_dir,
    _require_robot_free_config,
    _robot_free_training_config,
    _write_json_no_overwrite,
)
from .robot_free_semantic_cli import (
    DEFAULT_CONFIG_PATH,
    EPISODES_PER_TASK,
    POSITION_TOLERANCE_M,
    TARGET_X,
    TARGET_Z,
    TRAIN_TASK,
    TRAIN_TASK_INDEX,
    TRIAL_SEEDS,
    _CaseSpec,
    _suite_id,
    _write_mapping_manifest,
    _write_semantic_episodes,
    _write_semantic_manifests,
    _write_suite_manifest,
)
from .rollout import RolloutError
from .runtime import RuntimeErrorState
from .training import TrainingError, train_offline_candidate
from .training_data import (
    TrainingDataError,
    load_episode_records,
    validate_task_disjoint_split,
)


PILOT_SCHEMA = "so101_wam.semantic_pilot.v1"
MODEL_SEEDS = (3, 7, 11)
DATA_SEED = 7
POLICY_STEPS = 10
TRAIN_STEPS = 10
THREADS = 1
PILOT_ID = "robot-free-semantic-pilot"

CASES = (
    _CaseSpec(
        case_id="block-left",
        task="synthetic-place-block-left",
        task_index=201,
        object_body="task_block",
        initial_z=0.475,
        target_y=-0.04,
        phase=0.7,
    ),
    _CaseSpec(
        case_id="block-right",
        task="synthetic-place-block-right",
        task_index=202,
        object_body="task_block",
        initial_z=0.475,
        target_y=0.04,
        phase=0.9,
    ),
    _CaseSpec(
        case_id="cylinder-left",
        task="synthetic-place-cylinder-left",
        task_index=203,
        object_body="task_cylinder",
        initial_z=0.49,
        target_y=-0.04,
        phase=1.2,
    ),
    _CaseSpec(
        case_id="cylinder-right",
        task="synthetic-place-cylinder-right",
        task_index=204,
        object_body="task_cylinder",
        initial_z=0.49,
        target_y=0.04,
        phase=1.4,
    ),
)


class SemanticPilotError(ValueError):
    """Raised when the bounded semantic pilot cannot publish valid evidence."""


def run_semantic_pilot(
    output_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
    model_seeds: Sequence[int] = MODEL_SEEDS,
    data_seed: int = DATA_SEED,
) -> dict[str, Any]:
    """Run a fixed-input, multi-model-seed semantic simulation pilot."""

    if not _fixed_params(model_seeds=model_seeds, data_seed=data_seed):
        raise SemanticPilotError("semantic pilot uses fixed model and data seeds")
    if device != "cpu":
        raise SemanticPilotError("semantic pilot is fixed to device='cpu'")

    output_root = Path(output_dir)
    _require_empty_output_dir(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with _fixed_runtime():
        prepared = _prep_inputs(
            output_root,
            config_path=Path(config_path),
            data_seed=data_seed,
        )
        slots = _run_slots(
            output_root,
            prepared=prepared,
            model_seeds=tuple(model_seeds),
            data_seed=data_seed,
            device=device,
        )

    report = _pilot_report(
        prepared=prepared,
        slots=slots,
        model_seeds=tuple(model_seeds),
        data_seed=data_seed,
        device=device,
    )
    _write_json_no_overwrite(output_root / "semantic-pilot-report.json", report)
    return report


def _prep_inputs(
    output_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_seed: int = DATA_SEED,
) -> dict[str, Any]:
    shared = output_root / "shared"
    config_copy = shared / "effective_config.toml"
    train_dir = shared / "episodes" / "train"
    validation_dir = shared / "episodes" / "validation"
    suite_path = shared / "semantic-suite.json"
    mapping_path = shared / "semantic-mapping.json"

    source_config = ProjectConfig.load(config_path)
    _require_robot_free_config(source_config)
    _copy_file_no_overwrite(config_path, config_copy)
    config = ProjectConfig.load(config_copy)
    _require_robot_free_config(config)
    if config != source_config:
        raise SemanticPilotError("config changed while pilot inputs were frozen")

    prompts = _write_semantic_episodes(
        train_dir=train_dir,
        validation_dir=validation_dir,
        seed=data_seed,
        cases=CASES,
    )
    semantic_paths = _write_semantic_manifests(
        shared,
        cases=CASES,
        policy_steps=POLICY_STEPS,
        trial_seeds=TRIAL_SEEDS,
    )
    _write_suite_manifest(
        shared,
        suite_path=suite_path,
        semantic_paths=semantic_paths,
        prompts=prompts,
        seed=data_seed,
        cases=CASES,
    )
    _write_mapping_manifest(
        mapping_path,
        semantic_paths=semantic_paths,
        seed=data_seed,
        cases=CASES,
    )

    train_records = load_episode_records([train_dir])
    validation_records = load_episode_records([validation_dir])
    split = validate_task_disjoint_split(train_records, validation_records)
    inputs = {
        "config_path": config_copy,
        "train_dir": train_dir,
        "validation_dir": validation_dir,
        "suite_path": suite_path,
        "mapping_path": mapping_path,
        "config_sha256": file_sha256(config_copy),
        "suite_sha256": file_sha256(suite_path),
        "mapping_sha256": file_sha256(mapping_path),
        "train_digest": split.train_digest,
        "validation_digest": split.validation_digest,
        "input_fingerprints": [
            record.fingerprint
            for record in (*split.train, *split.validation)
        ],
        "input_artifacts": _file_tree_hashes(shared),
        "source_hashes": _source_hashes(),
        "runtime": _runtime(),
    }
    protocol = _protocol(inputs=inputs, data_seed=data_seed)
    inputs["protocol_sha256"] = canonical_json_sha256(protocol)
    _write_json_no_overwrite(shared / "pilot-protocol.json", protocol)
    inputs["shared_hashes"] = _file_tree_hashes(shared)
    return inputs


def _run_slots(
    output_root: Path,
    *,
    prepared: Mapping[str, Any],
    model_seeds: tuple[int, ...],
    data_seed: int,
    device: str,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for index, seed in enumerate(model_seeds):
        if stop_reason is not None:
            item = _slot(seed, index, status="not_attempted", reason=stop_reason)
            _write_slot(output_root, item)
            slots.append(item)
            continue
        try:
            _check_frozen(prepared)
            report = _run_slot(
                output_root,
                prepared=prepared,
                model_seed=seed,
                data_seed=data_seed,
                device=device,
            )
            _check_frozen(prepared)
            item = _slot(seed, index, report=report)
            item["input_fingerprints"] = list(prepared["input_fingerprints"])
            item["train_digest"] = prepared["train_digest"]
            item["validation_digest"] = prepared["validation_digest"]
            _write_slot(output_root, item)
            slots.append(item)
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            item = _slot(seed, index, status="failed", reason=stop_reason)
            item["input_fingerprints"] = list(prepared["input_fingerprints"])
            _write_slot(output_root, item)
            slots.append(item)
        except SemanticPilotError as error:
            stop_reason = str(error)
            item = _slot(seed, index, status="failed", reason=stop_reason)
            item["input_fingerprints"] = list(prepared["input_fingerprints"])
            _write_slot(output_root, item)
            slots.append(item)
        except Exception as error:
            item = _slot(seed, index, status="failed", reason=str(error))
            item["input_fingerprints"] = list(prepared["input_fingerprints"])
            _write_slot(output_root, item)
            slots.append(item)
    return slots


def _run_slot(
    output_root: Path,
    *,
    prepared: Mapping[str, Any],
    model_seed: int,
    data_seed: int,
    device: str,
) -> dict[str, Any]:
    slot_root = output_root / "slots" / f"seed-{model_seed}"
    checkpoint_path = slot_root / "candidate.pt"
    training_path = slot_root / "candidate.training.json"
    artifact_dir = slot_root / "semantic-artifacts"
    report_path = slot_root / "semantic-suite-report.json"

    train_offline_candidate(
        load_episode_records([Path(prepared["train_dir"])]),
        load_episode_records([Path(prepared["validation_dir"])]),
        checkpoint_path=checkpoint_path,
        report_path=training_path,
        checkpoint_id=f"{PILOT_ID}-model-seed-{model_seed}-data-seed-{data_seed}",
        config=_train_config(model_seed),
        device=device,
    )

    config = ProjectConfig.load(Path(prepared["config_path"]))
    report = dict(
        run_mujoco_semantic_suite(
            config,
            suite_path=Path(prepared["suite_path"]),
            mapping_path=Path(prepared["mapping_path"]),
            checkpoint_path=checkpoint_path,
            training_report_path=training_path,
            artifact_dir=artifact_dir,
            report_path=report_path,
            device=device,
        )
    )
    report["suite_report_path"] = report_path.relative_to(output_root).as_posix()
    report["suite_report_sha256"] = file_sha256(report_path)
    return report


def _pilot_report(
    *,
    prepared: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    model_seeds: tuple[int, ...],
    data_seed: int,
    device: str,
) -> dict[str, Any]:
    complete = all(slot.get("status") == "complete" for slot in slots)
    return {
        "schema_version": PILOT_SCHEMA,
        "pilot_id": PILOT_ID,
        "result": "complete" if complete else "incomplete",
        "device": device,
        "model_seeds": list(model_seeds),
        "data_seed": data_seed,
        "rollout_seeds": list(TRIAL_SEEDS),
        "policy_steps": POLICY_STEPS,
        "stage1_steps": TRAIN_STEPS,
        "stage2_steps": TRAIN_STEPS,
        "case_count": len(CASES),
        "episodes_per_task": EPISODES_PER_TASK,
        "train_task": TRAIN_TASK,
        "train_task_index": TRAIN_TASK_INDEX,
        "suite_id": _suite_id(data_seed),
        "input_fingerprints": list(prepared["input_fingerprints"]),
        "input_artifacts": dict(prepared["input_artifacts"]),
        "train_digest": prepared["train_digest"],
        "validation_digest": prepared["validation_digest"],
        "protocol_sha256": prepared["protocol_sha256"],
        "source_hashes": dict(prepared["source_hashes"]),
        "runtime": dict(prepared["runtime"]),
        "slots": list(slots),
        "aggregate": _aggregate(slots) if complete else None,
        "independent_mapping_verified": False,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "limitations": [
            "synthetic labels are not independent semantic demonstrations",
            "bounded 10 step training budgets are not adequate performance budgets",
            "simulation results do not certify real-world transfer",
        ],
    }


def _aggregate(slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    scored = 0
    execution = 0
    successes: dict[str, int] = {}
    weighted_error = 0.0
    macro_errors: list[float] = []
    max_errors: list[float] = []
    failures: dict[str, int] = {}
    for slot in slots:
        seed = str(slot["model_seed"])
        summary = _summary(slot)
        total += int(summary["total_trial_count"])
        scored += int(summary["scored_trial_count"])
        execution += int(summary["execution_failure_count"])
        successes[seed] = int(summary["success_count"])
        if summary["object_position_error_mean_m"] is not None:
            error = float(summary["object_position_error_mean_m"])
            weighted_error += error * int(summary["scored_trial_count"])
            macro_errors.append(error)
        if summary["object_position_error_max_m"] is not None:
            max_errors.append(float(summary["object_position_error_max_m"]))
        for key, value in dict(summary["failure_counts"]).items():
            failures[str(key)] = failures.get(str(key), 0) + int(value)

    return {
        "slot_count": len(slots),
        "total_trial_count": total,
        "scored_trial_count": scored,
        "execution_failure_count": execution,
        "success_by_seed": successes,
        "object_position_error_mean_m": (
            None if scored == 0 else weighted_error / scored
        ),
        "macro_seed_error_mean_m": (
            None if not macro_errors else sum(macro_errors) / len(macro_errors)
        ),
        "object_position_error_max_m": None if not max_errors else max(max_errors),
        "failure_counts": dict(sorted(failures.items())),
    }


def _slot(
    model_seed: int,
    index: int,
    *,
    report: Mapping[str, Any] | None = None,
    status: str = "complete",
    reason: str | None = None,
) -> dict[str, Any]:
    if report is not None:
        _validate_report(report)
    item: dict[str, Any] = {
        "slot_index": index,
        "model_seed": model_seed,
        "status": status,
        "reason": reason,
    }
    if report is not None:
        item.update(
            {
                "suite_result": report.get("result"),
                "case_count": report.get("case_count"),
                "distinct_object_task_count": report.get(
                    "distinct_object_task_count"
                ),
                "distinct_object_physical_profile_count": report.get(
                    "distinct_object_physical_profile_count"
                ),
                "mapping_status_counts": report.get("mapping_status_counts"),
                "independent_mapping_verified": report.get(
                    "independent_mapping_verified"
                ),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "training_report_sha256": report.get("training_report_sha256"),
                "suite_report_path": report.get("suite_report_path"),
                "suite_report_sha256": report.get("suite_report_sha256"),
                "summary": report.get("summary"),
            }
        )
    return item


def _summary(slot: Mapping[str, Any]) -> Mapping[str, Any]:
    value = slot.get("summary")
    if not isinstance(value, Mapping):
        raise SemanticPilotError("complete pilot slot is missing summary")
    return value


def _validate_report(report: Mapping[str, Any]) -> None:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise SemanticPilotError("semantic suite report is missing summary")
    expected_false = (
        "independent_mapping_verified",
        "semantic_heldout_success_claimed",
        "prompt_causality_claimed",
        "real_world_success_claimed",
        "official_zero_wam_claimed",
        "robot_used",
    )
    if report.get("result") != "complete":
        raise SemanticPilotError("semantic suite did not complete")
    if report.get("case_count") != len(CASES):
        raise SemanticPilotError("semantic suite case count mismatch")
    if summary.get("total_trial_count") != len(CASES) * len(TRIAL_SEEDS):
        raise SemanticPilotError("semantic suite trial count mismatch")
    if any(report.get(key) is not False for key in expected_false):
        raise SemanticPilotError("semantic suite made an unsupported claim")


def _protocol(
    *,
    inputs: Mapping[str, Any],
    data_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": PILOT_SCHEMA,
        "pilot_id": PILOT_ID,
        "data_seed": data_seed,
        "model_seeds": list(MODEL_SEEDS),
        "rollout_seeds": list(TRIAL_SEEDS),
        "policy_steps": POLICY_STEPS,
        "stage1_steps": TRAIN_STEPS,
        "stage2_steps": TRAIN_STEPS,
        "case_ids": [case.case_id for case in CASES],
        "dataset_task_indices": [case.task_index for case in CASES],
        "object_bodies": [case.object_body for case in CASES],
        "target_x": TARGET_X,
        "target_z": TARGET_Z,
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "input_fingerprints": list(inputs["input_fingerprints"]),
        "input_artifacts": dict(inputs["input_artifacts"]),
        "train_digest": inputs["train_digest"],
        "validation_digest": inputs["validation_digest"],
        "config_sha256": inputs["config_sha256"],
        "suite_sha256": inputs["suite_sha256"],
        "mapping_sha256": inputs["mapping_sha256"],
        "source_hashes": dict(inputs["source_hashes"]),
        "runtime": dict(inputs["runtime"]),
        "claims": {
            "independent_mapping_verified": False,
            "semantic_heldout_success_claimed": False,
            "prompt_causality_claimed": False,
            "real_world_success_claimed": False,
            "official_zero_wam_claimed": False,
        },
    }


def _check_frozen(prepared: Mapping[str, Any]) -> None:
    try:
        train_records = load_episode_records([Path(prepared["train_dir"])])
        validation_records = load_episode_records([Path(prepared["validation_dir"])])
        split = validate_task_disjoint_split(train_records, validation_records)
        current = {
            "config_sha256": file_sha256(Path(prepared["config_path"])),
            "suite_sha256": file_sha256(Path(prepared["suite_path"])),
            "mapping_sha256": file_sha256(Path(prepared["mapping_path"])),
            "train_digest": split.train_digest,
            "validation_digest": split.validation_digest,
            "input_fingerprints": [
                record.fingerprint
                for record in (*split.train, *split.validation)
            ],
            "shared_hashes": _file_tree_hashes(
                Path(prepared["config_path"]).parent
            ),
            "source_hashes": _source_hashes(),
            "runtime": _runtime(),
        }
        expected = {
            "config_sha256": prepared["config_sha256"],
            "suite_sha256": prepared["suite_sha256"],
            "mapping_sha256": prepared["mapping_sha256"],
            "train_digest": prepared["train_digest"],
            "validation_digest": prepared["validation_digest"],
            "input_fingerprints": list(prepared["input_fingerprints"]),
            "shared_hashes": dict(prepared["shared_hashes"]),
            "source_hashes": dict(prepared["source_hashes"]),
            "runtime": dict(prepared["runtime"]),
        }
    except (KeyError, OSError, TrainingDataError, DatasetError) as error:
        raise SemanticPilotError("pilot frozen input check failed") from error
    if current != expected:
        raise SemanticPilotError("pilot frozen inputs, source, or runtime changed")


def _fixed_params(
    *,
    model_seeds: Sequence[int],
    data_seed: int,
) -> bool:
    if not isinstance(data_seed, int) or isinstance(data_seed, bool):
        return False
    if data_seed != DATA_SEED:
        return False
    if len(model_seeds) != len(MODEL_SEEDS):
        return False
    return all(
        type(seed) is int and seed == expected
        for seed, expected in zip(model_seeds, MODEL_SEEDS, strict=True)
    )


def _train_config(seed: int):
    return replace(
        _robot_free_training_config(seed),
        stage1_steps=TRAIN_STEPS,
        stage2_steps=TRAIN_STEPS,
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    package_root = root.parent.parent
    hashes = {
        path.relative_to(package_root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    assets = root / "assets"
    if assets.is_dir():
        hashes.update(
            {
                path.relative_to(package_root).as_posix(): file_sha256(path)
                for path in sorted(assets.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
            }
        )
    for name in ("pyproject.toml", "uv.lock"):
        path = package_root / name
        if path.is_file():
            hashes[name] = file_sha256(path)
    config = Path(DEFAULT_ROBOT_FREE_CONFIG_PATH)
    if config.is_file():
        hashes[_rel_key(config, root=package_root)] = file_sha256(config)
    return hashes


def _rel_key(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_slot(output_root: Path, slot: Mapping[str, Any]) -> None:
    slot_root = output_root / "slots" / f"seed-{slot['model_seed']}"
    _write_json_no_overwrite(slot_root / "slot-status.json", slot)


def _runtime() -> dict[str, Any]:
    return {
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": _pkg_version("mujoco"),
        },
    }


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@contextmanager
def _fixed_runtime() -> Iterator[None]:
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_random = random.getstate()
    old_numpy = np.random.get_state()
    old_torch = torch.get_rng_state()
    try:
        torch.set_num_threads(THREADS)
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        random.setstate(old_random)
        np.random.set_state(old_numpy)
        torch.set_rng_state(old_torch)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
        torch.set_num_threads(old_threads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded robot-free semantic multi-seed pilot."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    try:
        report = run_semantic_pilot(
            args.output_dir,
            config_path=args.config,
            device=args.device,
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
        SemanticPilotError,
        TrainingDataError,
        TrainingError,
        OSError,
    ) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["result"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SemanticPilotError",
    "main",
    "run_semantic_pilot",
]
