from __future__ import annotations

from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import so101_wam.mujoco_semantic_suite as semantic_suite_module
from so101_wam.checkpoint import (
    compact_wam_architecture,
    save_compact_wam_checkpoint,
)
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, JOINT_KEYS, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.deployment import (
    canonical_json_sha256,
    file_sha256,
    project_config_sha256,
)
from so101_wam.model import CompactWAM
from so101_wam.mujoco_semantic_benchmark import (
    CHECKPOINT_SEMANTIC_GATE,
    CHECKPOINT_SEMANTIC_POLICY,
    CHECKPOINT_SEMANTIC_SCOPE,
    SCHEMA_VERSION as SEMANTIC_REPORT_SCHEMA,
)
from so101_wam.mujoco_benchmark import _wilson_interval
from so101_wam.mujoco_semantic_suite import (
    OBJECT_TASK_SIGNATURE_SCHEMA,
    SEMANTIC_MAPPING_SCHEMA,
    SEMANTIC_MAPPING_SCOPE,
    SEMANTIC_SUITE_SCHEMA,
    SEMANTIC_SUITE_REPORT_SCHEMA,
    MujocoSemanticSuiteError,
    load_semantic_mapping_bytes,
    load_semantic_suite_bytes,
    run_mujoco_semantic_suite,
)
from so101_wam.training import TRAINING_REPORT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
SUITE_ID = "semantic-suite-local-v1"
CASE_A = "block-left"
CASE_B = "cylinder-right"
CHECKPOINT_ID = "semantic-candidate"
TRAINING_EVIDENCE_SHA256 = "1" * 64
SAFE_GRIPPER_POSITION = 1.0
PROMPT_FPS = 30.0
PROMPT_FRAME_COUNT = 91
PROMPT_HEIGHT = 48
PROMPT_WIDTH = 64
OBJECT_PROFILES: dict[str, dict[str, object]] = {
    CASE_A: {
        "schema_version": 1,
        "geom_type": "box",
        "geom_size_m": [0.025, 0.025, 0.025],
        "body_mass_kg": 0.08,
        "body_inertia_kg_m2": [0.000066, 0.000066, 0.000066],
        "geom_friction": [1.0, 0.005, 0.0001],
    },
    CASE_B: {
        "schema_version": 1,
        "geom_type": "cylinder",
        "geom_size_m": [0.02, 0.04, 0.0],
        "body_mass_kg": 0.16,
        "body_inertia_kg_m2": [0.00009, 0.00009, 0.00003],
        "geom_friction": [0.6, 0.003, 0.0001],
    },
}


def _profile_sha256(profile: dict[str, object]) -> str:
    encoded = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


PROFILE_SHA256 = {
    case_id: _profile_sha256(profile)
    for case_id, profile in OBJECT_PROFILES.items()
}


def _semantic_manifest(
    case_id: str,
    task_index: int,
    task: str,
    *,
    object_body: str,
    target_y: float,
    initial_z: float = 0.475,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "benchmark_id": f"{case_id}-object-state-v1",
        "dataset_task": {
            "task_index": task_index,
            "task": task,
        },
        "train_task_ids": [f"train_{case_id}"],
        "heldout_tasks": [
            {
                "task_id": f"heldout_{case_id}",
                "label": f"held-out {case_id}",
                "policy_steps": 1,
                "seeds": [7, 13, 29],
                "object_body": object_body,
                "initial_object_positions": [
                    {"seed": 7, "position": [0.32, -0.02, initial_z]},
                    {"seed": 13, "position": [0.34, 0.0, initial_z]},
                    {"seed": 29, "position": [0.36, 0.02, initial_z]},
                ],
                "target_object_position": [0.34, target_y, 0.58],
                "position_tolerance_m": 0.03,
            }
        ],
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _case_files(
    root: Path,
    *,
    case_id: str,
    task_index: int,
    task: str,
    object_body: str,
    target_y: float,
) -> dict[str, Path]:
    semantic_manifest = _write_json(
        root / "semantic" / f"{case_id}.json",
        _semantic_manifest(
            case_id,
            task_index,
            task,
            object_body=object_body,
            target_y=target_y,
        ),
    )
    prompt = root / "prompts" / f"{case_id}.npz"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_bytes(f"prompt:{case_id}".encode("ascii"))
    prompt_manifest = _write_json(
        root / "prompts" / f"{case_id}.json",
        {
            "schema_version": 1,
            "case_id": case_id,
            "task_index": task_index,
            "task": task,
            "npz_sha256": file_sha256(prompt),
        },
    )
    return {
        "semantic_manifest": semantic_manifest,
        "prompt": prompt,
        "prompt_manifest": prompt_manifest,
    }


def _bundle(tmp_path: Path) -> dict[str, Path]:
    files_a = _case_files(
        tmp_path,
        case_id=CASE_A,
        task_index=201,
        task="synthetic-place-left",
        object_body="task_block",
        target_y=-0.04,
    )
    files_b = _case_files(
        tmp_path,
        case_id=CASE_B,
        task_index=202,
        task="synthetic-place-right",
        object_body="task_cylinder",
        target_y=0.04,
    )
    checkpoint = tmp_path / "semantic-candidate.pt"
    checkpoint.write_bytes(b"semantic-suite-checkpoint")
    training_report = _write_json(
        tmp_path / "semantic-candidate.training.json",
        {
            "schema_version": TRAINING_REPORT_SCHEMA,
            "evidence_level": "offline",
            "artifact_kind": "compact_wam_candidate",
            "result": "pass",
            "trained": False,
            "offline_trained": True,
            "deployment_ready": False,
            "checkpoint_id": CHECKPOINT_ID,
            "training_evidence_sha256": TRAINING_EVIDENCE_SHA256,
            "artifacts": {"checkpoint_sha256": file_sha256(checkpoint)},
        },
    )
    suite = _write_json(tmp_path / "semantic-suite.json", _suite_payload(files_a, files_b, tmp_path))
    mapping = _write_json(
        tmp_path / "semantic-mapping.json",
        _mapping_payload(files_a, files_b),
    )
    return {
        "suite": suite,
        "mapping": mapping,
        "checkpoint": checkpoint,
        "training_report": training_report,
        "case_a_semantic": files_a["semantic_manifest"],
        "case_b_semantic": files_b["semantic_manifest"],
        "case_a_prompt": files_a["prompt"],
        "case_b_prompt": files_b["prompt"],
    }


def _real_prompt(
    root: Path,
    *,
    case_id: str,
    task_index: int,
    task: str,
    pixel_value: int,
) -> tuple[Path, Path]:
    buffer = EpisodeBuffer(
        fps=PROMPT_FPS,
        task=task,
        task_index=task_index,
        episode_index=1,
    )
    joints = np.zeros(ACTION_DIM, dtype=np.float32)
    images = {
        key: np.full(
            (PROMPT_HEIGHT, PROMPT_WIDTH, 3),
            pixel_value,
            dtype=np.uint8,
        )
        for key in PRIMARY_CAMERA_KEYS
    }
    for frame_index in range(PROMPT_FRAME_COUNT):
        buffer.append(
            SensorimotorFrame(
                timestamp_s=frame_index / PROMPT_FPS,
                images=images,
                joint_position=joints,
                executed_action=joints,
            )
        )
    return buffer.save(root / "prompts", stem=case_id)


def _real_case_files(
    root: Path,
    *,
    case_id: str,
    task_index: int,
    task: str,
    object_body: str,
    target_y: float,
    initial_z: float,
    pixel_value: int,
) -> dict[str, Path]:
    semantic_manifest = _write_json(
        root / "semantic" / f"{case_id}.json",
        _semantic_manifest(
            case_id,
            task_index,
            task,
            object_body=object_body,
            target_y=target_y,
            initial_z=initial_z,
        ),
    )
    prompt, prompt_manifest = _real_prompt(
        root,
        case_id=case_id,
        task_index=task_index,
        task=task,
        pixel_value=pixel_value,
    )
    return {
        "semantic_manifest": semantic_manifest,
        "prompt": prompt,
        "prompt_manifest": prompt_manifest,
    }


def _real_bundle(tmp_path: Path) -> dict[str, Path]:
    files_a = _real_case_files(
        tmp_path,
        case_id=CASE_A,
        task_index=201,
        task="synthetic-place-left",
        object_body="task_block",
        target_y=-0.04,
        initial_z=0.475,
        pixel_value=0,
    )
    files_b = _real_case_files(
        tmp_path,
        case_id=CASE_B,
        task_index=202,
        task="synthetic-place-right",
        object_body="task_cylinder",
        target_y=0.04,
        initial_z=0.49,
        pixel_value=1,
    )
    model = CompactWAM(latent_dim=8, transformer_layers=1, transformer_heads=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    # Keep the deterministic policy inside the simulated gripper range.
    axis_mean = torch.zeros(ACTION_DIM)
    for index, key in enumerate(JOINT_KEYS):
        if key.endswith("gripper.pos"):
            axis_mean[index] = SAFE_GRIPPER_POSITION
    model.set_axis_normalization(axis_mean, torch.ones(ACTION_DIM))

    report_core = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "evidence_level": "offline",
        "artifact_kind": "compact_wam_candidate",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": CHECKPOINT_ID,
        "protocol": {
            "split": "task_disjoint",
            "real_output_authorized": False,
        },
        "data": {
            "train_task_count": 1,
            "validation_task_count": 2,
            "train_tasks": [
                {"task_index": 101, "task": "synthetic-train-reach"}
            ],
            "validation_tasks": [
                {"task_index": 201, "task": "synthetic-place-left"},
                {"task_index": 202, "task": "synthetic-place-right"},
            ],
        },
        "model": compact_wam_architecture(model),
    }
    training_evidence = canonical_json_sha256(report_core)
    checkpoint = tmp_path / "semantic-candidate.pt"
    save_compact_wam_checkpoint(
        model,
        checkpoint,
        metadata={
            "artifact_kind": "compact_wam_candidate",
            "evidence_level": "offline",
            "checkpoint_id": CHECKPOINT_ID,
            "trained": False,
            "offline_trained": True,
            "deployment_ready": False,
            "training_evidence_sha256": training_evidence,
        },
    )
    training_report = _write_json(
        tmp_path / "semantic-candidate.training.json",
        {
            **report_core,
            "training_evidence_sha256": training_evidence,
            "artifacts": {"checkpoint_sha256": file_sha256(checkpoint)},
        },
    )
    suite = _write_json(
        tmp_path / "semantic-suite.json",
        _suite_payload(files_a, files_b, tmp_path),
    )
    mapping = _write_json(
        tmp_path / "semantic-mapping.json",
        _mapping_payload(files_a, files_b),
    )
    return {
        "suite": suite,
        "mapping": mapping,
        "checkpoint": checkpoint,
        "training_report": training_report,
        "case_a_semantic": files_a["semantic_manifest"],
        "case_b_semantic": files_b["semantic_manifest"],
        "case_a_prompt": files_a["prompt"],
        "case_b_prompt": files_b["prompt"],
    }


def _suite_payload(
    files_a: dict[str, Path],
    files_b: dict[str, Path],
    root: Path,
) -> dict[str, object]:
    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix()

    return {
        "schema_version": SEMANTIC_SUITE_SCHEMA,
        "suite_id": SUITE_ID,
        "cases": [
            {
                "case_id": CASE_A,
                "semantic_manifest": rel(files_a["semantic_manifest"]),
                "prompt": rel(files_a["prompt"]),
                "prompt_manifest": rel(files_a["prompt_manifest"]),
            },
            {
                "case_id": CASE_B,
                "semantic_manifest": rel(files_b["semantic_manifest"]),
                "prompt": rel(files_b["prompt"]),
                "prompt_manifest": rel(files_b["prompt_manifest"]),
            },
        ],
    }


def _mapping_payload(
    files_a: dict[str, Path],
    files_b: dict[str, Path],
) -> dict[str, object]:
    return {
        "schema_version": SEMANTIC_MAPPING_SCHEMA,
        "suite_id": SUITE_ID,
        "scope": SEMANTIC_MAPPING_SCOPE,
        "mappings": [
            _mapping(
                case_id=CASE_A,
                task_index=201,
                task="synthetic-place-left",
                object_task_id=f"heldout_{CASE_A}",
                semantic_manifest=files_a["semantic_manifest"],
            ),
            _mapping(
                case_id=CASE_B,
                task_index=202,
                task="synthetic-place-right",
                object_task_id=f"heldout_{CASE_B}",
                semantic_manifest=files_b["semantic_manifest"],
            ),
        ],
    }


def _mapping(
    *,
    case_id: str,
    task_index: int,
    task: str,
    object_task_id: str,
    semantic_manifest: Path,
    semantic_match: str = "unverified",
    reviewer_id: str | None = None,
) -> dict[str, object]:
    manifest = json.loads(semantic_manifest.read_text(encoding="utf-8"))
    heldout_tasks = manifest["heldout_tasks"]
    assert isinstance(heldout_tasks, list) and len(heldout_tasks) == 1
    heldout_task = heldout_tasks[0]
    assert isinstance(heldout_task, dict)
    object_body = heldout_task["object_body"]
    return {
        "case_id": case_id,
        "semantic_match": semantic_match,
        "reviewer_id": reviewer_id,
        "dataset_task": {
            "task_index": task_index,
            "task": task,
        },
        "object_task_id": object_task_id,
        "object_task_signature_sha256": _object_task_signature(manifest),
        "semantic_manifest_sha256": file_sha256(semantic_manifest),
        "criterion": "object_body_position",
        "object_body": object_body,
        "mapping_basis": "dataset_task_label_to_object_state_manifest",
    }


def _object_task_signature(manifest: dict[str, object]) -> str:
    heldout_tasks = manifest["heldout_tasks"]
    assert isinstance(heldout_tasks, list) and len(heldout_tasks) == 1
    task = heldout_tasks[0]
    assert isinstance(task, dict)
    initial_positions = task["initial_object_positions"]
    assert isinstance(initial_positions, list)
    canonical = {
        "schema_version": OBJECT_TASK_SIGNATURE_SCHEMA,
        "criterion": "object_body_position",
        "object_body": task["object_body"],
        "initial_object_positions": sorted(
            initial_positions,
            key=lambda item: item["seed"],
        ),
        "target_object_position": task["target_object_position"],
        "position_tolerance_m": task["position_tolerance_m"],
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _load_suite(path: Path):
    return load_semantic_suite_bytes(path.read_bytes(), root=path.parent)


def _load_mapping(path: Path, suite: object):
    return load_semantic_mapping_bytes(
        path.read_bytes(),
        root=path.parent,
        suite=suite,
    )


MODEL_IDENTITY = {
    "schema_version": "so101_wam.mujoco_compiled_model.v1",
    "engine_version": "3.12.0",
    "compiled_model_sha256": "d" * 64,
    "compiled_model_bytes": 128,
}


def _fake_runner(
    calls: list[dict[str, object]],
    *,
    invalid: bool = False,
    mutation: str | None = None,
    execution_failure: bool = False,
    all_execution_failures: bool = False,
):
    def run(
        config: ProjectConfig,
        *,
        manifest_path: str | Path,
        artifact_dir: str | Path,
        checkpoint_path: str | Path,
        training_report_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None = None,
        device: str = "cpu",
    ) -> dict[str, object]:
        case_id = Path(prompt_path).stem
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        heldout_task = manifest["heldout_tasks"][0]
        object_body = heldout_task["object_body"]
        object_profile = OBJECT_PROFILES[case_id]
        object_profile_sha256 = PROFILE_SHA256[case_id]
        model_identity = dict(MODEL_IDENTITY)
        if mutation == "case_model" and case_id == CASE_B:
            model_identity["compiled_model_sha256"] = "e" * 64
        calls.append(
            {
                "manifest_path": Path(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "artifact_dir": Path(artifact_dir),
                "checkpoint_path": Path(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "training_report_path": Path(training_report_path),
                "training_report_sha256": file_sha256(training_report_path),
                "prompt_path": Path(prompt_path),
                "prompt_manifest_path": Path(prompt_manifest_path),
                "device": device,
            }
        )
        if invalid:
            return {"schema_version": 1, "result": "pass"}

        trial_root = Path(artifact_dir)
        trial_root.mkdir(parents=True)
        trials = []
        for seed in (7, 13, 29):
            artifact = trial_root / f"{case_id}-seed-{seed}.json"
            aborted = all_execution_failures or (
                execution_failure and case_id == CASE_B and seed == 7
            )
            identity_source = "post_failure_inspection" if aborted else "rollout_session"
            artifact.write_text(
                json.dumps({
                    "case_id": case_id, "seed": seed,
                    "mujoco_model_identity": model_identity,
                    "model_identity_source": identity_source,
                }),
                encoding="utf-8",
            )
            trials.append(
                {
                    "trial_id": f"{case_id}:seed:{seed}",
                    "task_id": f"heldout_{case_id}",
                    "object_body": object_body,
                    "object_physical_profile_sha256": object_profile_sha256,
                    "mujoco_model_identity": dict(model_identity),
                    "model_identity_source": identity_source,
                    "seed": seed,
                    "status": "execution_failure" if aborted else "scored",
                    "success": case_id == CASE_A and not aborted,
                    "object_position_error_m": (
                        None if aborted else 0.02 if case_id == CASE_A else 0.09
                    ),
                    "failure_reason": (
                        "safety_watchdog"
                        if aborted
                        else None
                        if case_id == CASE_A
                        else "object_body_position_tolerance"
                    ),
                    "failure_evidence": (
                        {
                            "error_type": "SafetyRejectedError",
                            "reasons": ["stale_action"],
                        }
                        if aborted
                        else None
                    ),
                    "artifact": artifact.name,
                    "artifact_sha256": file_sha256(artifact),
                }
            )
        success_count = sum(1 for trial in trials if trial["success"])
        scored_errors = [
            float(trial["object_position_error_m"])
            for trial in trials
            if trial["status"] == "scored"
        ]
        scored_count = len(scored_errors)
        failure_counts = dict(
            sorted(
                Counter(
                    str(trial["failure_reason"])
                    for trial in trials
                    if trial["failure_reason"] is not None
                ).items()
            )
        )
        report: dict[str, object] = {
            "schema_version": SEMANTIC_REPORT_SCHEMA,
            "gate": CHECKPOINT_SEMANTIC_GATE,
            "result": "pass" if success_count == 3 else "fail",
            "mode": "mujoco",
            "evidence_level": "simulation",
            "benchmark_id": f"{case_id}-object-state-v1",
            "config_sha256": project_config_sha256(config),
            "manifest_sha256": file_sha256(manifest_path),
            "dataset_task": f"synthetic-place-{'left' if case_id == CASE_A else 'right'}",
            "dataset_task_index": 201 if case_id == CASE_A else 202,
            "policy": CHECKPOINT_SEMANTIC_POLICY,
            "benchmark_scope": CHECKPOINT_SEMANTIC_SCOPE,
            "criterion": "object_body_position",
            "object_body": object_body,
            "object_physical_profile": object_profile,
            "object_physical_profile_sha256": object_profile_sha256,
            "mujoco_model_identity": dict(model_identity),
            "model_identity_sources": sorted({
                trial["model_identity_source"] for trial in trials
            }),
            "task_disjoint": True,
            "train_task_ids": [f"train_{case_id}"],
            "heldout_task_ids": [f"heldout_{case_id}"],
            "robot_used": False,
            "semantic_mujoco_object_state_evaluated": scored_count > 0,
            "semantic_heldout_success_claimed": False,
            "prompt_causality_claimed": False,
            "real_world_success_claimed": False,
            "official_zero_wam_claimed": False,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_id": CHECKPOINT_ID,
            "training_report_sha256": file_sha256(training_report_path),
            "training_evidence_sha256": TRAINING_EVIDENCE_SHA256,
            "prompt_npz_sha256": file_sha256(prompt_path),
            "prompt_manifest_sha256": file_sha256(prompt_manifest_path),
            "prompt_task": f"synthetic-place-{'left' if case_id == CASE_A else 'right'}",
            "prompt_task_index": 201 if case_id == CASE_A else 202,
            "prompt_task_expectation": "match_manifest",
            "prompt_task_identity_matches_manifest": True,
            "prompt_task_match": "dataset_identity_verified",
            "task_disjoint_basis": "checkpoint_bound_task_inventory",
            "checkpoint_task_split": "training_report_verified",
            "semantic_task_mapping": "manifest_declared",
            "summary": {
                "mujoco_model_identity": dict(model_identity),
                "model_identity_sources": sorted({
                    trial["model_identity_source"] for trial in trials
                }),
                "task_count": 1,
                "trial_count": 3,
                "scored_trial_count": scored_count,
                "execution_failure_count": 3 - scored_count,
                "success_count": success_count,
                "success_rate": success_count / 3,
                "success_rate_95ci": list(_wilson_interval(success_count, 3)),
                "all_trials_successful": success_count == 3,
                "object_position_error_mean_m": (
                    None if not scored_errors else sum(scored_errors) / scored_count
                ),
                "object_position_error_max_m": (
                    None if not scored_errors else max(scored_errors)
                ),
                "failure_counts": failure_counts,
            },
            "tasks": [
                {
                    "task_id": f"heldout_{case_id}",
                    "label": f"held-out {case_id}",
                    "object_body": object_body,
                    "object_physical_profile_sha256": object_profile_sha256,
                    "mujoco_model_identity": dict(model_identity),
                    "model_identity_sources": sorted({
                        trial["model_identity_source"] for trial in trials
                    }),
                    "trial_count": 3,
                    "scored_trial_count": scored_count,
                    "execution_failure_count": 3 - scored_count,
                    "success_count": success_count,
                    "success_rate": success_count / 3,
                    "success_rate_95ci": list(_wilson_interval(success_count, 3)),
                    "object_position_error_mean_m": (
                        None if not scored_errors else sum(scored_errors) / scored_count
                    ),
                    "object_position_error_max_m": (
                        None if not scored_errors else max(scored_errors)
                    ),
                    "failure_counts": failure_counts,
                }
            ],
            "trials": trials,
        }
        if mutation == "prompt_guard":
            report["prompt_task_match"] = "not_verified"
        elif mutation == "profile_hash":
            report["object_physical_profile_sha256"] = "0" * 64
        elif mutation == "task_summary":
            report["tasks"] = []
        elif mutation == "summary_ci":
            summary = report["summary"]
            assert isinstance(summary, dict)
            summary["success_rate_95ci"] = [0.0, 0.0]
        elif mutation == "failure_taxonomy" and case_id == CASE_B:
            trials[0]["failure_reason"] = "safety_joint_limit"
        elif mutation == "model_missing":
            report.pop("mujoco_model_identity")
        elif mutation == "trial_model":
            trials[0]["mujoco_model_identity"] = {
                **model_identity, "compiled_model_sha256": "f" * 64,
            }
        elif mutation == "artifact_model":
            artifact = trial_root / trials[0]["artifact"]
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["mujoco_model_identity"]["compiled_model_sha256"] = "f" * 64
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            trials[0]["artifact_sha256"] = file_sha256(artifact)
        elif mutation == "false_source":
            trials[0]["model_identity_source"] = "post_failure_inspection"
        elif mutation == "source_type":
            trials[0]["model_identity_source"] = ["rollout_session"]
        elif mutation == "summary_model":
            report["summary"]["mujoco_model_identity"]["compiled_model_sha256"] = "f" * 64
        elif mutation == "summary_sources":
            report["summary"]["model_identity_sources"] = ["post_failure_inspection"]
        return report

    return run


def test_semantic_suite_manifest_loads_two_bundle_safe_cases(
    tmp_path: Path,
) -> None:
    paths = _bundle(tmp_path)

    suite = _load_suite(paths["suite"])

    assert suite.suite_id == SUITE_ID
    assert [case.case_id for case in suite.cases] == [CASE_A, CASE_B]
    assert suite.cases[0].semantic_manifest_path == paths["case_a_semantic"]
    assert suite.cases[1].semantic_manifest_path == paths["case_b_semantic"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("one_case", "at least 2 cases"),
        ("duplicate_case", "case_id"),
        ("unknown_field", "unknown fields"),
        ("unsafe_path", "bundle-relative"),
    ),
)
def test_semantic_suite_manifest_rejects_invalid_case_contracts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _bundle(tmp_path)
    payload = json.loads(paths["suite"].read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert isinstance(cases, list)
    if mutation == "one_case":
        del cases[1]
    elif mutation == "duplicate_case":
        cases[1]["case_id"] = CASE_A
    elif mutation == "unknown_field":
        cases[0]["post_hoc_label"] = "extra"
    else:
        cases[0]["semantic_manifest"] = "../outside.json"

    with pytest.raises(MujocoSemanticSuiteError, match=message):
        load_semantic_suite_bytes(json.dumps(payload).encode("utf-8"), root=tmp_path)


def test_semantic_suite_manifest_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    payload = paths["suite"].read_text(encoding="utf-8").replace(
        '"suite_id": "semantic-suite-local-v1",',
        '"suite_id": "mutated", "suite_id": "semantic-suite-local-v1",',
    )

    with pytest.raises(MujocoSemanticSuiteError, match="duplicate field"):
        load_semantic_suite_bytes(payload.encode("utf-8"), root=tmp_path)


def test_semantic_mapping_attests_one_to_one_case_coverage(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    suite = _load_suite(paths["suite"])

    mapping = _load_mapping(paths["mapping"], suite)

    assert mapping.suite_id == SUITE_ID
    assert mapping.scope == SEMANTIC_MAPPING_SCOPE
    assert [item.case_id for item in mapping.mappings] == [CASE_A, CASE_B]
    assert [item.semantic_match for item in mapping.mappings] == [
        "unverified",
        "unverified",
    ]
    assert all(item.reviewer_id is None for item in mapping.mappings)
    assert len({item.object_task_signature_sha256 for item in mapping.mappings}) == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("suite", "suite_id"),
        ("scope", "scope"),
        ("coverage", "1:1 case coverage"),
        ("duplicate_dataset", "dataset identities"),
        ("duplicate_object", "object_task_id"),
        ("object_manifest_mismatch", "object_task_id"),
        ("object_body_mismatch", "object_body"),
        ("stale_hash", "semantic_manifest_sha256"),
        ("stale_signature", "object_task_signature_sha256"),
        ("review_missing", "reviewer_id"),
        ("unverified_reviewer", "reviewer_id"),
        ("unknown_field", "unknown fields"),
    ),
)
def test_semantic_mapping_rejects_invalid_attestations(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _bundle(tmp_path)
    suite = _load_suite(paths["suite"])
    payload = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    mappings = payload["mappings"]
    assert isinstance(mappings, list)
    if mutation == "suite":
        payload["suite_id"] = "different-suite"
    elif mutation == "scope":
        payload["scope"] = "post_hoc_mapping"
    elif mutation == "coverage":
        del mappings[1]
    elif mutation == "duplicate_dataset":
        mappings[1]["dataset_task"] = mappings[0]["dataset_task"]
    elif mutation == "duplicate_object":
        mappings[1]["object_task_id"] = mappings[0]["object_task_id"]
    elif mutation == "object_manifest_mismatch":
        mappings[0]["object_task_id"] = "different-heldout-task"
    elif mutation == "object_body_mismatch":
        mappings[0]["object_body"] = "task_cylinder"
    elif mutation == "stale_hash":
        mappings[0]["semantic_manifest_sha256"] = "0" * 64
    elif mutation == "stale_signature":
        mappings[0]["object_task_signature_sha256"] = "0" * 64
    elif mutation == "review_missing":
        mappings[0]["semantic_match"] = "human_reviewed"
    elif mutation == "unverified_reviewer":
        mappings[0]["reviewer_id"] = "reviewer-a"
    else:
        mappings[0]["notes"] = "extra"

    with pytest.raises(MujocoSemanticSuiteError, match=message):
        load_semantic_mapping_bytes(
            json.dumps(payload).encode("utf-8"),
            root=tmp_path,
            suite=suite,
        )


def test_semantic_mapping_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    suite = _load_suite(paths["suite"])
    payload = paths["mapping"].read_text(encoding="utf-8").replace(
        '"scope": "local_dataset_object_mapping_attestation_only",',
        '"scope": "mutated", "scope": "local_dataset_object_mapping_attestation_only",',
    )

    with pytest.raises(MujocoSemanticSuiteError, match="duplicate field"):
        load_semantic_mapping_bytes(
            payload.encode("utf-8"),
            root=tmp_path,
            suite=suite,
        )


@pytest.mark.parametrize("shared_field", ("task_index", "task"))
def test_semantic_mapping_requires_bijective_dataset_identities(
    tmp_path: Path,
    shared_field: str,
) -> None:
    paths = _bundle(tmp_path)
    manifest = json.loads(paths["case_b_semantic"].read_text(encoding="utf-8"))
    mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    manifest["dataset_task"][shared_field] = mapping["mappings"][0]["dataset_task"][
        shared_field
    ]
    paths["case_b_semantic"].write_text(json.dumps(manifest), encoding="utf-8")
    mapping["mappings"][1]["dataset_task"] = manifest["dataset_task"]
    mapping["mappings"][1]["semantic_manifest_sha256"] = file_sha256(
        paths["case_b_semantic"]
    )
    paths["mapping"].write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(MujocoSemanticSuiteError, match="dataset identities"):
        _load_mapping(paths["mapping"], _load_suite(paths["suite"]))


def test_semantic_mapping_rejects_duplicate_object_task_geometry(
    tmp_path: Path,
) -> None:
    paths = _bundle(tmp_path)
    first_manifest = json.loads(
        paths["case_a_semantic"].read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        paths["case_b_semantic"].read_text(encoding="utf-8")
    )
    second_task = second_manifest["heldout_tasks"][0]
    first_task = first_manifest["heldout_tasks"][0]
    for field in (
        "object_body",
        "initial_object_positions",
        "target_object_position",
        "position_tolerance_m",
    ):
        second_task[field] = first_task[field]
    second_task["initial_object_positions"] = list(
        reversed(second_task["initial_object_positions"])
    )
    paths["case_b_semantic"].write_text(
        json.dumps(second_manifest),
        encoding="utf-8",
    )

    mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    mapping["mappings"][1]["object_body"] = second_task["object_body"]
    mapping["mappings"][1]["semantic_manifest_sha256"] = file_sha256(
        paths["case_b_semantic"]
    )
    mapping["mappings"][1]["object_task_signature_sha256"] = (
        _object_task_signature(second_manifest)
    )
    paths["mapping"].write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(MujocoSemanticSuiteError, match="object task signatures"):
        _load_mapping(paths["mapping"], _load_suite(paths["suite"]))


def test_mujoco_semantic_suite_rejects_prompt_reuse_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    paths["case_b_prompt"].write_bytes(paths["case_a_prompt"].read_bytes())
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls),
    )

    with pytest.raises(MujocoSemanticSuiteError, match="prompt NPZ bytes"):
        run_mujoco_semantic_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            suite_path=paths["suite"],
            mapping_path=paths["mapping"],
            checkpoint_path=paths["checkpoint"],
            training_report_path=paths["training_report"],
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "semantic-suite-report.json",
        )

    assert calls == []
    assert not (tmp_path / "artifacts").exists()


def test_mujoco_semantic_suite_runs_each_case_with_shared_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls),
    )

    report = run_mujoco_semantic_suite(
        config,
        suite_path=paths["suite"],
        mapping_path=paths["mapping"],
        checkpoint_path=paths["checkpoint"],
        training_report_path=paths["training_report"],
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "semantic-suite-report.json",
        device="cpu",
    )

    assert [call["manifest_sha256"] for call in calls] == [
        file_sha256(paths["case_a_semantic"]),
        file_sha256(paths["case_b_semantic"]),
    ]
    assert {call["checkpoint_sha256"] for call in calls} == {
        file_sha256(paths["checkpoint"])
    }
    assert {call["training_report_sha256"] for call in calls} == {
        file_sha256(paths["training_report"])
    }
    assert report["result"] == "complete"
    assert report["schema_version"] == SEMANTIC_SUITE_REPORT_SCHEMA
    assert report["suite_id"] == SUITE_ID
    assert report["case_count"] == 2
    assert report["distinct_object_task_count"] == 2
    assert report["distinct_object_body_count"] == 2
    assert report["distinct_object_physical_profile_count"] == 2
    assert report["physical_object_diversity_observed"] is True
    assert report["summary"]["total_trial_count"] == 6
    assert report["summary"]["success_count"] == 3
    assert report["summary"]["failure_counts"] == {
        "object_body_position_tolerance": 3
    }
    assert [case["case_id"] for case in report["cases"]] == [CASE_A, CASE_B]
    assert [case["result"] for case in report["cases"]] == ["pass", "fail"]
    assert len(
        {case["object_task_signature_sha256"] for case in report["cases"]}
    ) == 2
    assert [case["object_body"] for case in report["cases"]] == [
        "task_block",
        "task_cylinder",
    ]
    assert len(
        {case["object_physical_profile_sha256"] for case in report["cases"]}
    ) == 2
    assert report["suite_sha256"] == file_sha256(paths["suite"])
    assert report["mapping_sha256"] == file_sha256(paths["mapping"])
    assert report["checkpoint_sha256"] == file_sha256(paths["checkpoint"])
    assert report["training_report_sha256"] == file_sha256(paths["training_report"])
    assert report["mujoco_config_sha256"] == project_config_sha256(config)
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["real_world_success_claimed"] is False
    assert report["official_zero_wam_claimed"] is False
    assert report["independent_mapping_verified"] is False


def test_mujoco_semantic_suite_runs_real_named_object_children(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    paths = _real_bundle(tmp_path)
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    config = replace(
        config,
        mujoco=replace(
            config.mujoco,
            camera_width=PROMPT_WIDTH,
            camera_height=PROMPT_HEIGHT,
        ),
    )
    artifact_dir = tmp_path / "real-artifacts"

    report = run_mujoco_semantic_suite(
        config,
        suite_path=paths["suite"],
        mapping_path=paths["mapping"],
        checkpoint_path=paths["checkpoint"],
        training_report_path=paths["training_report"],
        artifact_dir=artifact_dir,
        report_path=tmp_path / "real-semantic-suite-report.json",
    )

    assert report["result"] == "complete"
    assert report["case_count"] == 2
    assert report["distinct_object_task_count"] == 2
    assert report["distinct_object_body_count"] == 2
    assert report["distinct_object_physical_profile_count"] == 2
    assert report["physical_object_diversity_observed"] is True
    assert report["mapping_status_counts"] == {"unverified": 2}
    assert report["independent_mapping_verified"] is False
    assert report["summary"]["total_trial_count"] == 6
    assert report["summary"]["scored_trial_count"] == 6
    assert report["summary"]["execution_failure_count"] == 0
    assert report["summary"]["success_count"] == 0
    assert [case["result"] for case in report["cases"]] == ["fail", "fail"]
    assert {
        case["object_body"]: case["object_physical_profile"]["geom"]["type"]
        for case in report["cases"]
    } == {
        "task_block": "box",
        "task_cylinder": "cylinder",
    }

    for case in report["cases"]:
        assert case["object_physical_profile_sha256"] == canonical_json_sha256(
            case["object_physical_profile"]
        )
        nested_path = artifact_dir / case["benchmark_report"]
        assert file_sha256(nested_path) == case["benchmark_report_sha256"]
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        assert nested["checkpoint_task_split"] == "training_report_verified"
        assert nested["prompt_task_match"] == "dataset_identity_verified"
        assert nested["summary"]["trial_count"] == 3
        assert {
            trial["object_physical_profile_sha256"]
            for trial in nested["trials"]
        } == {case["object_physical_profile_sha256"]}


def test_mujoco_semantic_suite_aggregates_execution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls, execution_failure=True),
    )

    report = run_mujoco_semantic_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        suite_path=paths["suite"],
        mapping_path=paths["mapping"],
        checkpoint_path=paths["checkpoint"],
        training_report_path=paths["training_report"],
        artifact_dir=tmp_path / "recoverable-artifacts",
        report_path=tmp_path / "recoverable-report.json",
    )

    assert len(calls) == 2
    assert report["result"] == "complete"
    assert report["summary"]["total_trial_count"] == 6
    assert report["summary"]["scored_trial_count"] == 5
    assert report["summary"]["execution_failure_count"] == 1
    assert report["summary"]["success_count"] == 3
    assert report["summary"]["object_position_error_mean_m"] == pytest.approx(0.048)
    assert report["summary"]["failure_counts"] == {
        "object_body_position_tolerance": 2,
        "safety_watchdog": 1,
    }


def test_mujoco_semantic_suite_handles_no_scored_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls, all_execution_failures=True),
    )

    report = run_mujoco_semantic_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        suite_path=paths["suite"],
        mapping_path=paths["mapping"],
        checkpoint_path=paths["checkpoint"],
        training_report_path=paths["training_report"],
        artifact_dir=tmp_path / "unscored-artifacts",
        report_path=tmp_path / "unscored-report.json",
    )

    assert len(calls) == 2
    assert report["summary"]["scored_trial_count"] == 0
    assert report["summary"]["execution_failure_count"] == 6
    assert report["summary"]["success_count"] == 0
    assert report["summary"]["object_position_error_mean_m"] is None
    assert report["summary"]["object_position_error_max_m"] is None
    assert report["summary"]["failure_counts"] == {"safety_watchdog": 6}
    assert report["semantic_mujoco_object_state_evaluated"] is False
    assert report["mujoco_model_identity"] == MODEL_IDENTITY
    assert report["model_identity_sources"] == ["post_failure_inspection"]


def test_mujoco_semantic_suite_rejects_tampered_failure_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(
            calls,
            execution_failure=True,
            mutation="failure_taxonomy",
        ),
    )

    with pytest.raises(MujocoSemanticSuiteError, match="failure"):
        run_mujoco_semantic_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            suite_path=paths["suite"],
            mapping_path=paths["mapping"],
            checkpoint_path=paths["checkpoint"],
            training_report_path=paths["training_report"],
            artifact_dir=tmp_path / "tampered-artifacts",
            report_path=tmp_path / "tampered-report.json",
        )
def test_mujoco_semantic_suite_runs_from_frozen_case_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    checkpoint_hash = file_sha256(paths["checkpoint"])
    training_hash = file_sha256(paths["training_report"])
    semantic_hash = file_sha256(paths["case_b_semantic"])
    calls: list[dict[str, object]] = []
    fake = _fake_runner(calls)

    def mutate_sources_after_first_case(*args: object, **kwargs: object):
        report = fake(*args, **kwargs)
        if len(calls) == 1:
            paths["checkpoint"].write_bytes(b"mutated checkpoint")
            paths["training_report"].write_text("{}", encoding="utf-8")
            paths["case_b_semantic"].write_text("{}", encoding="utf-8")
        return report

    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        mutate_sources_after_first_case,
    )

    report = run_mujoco_semantic_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        suite_path=paths["suite"],
        mapping_path=paths["mapping"],
        checkpoint_path=paths["checkpoint"],
        training_report_path=paths["training_report"],
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "semantic-suite-report.json",
    )

    assert report["result"] == "complete"
    assert report["checkpoint_sha256"] == checkpoint_hash
    assert report["training_report_sha256"] == training_hash
    assert report["cases"][1]["semantic_manifest_sha256"] == semantic_hash
    assert all(Path(call["checkpoint_path"]) != paths["checkpoint"] for call in calls)


def test_mujoco_semantic_suite_rejects_stale_mapping_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    payload = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    payload["mappings"][0]["semantic_manifest_sha256"] = "0" * 64
    paths["mapping"].write_text(json.dumps(payload), encoding="utf-8")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls),
    )

    with pytest.raises(MujocoSemanticSuiteError, match="semantic_manifest_sha256"):
        run_mujoco_semantic_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            suite_path=paths["suite"],
            mapping_path=paths["mapping"],
            checkpoint_path=paths["checkpoint"],
            training_report_path=paths["training_report"],
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "semantic-suite-report.json",
        )

    assert calls == []
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "semantic-suite-report.json").exists()


def test_mujoco_semantic_suite_uses_frozen_suite_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    original_suite = paths["suite"].read_bytes()
    real_loader = semantic_suite_module.load_semantic_suite_bytes

    def mutate_then_load(source: bytes, *, root: str | Path) -> object:
        payload = json.loads(paths["suite"].read_text(encoding="utf-8"))
        payload["cases"][0]["case_id"] = "mutated-case"
        paths["suite"].write_text(json.dumps(payload), encoding="utf-8")
        return real_loader(source, root=root)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "load_semantic_suite_bytes",
        mutate_then_load,
    )
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls),
    )

    report = run_mujoco_semantic_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        suite_path=paths["suite"],
        mapping_path=paths["mapping"],
        checkpoint_path=paths["checkpoint"],
        training_report_path=paths["training_report"],
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "semantic-suite-report.json",
    )

    assert [case["case_id"] for case in report["cases"]] == [CASE_A, CASE_B]
    assert report["suite_sha256"] == sha256(original_suite).hexdigest()


def test_mujoco_semantic_suite_rejects_invalid_nested_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls, invalid=True),
    )

    with pytest.raises(MujocoSemanticSuiteError, match="nested"):
        run_mujoco_semantic_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            suite_path=paths["suite"],
            mapping_path=paths["mapping"],
            checkpoint_path=paths["checkpoint"],
            training_report_path=paths["training_report"],
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "semantic-suite-report.json",
        )

    assert len(calls) == 1
    assert not (tmp_path / "semantic-suite-report.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("prompt_guard", "prompt_task_match"),
        ("profile_hash", "physical profile SHA-256"),
        ("task_summary", "task summary"),
        ("summary_ci", "success_rate_95ci"),
        ("model_missing", "model identity"),
        ("trial_model", "model identity"),
        ("artifact_model", "model identity"),
        ("case_model", "model identit"),
        ("false_source", "identity source"),
        ("source_type", "identity source"),
        ("summary_model", "model identity"),
        ("summary_sources", "identity sources"),
    ),
)
def test_mujoco_semantic_suite_revalidates_nested_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    paths = _bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls, mutation=mutation),
    )

    with pytest.raises(MujocoSemanticSuiteError, match=message):
        run_mujoco_semantic_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            suite_path=paths["suite"],
            mapping_path=paths["mapping"],
            checkpoint_path=paths["checkpoint"],
            training_report_path=paths["training_report"],
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "semantic-suite-report.json",
        )


def test_mujoco_semantic_suite_rejects_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle(tmp_path)
    report_path = tmp_path / "semantic-suite-report.json"
    report_path.write_text("existing", encoding="utf-8")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_suite_module,
        "run_checkpoint_benchmark",
        _fake_runner(calls),
    )

    with pytest.raises(MujocoSemanticSuiteError, match="already exists"):
        run_mujoco_semantic_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            suite_path=paths["suite"],
            mapping_path=paths["mapping"],
            checkpoint_path=paths["checkpoint"],
            training_report_path=paths["training_report"],
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert calls == []
    assert report_path.read_text(encoding="utf-8") == "existing"
