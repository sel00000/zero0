from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import so101_wam.mujoco_semantic_prompt_controls as semantic_controls_module
import so101_wam.prompt_controls as prompt_controls_module
import so101_wam.semantic_prompt_directionality as semantic_direction_module
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer, load_episode
from so101_wam.deployment import file_sha256, project_config_sha256
from so101_wam.mujoco_semantic_benchmark import (
    CHECKPOINT_SEMANTIC_GATE,
    CHECKPOINT_SEMANTIC_POLICY,
    CHECKPOINT_SEMANTIC_SCOPE,
    ModelIdentitySource,
    SCHEMA_VERSION as SEMANTIC_REPORT_SCHEMA,
    PromptTaskExpectation,
)
from so101_wam.mujoco_semantic_prompt_controls import (
    MUJOCO_SEMANTIC_PROMPT_CONTROL_GATE,
    MUJOCO_SEMANTIC_PROMPT_CONTROL_SCOPE,
    run_mujoco_semantic_prompt_control_suite,
)
from so101_wam.prompt_controls import PromptCondition, PromptControlError
from so101_wam.semantic_prompt_directionality import (
    SEMANTIC_DIRECTION_CLAIM_SCOPE,
    SEMANTIC_DIRECTION_PLAN_SCHEMA,
)
from so101_wam.training import TRAINING_REPORT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = tuple(condition.value for condition in PromptCondition)
DATASET_TASK = "synthetic-validation-place"
DATASET_TASK_INDEX = 201
WRONG_TASK = "synthetic-control-reach"
WRONG_TASK_INDEX = 999
OBJECT_PROFILE = {
    "schema_version": 1,
    "geom_type": "box",
    "geom_size_m": [0.025, 0.025, 0.025],
    "body_mass_kg": 0.08,
    "body_inertia_kg_m2": [0.000066, 0.000066, 0.000066],
    "geom_friction": [1.0, 0.005, 0.0001],
}
OBJECT_PROFILE_SHA256 = sha256(
    json.dumps(
        OBJECT_PROFILE,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
MODEL_IDENTITY = {
    "schema_version": "so101_wam.mujoco_compiled_model.v1",
    "engine_version": "3.12.0",
    "compiled_model_sha256": "d" * 64,
    "compiled_model_bytes": 128,
}
DRIFT_MODEL_IDENTITY = {
    **MODEL_IDENTITY,
    "compiled_model_sha256": "e" * 64,
}


def _episode(
    root: Path,
    *,
    stem: str,
    task: str,
    task_index: int,
    episode_index: int,
    offset: int,
) -> Path:
    buffer = EpisodeBuffer(
        fps=30.0,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
    )
    for frame_index in range(91):
        value = float(offset + frame_index % 3)
        joints = np.full(ACTION_DIM, value, dtype=np.float32)
        joints[5] = 50.0 + value
        joints[11] = 50.0 - value
        images = {
            PRIMARY_CAMERA_KEYS[0]: np.full(
                (8, 8, 3),
                offset + frame_index,
                dtype=np.uint8,
            ),
            PRIMARY_CAMERA_KEYS[1]: np.full(
                (8, 8, 3),
                offset + frame_index + 1,
                dtype=np.uint8,
            ),
        }
        buffer.append(
            SensorimotorFrame(
                timestamp_s=frame_index / 30.0,
                images=images,
                joint_position=joints,
                executed_action=joints,
            )
        )
    return buffer.save(root, stem=stem)[0]


def _prompt_control_bundle(tmp_path: Path) -> Path:
    episodes = tmp_path / "episodes"
    paths = {
        "live": _episode(
            episodes,
            stem="live",
            task=DATASET_TASK,
            task_index=DATASET_TASK_INDEX,
            episode_index=1,
            offset=1,
        ),
        "matched": _episode(
            episodes,
            stem="matched",
            task=DATASET_TASK,
            task_index=DATASET_TASK_INDEX,
            episode_index=2,
            offset=4,
        ),
        "alternate": _episode(
            episodes,
            stem="alternate",
            task=DATASET_TASK,
            task_index=DATASET_TASK_INDEX,
            episode_index=3,
            offset=8,
        ),
        "wrong": _episode(
            episodes,
            stem="wrong",
            task=WRONG_TASK,
            task_index=WRONG_TASK_INDEX,
            episode_index=4,
            offset=12,
        ),
    }
    config = tmp_path / "source.toml"
    config.write_bytes((ROOT / "configs/fake.toml").read_bytes())
    checkpoint = tmp_path / "semantic-candidate.pt"
    checkpoint.write_bytes(b"semantic-checkpoint-for-prompt-controls")
    manifest = tmp_path / "prompt-controls.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "semantic-prompt-control-test",
                "seed": 17,
                "config": config.name,
                "checkpoint": checkpoint.name,
                "live_episode": paths["live"].relative_to(tmp_path).as_posix(),
                "matched_prompt": paths["matched"].relative_to(tmp_path).as_posix(),
                "same_task_alternate_prompt": paths["alternate"]
                .relative_to(tmp_path)
                .as_posix(),
                "wrong_task_prompt": paths["wrong"].relative_to(tmp_path).as_posix(),
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _semantic_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "semantic-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "benchmark_id": "local-object-state-v1",
                "dataset_task": {
                    "task_index": DATASET_TASK_INDEX,
                    "task": DATASET_TASK,
                },
                "train_task_ids": ["train_block_home"],
                "heldout_tasks": [
                    {
                        "task_id": "heldout_block_lift",
                        "label": "held-out block lift",
                        "policy_steps": 1,
                        "seeds": [7, 13, 29],
                        "object_body": "task_block",
                        "initial_object_positions": [
                            {"seed": 7, "position": [0.32, -0.02, 0.475]},
                            {"seed": 13, "position": [0.34, 0.0, 0.475]},
                            {"seed": 29, "position": [0.36, 0.02, 0.475]},
                        ],
                        "target_object_position": [0.34, 0.0, 0.58],
                        "position_tolerance_m": 0.03,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _training_report(tmp_path: Path, checkpoint: Path) -> Path:
    report = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "evidence_level": "offline",
        "artifact_kind": "compact_wam_candidate",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": "semantic-candidate",
        "training_evidence_sha256": "1" * 64,
        "artifacts": {"checkpoint_sha256": file_sha256(checkpoint)},
    }
    path = tmp_path / "semantic-candidate.training.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _direction_plan(
    tmp_path: Path,
    *,
    prompt_manifest: Path,
    semantic_manifest: Path,
    training_report: Path,
    config: ProjectConfig,
    mutation: tuple[str, object] | None = None,
) -> Path:
    checkpoint = tmp_path / "semantic-candidate.pt"
    payload: dict[str, object] = {
        "schema_version": SEMANTIC_DIRECTION_PLAN_SCHEMA,
        "suite_id": "semantic-prompt-control-test",
        "hypothesis_id": "object-state-direction-v1",
        "claim_scope": SEMANTIC_DIRECTION_CLAIM_SCOPE,
        "prompt_control_manifest_sha256": file_sha256(prompt_manifest),
        "semantic_manifest_sha256": file_sha256(semantic_manifest),
        "training_report_sha256": file_sha256(training_report),
        "checkpoint_sha256": file_sha256(checkpoint),
        "mujoco_config_sha256": project_config_sha256(config),
        "minimum_trials_per_condition": 3,
        "primary_endpoint": "success_rate",
        "secondary_endpoint": "object_position_error_mean_m",
        "same_task_noninferiority_margin": 0.34,
        "negative_success_margin": 0.34,
        "negative_object_position_error_margin_m": 0.05,
        "negative_conditions": ["wrong_task", "null", "counterfactual"],
        "descriptive_conditions": [
            "temporal_shuffle",
            "image_frame_shuffle",
        ],
    }
    if mutation is not None:
        payload[mutation[0]] = mutation[1]
    path = tmp_path / "semantic-direction-plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_semantic_benchmark(
    calls: list[dict[str, object]],
    *,
    corrupt_training_hash: bool = False,
    condition_metrics: dict[str, tuple[int, float]] | None = None,
    execution_failure_condition: str | None = None,
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
        prompt_task_expectation: PromptTaskExpectation = (
            PromptTaskExpectation.MATCH_MANIFEST
        ),
    ) -> dict[str, object]:
        prompt_target = Path(prompt_path)
        prompt_manifest = Path(
            prompt_manifest_path or prompt_target.with_suffix(".json")
        )
        training_report = Path(training_report_path)
        condition = prompt_target.stem
        success_count, condition_error = (condition_metrics or {}).get(
            condition,
            (3, 0.02),
        )
        aborted = execution_failure_condition in {"*", condition}
        if aborted:
            success_count = 0
        prompt = load_episode(prompt_target, prompt_manifest)
        trial_root = Path(artifact_dir)
        trial_root.mkdir(parents=True)
        trials: list[dict[str, object]] = []
        errors: list[float] = []
        for trial_index, seed in enumerate((7, 13, 29)):
            error = None if aborted else condition_error
            success = trial_index < success_count
            artifact = trial_root / f"heldout_block_lift-seed-{seed}.json"
            artifact.write_text(
                json.dumps(
                    {
                        "condition": condition,
                        "seed": seed,
                        "mujoco_model_identity": MODEL_IDENTITY,
                        "model_identity_source": (
                            ModelIdentitySource.ROLLOUT_SESSION.value
                        ),
                    }
                ),
                encoding="utf-8",
            )
            if error is not None:
                errors.append(error)
            trials.append(
                {
                    "trial_id": f"heldout_block_lift:seed:{seed}",
                    "task_id": "heldout_block_lift",
                    "object_body": "task_block",
                    "object_physical_profile_sha256": OBJECT_PROFILE_SHA256,
                    "mujoco_model_identity": MODEL_IDENTITY,
                    "model_identity_source": (
                        ModelIdentitySource.ROLLOUT_SESSION.value
                    ),
                    "seed": seed,
                    "status": "execution_failure" if aborted else "scored",
                    "success": success,
                    "object_position_error_m": error,
                    "failure_reason": (
                        "safety_watchdog"
                        if aborted
                        else None if success else "object_body_position_tolerance"
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
        calls.append(
            {
                "condition": condition,
                "manifest_path": Path(manifest_path),
                "checkpoint_path": Path(checkpoint_path),
                "training_report_path": training_report,
                "prompt_task_expectation": prompt_task_expectation,
                "device": device,
            }
        )
        training_report_sha256 = file_sha256(training_report)
        if corrupt_training_hash and condition == PromptCondition.MATCHED.value:
            training_report_sha256 = "0" * 64
        task_summary = {
            "task_id": "heldout_block_lift",
            "label": "Heldout block lift",
            "object_body": "task_block",
            "object_physical_profile_sha256": OBJECT_PROFILE_SHA256,
            "mujoco_model_identity": MODEL_IDENTITY,
            "model_identity_sources": [ModelIdentitySource.ROLLOUT_SESSION.value],
            "trial_count": 3,
            "scored_trial_count": len(errors),
            "execution_failure_count": 3 - len(errors),
            "success_count": success_count,
            "success_rate": success_count / 3,
            "success_rate_95ci": [0.0, 1.0],
            "object_position_error_mean_m": (
                None if not errors else sum(errors) / len(errors)
            ),
            "object_position_error_max_m": None if not errors else max(errors),
            "failure_counts": (
                {"safety_watchdog": 3}
                if aborted
                else {}
                if success_count == 3
                else {"object_body_position_tolerance": 3 - success_count}
            ),
        }
        return {
            "schema_version": SEMANTIC_REPORT_SCHEMA,
            "gate": CHECKPOINT_SEMANTIC_GATE,
            "result": "pass" if success_count == 3 else "fail",
            "mode": "mujoco",
            "evidence_level": "simulation",
            "benchmark_id": "local-object-state-v1",
            "config_sha256": project_config_sha256(config),
            "manifest_sha256": file_sha256(manifest_path),
            "dataset_task": DATASET_TASK,
            "dataset_task_index": DATASET_TASK_INDEX,
            "policy": CHECKPOINT_SEMANTIC_POLICY,
            "benchmark_scope": CHECKPOINT_SEMANTIC_SCOPE,
            "criterion": "object_body_position",
            "object_body": "task_block",
            "object_physical_profile": OBJECT_PROFILE,
            "object_physical_profile_sha256": OBJECT_PROFILE_SHA256,
            "mujoco_model_identity": MODEL_IDENTITY,
            "model_identity_sources": [ModelIdentitySource.ROLLOUT_SESSION.value],
            "task_disjoint": True,
            "train_task_ids": ["train_block_home"],
            "heldout_task_ids": ["heldout_block_lift"],
            "robot_used": False,
            "semantic_mujoco_object_state_evaluated": bool(errors),
            "semantic_heldout_success_claimed": False,
            "prompt_causality_claimed": False,
            "real_world_success_claimed": False,
            "official_zero_wam_claimed": False,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_id": "semantic-candidate",
            "training_report_sha256": training_report_sha256,
            "training_evidence_sha256": "1" * 64,
            "prompt_npz_sha256": file_sha256(prompt_target),
            "prompt_manifest_sha256": file_sha256(prompt_manifest),
            "prompt_fingerprint": prompt.fingerprint,
            "prompt_task": prompt.task,
            "prompt_task_index": prompt.task_index,
            "prompt_task_expectation": prompt_task_expectation.value,
            "prompt_task_identity_matches_manifest": (
                condition != PromptCondition.WRONG_TASK.value
            ),
            "prompt_task_match": (
                "control_mismatch_verified"
                if condition == PromptCondition.WRONG_TASK.value
                else "dataset_identity_verified"
            ),
            "task_disjoint_basis": "checkpoint_bound_task_inventory",
            "checkpoint_task_split": "training_report_verified",
            "semantic_task_mapping": "manifest_declared",
            "summary": {
                "task_count": 1,
                "trial_count": 3,
                "scored_trial_count": len(errors),
                "execution_failure_count": 3 - len(errors),
                "success_count": success_count,
                "success_rate": success_count / 3,
                "success_rate_95ci": [0.0, 1.0],
                "all_trials_successful": success_count == 3,
                "object_position_error_mean_m": (
                    None if not errors else sum(errors) / len(errors)
                ),
                "object_position_error_max_m": None if not errors else max(errors),
                "failure_counts": (
                    {"safety_watchdog": 3}
                    if aborted
                    else {}
                    if success_count == 3
                    else {"object_body_position_tolerance": 3 - success_count}
                ),
                "mujoco_model_identity": MODEL_IDENTITY,
                "model_identity_sources": [ModelIdentitySource.ROLLOUT_SESSION.value],
            },
            "tasks": [task_summary],
            "trials": trials,
        }

    return run


def test_semantic_prompt_controls_run_all_conditions_with_expected_task_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_semantic_prompt_controls.run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
        device="cpu",
    )

    assert tuple(call["condition"] for call in calls) == CONDITIONS
    assert {
        call["prompt_task_expectation"]
        for call in calls
        if call["condition"] != "wrong_task"
    } == {PromptTaskExpectation.MATCH_MANIFEST}
    assert [
        call["prompt_task_expectation"]
        for call in calls
        if call["condition"] == "wrong_task"
    ] == [PromptTaskExpectation.EXPECTED_MISMATCH]
    assert {call["manifest_path"] for call in calls} == {semantic_manifest}
    assert {call["checkpoint_path"] for call in calls} == {checkpoint}
    assert {call["training_report_path"] for call in calls} == {training_report}

    assert report["gate"] == MUJOCO_SEMANTIC_PROMPT_CONTROL_GATE
    assert report["schema_version"] == 4
    assert report["result"] == "complete"
    assert report["scope"] == MUJOCO_SEMANTIC_PROMPT_CONTROL_SCOPE
    assert report["semantic_mujoco_object_state_evaluated"] is True
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["directionality_preregistered"] is False
    assert report["scientific_pass_fail_evaluated"] is False
    assert report["task_disjoint_basis"] == "checkpoint_bound_task_inventory"
    assert report["checkpoint_task_split"] == "training_report_verified"
    assert report["semantic_task_mapping"] == "manifest_declared"
    assert report["object_body"] == "task_block"
    assert report["object_physical_profile_sha256"] == OBJECT_PROFILE_SHA256
    assert report["mujoco_model_identity"] == MODEL_IDENTITY
    assert report["model_identity_sources"] == [
        ModelIdentitySource.ROLLOUT_SESSION.value
    ]
    assert report["prompt_control_manifest_sha256"] == file_sha256(prompt_manifest)
    assert report["semantic_manifest_sha256"] == file_sha256(semantic_manifest)
    assert report["training_report_sha256"] == file_sha256(training_report)
    assert report["checkpoint_sha256"] == file_sha256(checkpoint)
    assert report["summary"]["condition_count"] == len(CONDITIONS)
    assert report["summary"]["trials_per_condition"] == 3
    assert report["summary"]["total_trial_count"] == 3 * len(CONDITIONS)
    assert report["summary"]["mujoco_model_identity"] == MODEL_IDENTITY
    assert report["summary"]["model_identity_sources"] == [
        ModelIdentitySource.ROLLOUT_SESSION.value
    ]

    condition_reports = {
        condition["condition"]: condition for condition in report["conditions"]
    }
    assert tuple(condition_reports) == CONDITIONS
    assert (
        condition_reports["matched"]["object_position_error_mean_delta_vs_matched_m"]
        == 0.0
    )
    assert all(
        condition["training_report_sha256"] == file_sha256(training_report)
        for condition in report["conditions"]
    )
    assert all(
        condition["mujoco_model_identity"] == MODEL_IDENTITY
        and condition["model_identity_sources"]
        == [ModelIdentitySource.ROLLOUT_SESSION.value]
        for condition in report["conditions"]
    )
    for condition in report["conditions"]:
        nested_path = tmp_path / "artifacts" / condition["benchmark_report"]
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        assert nested["mujoco_model_identity"] == MODEL_IDENTITY
        assert nested["summary"]["mujoco_model_identity"] == MODEL_IDENTITY
        assert nested["tasks"][0]["mujoco_model_identity"] == MODEL_IDENTITY
        for trial in nested["trials"]:
            assert trial["mujoco_model_identity"] == MODEL_IDENTITY
            assert trial["model_identity_source"] == (
                ModelIdentitySource.ROLLOUT_SESSION.value
            )
    assert (
        condition_reports["wrong_task"]["prompt_task_expectation"]
        == "expected_mismatch"
    )
    assert (
        condition_reports["wrong_task"]["prompt_task_match"]
        == "control_mismatch_verified"
    )
    assert all(
        condition["prompt_task_expectation"] == "match_manifest"
        for name, condition in condition_reports.items()
        if name != "wrong_task"
    )
    assert all(
        condition["prompt_task_match"] == "dataset_identity_verified"
        for name, condition in condition_reports.items()
        if name != "wrong_task"
    )


def test_semantic_prompt_controls_preserve_execution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(
            calls,
            execution_failure_condition=PromptCondition.MATCHED.value,
        ),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        artifact_dir=tmp_path / "recoverable-artifacts",
        report_path=tmp_path / "recoverable-report.json",
    )

    by_condition = {
        condition["condition"]: condition for condition in report["conditions"]
    }
    matched = by_condition[PromptCondition.MATCHED.value]
    assert len(calls) == len(CONDITIONS)
    assert report["result"] == "complete"
    assert report["summary"]["scored_trial_count"] == 18
    assert report["summary"]["execution_failure_count"] == 3
    assert report["semantic_mujoco_object_state_evaluated"] is True
    assert report["summary"]["matched_object_position_error_mean_m"] is None
    assert matched["scored_trial_count"] == 0
    assert matched["execution_failure_count"] == 3
    assert matched["object_position_error_mean_m"] is None
    assert matched["object_position_error_mean_delta_vs_matched_m"] is None
    assert all(
        condition["object_position_error_mean_delta_vs_matched_m"] is None
        for condition in report["conditions"]
    )


def test_prompt_missing_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    fake_run = _fake_semantic_benchmark([])

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = fake_run(*args, **kwargs)
        del report["mujoco_model_identity"]
        return report

    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        tampered_run,
    )

    with pytest.raises(PromptControlError, match="model identity"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_prompt_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    fake_run = _fake_semantic_benchmark([])

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = fake_run(*args, **kwargs)
        if Path(str(kwargs["prompt_path"])).stem == PromptCondition.WRONG_TASK.value:
            report["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
            report["summary"]["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
            report["tasks"][0]["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
            for trial in report["trials"]:
                trial["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
                artifact_path = Path(str(kwargs["artifact_dir"])) / str(
                    trial["artifact"]
                )
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                artifact["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
                artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
                trial["artifact_sha256"] = file_sha256(artifact_path)
        return report

    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        tampered_run,
    )

    with pytest.raises(PromptControlError, match="model identity changed"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_prompt_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    fake_run = _fake_semantic_benchmark([])

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = fake_run(*args, **kwargs)
        trial = report["trials"][0]
        artifact_path = Path(str(kwargs["artifact_dir"])) / str(trial["artifact"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        trial["artifact_sha256"] = file_sha256(artifact_path)
        return report

    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        tampered_run,
    )

    with pytest.raises(PromptControlError, match="model identity"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_prompt_invalid_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    fake_run = _fake_semantic_benchmark([])

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = fake_run(*args, **kwargs)
        report["trials"][0]["model_identity_source"] = [
            ModelIdentitySource.ROLLOUT_SESSION.value
        ]
        return report

    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        tampered_run,
    )

    with pytest.raises(PromptControlError, match="model identity source"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_prompt_summary_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    fake_run = _fake_semantic_benchmark([])

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = fake_run(*args, **kwargs)
        report["summary"]["model_identity_sources"] = [
            ModelIdentitySource.POST_FAILURE_INSPECTION.value
        ]
        return report

    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        tampered_run,
    )

    with pytest.raises(PromptControlError, match="summary model identity"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_prompt_task_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    fake_run = _fake_semantic_benchmark([])

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = fake_run(*args, **kwargs)
        report["tasks"] = []
        return report

    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        tampered_run,
    )

    with pytest.raises(PromptControlError, match="task summaries"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_semantic_prompt_controls_handle_no_scored_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls, execution_failure_condition="*"),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        artifact_dir=tmp_path / "unscored-artifacts",
        report_path=tmp_path / "unscored-report.json",
    )

    assert len(calls) == len(CONDITIONS)
    assert report["summary"]["scored_trial_count"] == 0
    assert report["summary"]["execution_failure_count"] == 21
    assert report["semantic_mujoco_object_state_evaluated"] is False
    assert report["summary"]["matched_object_position_error_mean_m"] is None


def test_semantic_prompt_controls_reject_missing_training_report_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_semantic_prompt_controls.run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls),
    )

    with pytest.raises(PromptControlError, match="training report"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=tmp_path / "missing.training.json",
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert calls == []
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "report.json").exists()


def test_semantic_prompt_controls_reject_nested_training_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_semantic_prompt_controls.run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls, corrupt_training_hash=True),
    )

    with pytest.raises(PromptControlError, match="training_report_sha256"):
        run_mujoco_semantic_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert len(calls) == 1
    assert not (tmp_path / "report.json").exists()


def test_semantic_prompt_controls_parse_frozen_prompt_manifest_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    original = prompt_manifest.read_bytes()
    mutated_checkpoint = tmp_path / "mutated.pt"
    mutated_checkpoint.write_bytes(b"mutated-checkpoint")
    real_loader = prompt_controls_module.load_prompt_control_manifest_bytes

    def mutate_then_load(
        source: bytes,
        *,
        root: str | Path,
    ) -> object:
        payload = json.loads(prompt_manifest.read_text(encoding="utf-8"))
        payload["suite_id"] = "mutated-suite"
        payload["checkpoint"] = mutated_checkpoint.name
        prompt_manifest.write_text(json.dumps(payload), encoding="utf-8")
        return real_loader(source, root=root)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "load_prompt_control_manifest_bytes",
        mutate_then_load,
    )
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    assert report["suite_id"] == "semantic-prompt-control-test"
    assert report["prompt_control_manifest_sha256"] == sha256(original).hexdigest()
    assert {call["checkpoint_path"] for call in calls} == {checkpoint}


def test_semantic_prompt_controls_reject_invalid_directionality_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    direction_plan = _direction_plan(
        tmp_path,
        prompt_manifest=prompt_manifest,
        semantic_manifest=semantic_manifest,
        training_report=training_report,
        config=config,
        mutation=("schema_version", 2),
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls),
    )

    with pytest.raises(PromptControlError, match="schema_version"):
        run_mujoco_semantic_prompt_control_suite(
            config,
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            direction_preregistration_path=direction_plan,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert calls == []
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "report.json").exists()


def test_semantic_prompt_controls_bind_directionality_hashes_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    direction_plan = _direction_plan(
        tmp_path,
        prompt_manifest=prompt_manifest,
        semantic_manifest=semantic_manifest,
        training_report=training_report,
        config=config,
        mutation=("training_report_sha256", "f" * 64),
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls),
    )

    with pytest.raises(PromptControlError, match="training_report_sha256"):
        run_mujoco_semantic_prompt_control_suite(
            config,
            prompt_control_manifest_path=prompt_manifest,
            semantic_manifest_path=semantic_manifest,
            training_report_path=training_report,
            direction_preregistration_path=direction_plan,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert calls == []
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "report.json").exists()


def test_semantic_prompt_controls_pass_preregistered_object_state_directionality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    direction_plan = _direction_plan(
        tmp_path,
        prompt_manifest=prompt_manifest,
        semantic_manifest=semantic_manifest,
        training_report=training_report,
        config=config,
    )
    original_plan = direction_plan.read_bytes()
    metrics = {
        "matched": (3, 0.020),
        "same_task_alternate": (3, 0.024),
        "wrong_task": (0, 0.090),
        "temporal_shuffle": (2, 0.050),
        "image_frame_shuffle": (2, 0.055),
        "null": (0, 0.120),
        "counterfactual": (0, 0.110),
    }
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls, condition_metrics=metrics),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        config,
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        direction_preregistration_path=direction_plan,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    assert len(calls) == len(CONDITIONS)
    assert report["result"] == "complete"
    assert report["directionality_preregistered"] is True
    assert report["scientific_pass_fail_evaluated"] is True
    assert report["directionality_evaluation"]["result"] == "pass"
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["real_world_success_claimed"] is False
    assert report["official_zero_wam_claimed"] is False
    assert report["preregistration_level"] == (
        "local_input_hash_bound_before_rollout"
    )
    assert report["external_preregistration_timestamp_verified"] is False
    artifact = tmp_path / "artifacts" / report[
        "semantic_direction_preregistration_artifact"
    ]
    assert artifact.read_bytes() == original_plan
    assert report["semantic_direction_preregistration_sha256"] == sha256(
        original_plan
    ).hexdigest()
    assert report["semantic_direction_preregistration_artifact_sha256"] == (
        file_sha256(artifact)
    )


def test_semantic_prompt_controls_keep_complete_for_negative_directionality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    direction_plan = _direction_plan(
        tmp_path,
        prompt_manifest=prompt_manifest,
        semantic_manifest=semantic_manifest,
        training_report=training_report,
        config=config,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        config,
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        direction_preregistration_path=direction_plan,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    assert report["result"] == "complete"
    assert report["directionality_evaluation"]["result"] == "fail"
    assert report["directionality_evaluation"]["passed_rule_count"] == 2
    assert report["prompt_causality_claimed"] is False


def test_semantic_directionality_fails_when_terminal_endpoint_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    direction_plan = _direction_plan(
        tmp_path,
        prompt_manifest=prompt_manifest,
        semantic_manifest=semantic_manifest,
        training_report=training_report,
        config=config,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(
            calls,
            execution_failure_condition=PromptCondition.MATCHED.value,
        ),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        config,
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        direction_preregistration_path=direction_plan,
        artifact_dir=tmp_path / "missing-endpoint-artifacts",
        report_path=tmp_path / "missing-endpoint-report.json",
    )

    rules = {
        rule["rule_id"]: rule
        for rule in report["directionality_evaluation"]["rules"]
    }
    error_rule = rules["negative_object_position_error_margin"]
    assert len(calls) == len(CONDITIONS)
    assert report["result"] == "complete"
    assert report["directionality_evaluation"]["result"] == "fail"
    assert error_rule["result"] == "fail"
    assert error_rule["endpoint_available"] is False
    assert error_rule["observed_margin_m"] is None


def test_semantic_prompt_controls_freeze_directionality_bytes_before_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    prompt_manifest = _prompt_control_bundle(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint = tmp_path / "semantic-candidate.pt"
    training_report = _training_report(tmp_path, checkpoint)
    direction_plan = _direction_plan(
        tmp_path,
        prompt_manifest=prompt_manifest,
        semantic_manifest=semantic_manifest,
        training_report=training_report,
        config=config,
    )
    original_plan = direction_plan.read_bytes()
    real_loader = semantic_direction_module.load_semantic_direction_plan_bytes

    def mutate_then_load(source: bytes, **expected: str):
        payload = json.loads(direction_plan.read_text(encoding="utf-8"))
        payload["negative_success_margin"] = 1.0
        direction_plan.write_text(json.dumps(payload), encoding="utf-8")
        return real_loader(source, **expected)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_controls_module,
        "load_semantic_direction_plan_bytes",
        mutate_then_load,
    )
    monkeypatch.setattr(
        semantic_controls_module,
        "run_checkpoint_benchmark",
        _fake_semantic_benchmark(calls),
    )

    report = run_mujoco_semantic_prompt_control_suite(
        config,
        prompt_control_manifest_path=prompt_manifest,
        semantic_manifest_path=semantic_manifest,
        training_report_path=training_report,
        direction_preregistration_path=direction_plan,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    artifact = tmp_path / "artifacts" / report[
        "semantic_direction_preregistration_artifact"
    ]
    assert artifact.read_bytes() == original_plan
    assert report["semantic_direction_preregistration_sha256"] == sha256(
        original_plan
    ).hexdigest()


def test_semantic_direction_artifact_rejects_concurrent_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    target = artifact_root / "semantic_directionality_preregistration.json"
    target.write_bytes(b"concurrent-evidence")
    real_exists = Path.exists

    def hide_target(path: Path) -> bool:
        if path == target:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", hide_target)

    with pytest.raises(PromptControlError, match="already exists"):
        semantic_controls_module._copy_directionality_plan(
            b"new-evidence",
            artifact_root,
        )

    assert target.read_bytes() == b"concurrent-evidence"
