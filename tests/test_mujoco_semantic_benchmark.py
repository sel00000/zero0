from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import so101_wam.mujoco_semantic_benchmark as semantic_module
from so101_wam.checkpoint import (
    compact_wam_architecture,
    save_compact_wam_checkpoint,
)
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import ActionChunk, SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.deployment import canonical_json_sha256, file_sha256
from so101_wam.model import CompactWAM
from so101_wam.adapters.mujoco import (
    CollisionContact,
    MujocoAdapterError,
    MujocoCollisionError,
)
from so101_wam.mujoco_cli import MujocoCLIError
from so101_wam.mujoco_semantic_benchmark import (
    CHECKPOINT_SEMANTIC_GATE,
    CHECKPOINT_SEMANTIC_POLICY,
    CHECKPOINT_SEMANTIC_SCOPE,
    PromptTaskExpectation,
    SEMANTIC_BENCHMARK_POLICY,
    SEMANTIC_BENCHMARK_SCOPE,
    MujocoSemanticBenchmarkError,
    SemanticTask,
    SemanticTrialResult,
    SemanticTrialStatus,
    load_semantic_manifest,
    run_benchmark,
    run_checkpoint_benchmark,
    run_checkpoint_task_trial,
    run_task_spec_checkpoint_task_trial,
    score_object_position,
    validate_failure_evidence,
)
from so101_wam.mujoco_identity import MujocoModelIdentity
from so101_wam.policy import PolicyError
from so101_wam.rollout import ExecutabilityTrace, RolloutError, SafetyRejectedError
from so101_wam.training import TRAINING_REPORT_SCHEMA


MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
requires_mujoco = pytest.mark.skipif(
    not MUJOCO_AVAILABLE, reason="optional mujoco dependency is absent"
)

# This stays inside safety limits and reaches torso/right-arm contact in three steps.
COLLISION_POLICY_STEPS = 3
COLLISION_TARGET = (
    11.5068,
    -23.0668,
    -15.6688,
    -21.3850,
    -15.7097,
    2.9071,
    -12.7163,
    28.9229,
    21.6912,
    -14.3623,
    23.6024,
    8.5892,
)
WATCHDOG_POLICY_HZ = 5.0
TRACE_LOWER = (-10.0,) * ACTION_DIM
TRACE_UPPER = (10.0,) * ACTION_DIM
OBJECT_PROFILE = {
    "schema_version": 1,
    "body": {
        "mass": 0.08,
        "inertia": [0.000066, 0.000066, 0.000066],
        "inertial_pos": [0.0, 0.0, 0.0],
        "inertial_quat": [1.0, 0.0, 0.0, 0.0],
    },
    "geom": {
        "type": "box",
        "size": [0.025, 0.025, 0.025],
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "friction": [1.0, 0.005, 0.0001],
        "margin": 0.0,
        "gap": 0.0,
        "solref": [0.02, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    },
}
OBJECT_PROFILE_SHA256 = canonical_json_sha256(OBJECT_PROFILE)
MUJOCO_MODEL_IDENTITY = MujocoModelIdentity(
    engine_version="3.12.0",
    compiled_model_sha256="f" * 64,
    compiled_model_bytes=128,
).to_payload()


def _model_identity(sha256_value: str = "f" * 64) -> dict[str, object]:
    result = dict(MUJOCO_MODEL_IDENTITY)
    result["compiled_model_sha256"] = sha256_value
    return result


def _scored_trace(task: SemanticTask, seed: int) -> dict[str, Any]:
    trace = ExecutabilityTrace(
        f"{task.task_id}:seed:{seed}",
        joint_lower=TRACE_LOWER,
        joint_upper=TRACE_UPPER,
    )
    safety = SimpleNamespace(accepted=True, clipped=False, reasons=())
    for policy_index in range(task.policy_steps):
        target = np.zeros((1, ACTION_DIM), dtype=np.float32)
        action = ActionChunk(
            target_joint_position=target,
            dt_s=0.02,
            created_at_s=float(policy_index),
        )
        trace.on_policy_step(
            policy_index=policy_index,
            step=SimpleNamespace(action=action),
        )
        trace.on_future_latent_step(
            policy_index=policy_index,
            observation_timestamp_s=float(policy_index + 1),
            future_latents=np.ones((2, 2, 3), dtype=np.float32),
            observed_latent=np.zeros((2, 3), dtype=np.float32),
        )
        trace.on_servo_step(
            policy_index=policy_index,
            servo_index=0,
            step=SimpleNamespace(
                action=action,
                observation=SimpleNamespace(joint_position=target[0]),
                executed_action=target[0],
                sent=True,
                shadow=False,
                safety=safety,
            ),
        )
        trace.on_contact_sample(
            policy_index=policy_index,
            servo_index=0,
            phase="servo",
            contacts=(),
        )
    return trace.snapshot()


def _initial_positions() -> list[dict[str, object]]:
    return [
        {"seed": 7, "position": [0.32, -0.02, 0.475]},
        {"seed": 13, "position": [0.34, 0.0, 0.475]},
        {"seed": 29, "position": [0.36, 0.02, 0.475]},
    ]


def _manifest(tmp_path: Path, **overrides: object) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 3,
        "benchmark_id": "local-object-state-v1",
        "dataset_task": {
            "task_index": 201,
            "task": "synthetic-validation-place",
        },
        "train_task_ids": ["train_block_home"],
        "heldout_tasks": [
            {
                "task_id": "heldout_block_lift",
                "label": "held-out block lift",
                "policy_steps": 1,
                "seeds": [7, 13, 29],
                "object_body": "task_block",
                "initial_object_positions": _initial_positions(),
                "target_object_position": [0.34, 0.0, 0.58],
                "position_tolerance_m": 0.03,
            }
        ],
    }
    payload.update(overrides)
    path = tmp_path / "semantic-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _task(*, target: tuple[float, float, float] = (0.34, 0.0, 0.58)) -> SemanticTask:
    return SemanticTask(
        task_id="heldout_block_lift",
        label="held-out block lift",
        policy_steps=1,
        seeds=(7, 13, 29),
        object_body="task_block",
        initial_object_positions=(
            (7, (0.32, -0.02, 0.475)),
            (13, (0.34, 0.0, 0.475)),
            (29, (0.36, 0.02, 0.475)),
        ),
        target_object_position=target,
        position_tolerance_m=0.03,
    )


def _prompt_episode(
    tmp_path: Path,
    *,
    task: str = "synthetic-validation-place",
    task_index: int = 201,
    image_height: int = 48,
    image_width: int = 64,
) -> tuple[Path, Path]:
    buffer = EpisodeBuffer(
        fps=30.0,
        task=task,
        task_index=task_index,
        episode_index=1,
    )
    joints = np.zeros(ACTION_DIM, dtype=np.float32)
    images = {
        key: np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for key in PRIMARY_CAMERA_KEYS
    }
    for frame_index in range(91):
        buffer.append(
            SensorimotorFrame(
                timestamp_s=frame_index / 30.0,
                images=images,
                joint_position=joints,
                executed_action=joints,
            )
        )
    return buffer.save(tmp_path, stem="object-prompt")


def _constant_target_checkpoint(
    tmp_path: Path,
    *,
    checkpoint_id: str,
    target: tuple[float, ...],
) -> Path:
    model = CompactWAM(
        latent_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        action_history_steps=1,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.set_axis_normalization(torch.tensor(target), torch.ones(ACTION_DIM))

    checkpoint = tmp_path / f"{checkpoint_id}.pt"
    save_compact_wam_checkpoint(
        model,
        checkpoint,
        metadata={"checkpoint_id": checkpoint_id},
    )
    return checkpoint


def _candidate_with_training_report(
    tmp_path: Path,
    *,
    checkpoint_id: str,
    train_tasks: list[dict[str, object]] | None = None,
    validation_tasks: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    model = CompactWAM(latent_dim=8, transformer_layers=1, transformer_heads=2)
    train_inventory = train_tasks or [
        {"task_index": 101, "task": "synthetic-train-reach"}
    ]
    validation_inventory = validation_tasks or [
        {"task_index": 201, "task": "synthetic-validation-place"}
    ]
    report_core = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "evidence_level": "offline",
        "artifact_kind": "compact_wam_candidate",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": checkpoint_id,
        "protocol": {
            "split": "task_disjoint",
            "real_output_authorized": False,
        },
        "data": {
            "train_task_count": len(train_inventory),
            "validation_task_count": len(validation_inventory),
            "train_tasks": train_inventory,
            "validation_tasks": validation_inventory,
        },
        "model": compact_wam_architecture(model),
    }
    training_evidence = canonical_json_sha256(report_core)
    checkpoint = tmp_path / f"{checkpoint_id}.pt"
    save_compact_wam_checkpoint(
        model,
        checkpoint,
        metadata={
            "artifact_kind": "compact_wam_candidate",
            "evidence_level": "offline",
            "checkpoint_id": checkpoint_id,
            "trained": False,
            "offline_trained": True,
            "deployment_ready": False,
            "training_evidence_sha256": training_evidence,
        },
    )
    report = {
        **report_core,
        "training_evidence_sha256": training_evidence,
        "artifacts": {"checkpoint_sha256": file_sha256(checkpoint)},
    }
    report_path = tmp_path / f"{checkpoint_id}.training.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return checkpoint, report_path


def test_manifest_binds_one_initial_object_position_per_seed(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        heldout_tasks=[
            {
                    "task_id": "heldout_block_lift",
                    "label": "held-out block lift",
                    "policy_steps": 1,
                    "seeds": [7, 13, 29],
                    "object_body": "task_block",
                    "initial_object_positions": _initial_positions()[:-1],
                "target_object_position": [0.34, 0.0, 0.58],
                "position_tolerance_m": 0.03,
            }
        ],
    )

    with pytest.raises(MujocoSemanticBenchmarkError, match="exactly one"):
        load_semantic_manifest(path)


def test_manifest_rejects_duplicate_and_unknown_fields(tmp_path: Path) -> None:
    duplicate = _manifest(tmp_path)
    duplicate.write_text(
        duplicate.read_text(encoding="utf-8").replace(
            '"position_tolerance_m": 0.03',
            '"position_tolerance_m": 0.03, "position_tolerance_m": 0.04',
        ),
        encoding="utf-8",
    )

    with pytest.raises(MujocoSemanticBenchmarkError, match="duplicate"):
        load_semantic_manifest(duplicate)

    unknown = _manifest(tmp_path)
    payload = json.loads(unknown.read_text(encoding="utf-8"))
    payload["heldout_tasks"][0]["ignored_target"] = [1, 2, 3]
    unknown.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MujocoSemanticBenchmarkError, match="unknown fields"):
        load_semantic_manifest(unknown)


def test_benchmark_hashes_the_manifest_bytes_it_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _manifest(tmp_path)
    source_bytes = manifest_path.read_bytes()
    real_loader = semantic_module.load_semantic_manifest_bytes

    def mutate_after_snapshot(source: bytes) -> object:
        manifest_path.write_text("{}", encoding="utf-8")
        return real_loader(source)

    monkeypatch.setattr(
        semantic_module,
        "load_semantic_manifest_bytes",
        mutate_after_snapshot,
    )

    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=_scored_trace(task, seed),
            mujoco_model_identity=_model_identity(),
            model_identity_source="rollout_session",
        )

    report = run_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=manifest_path,
        artifact_dir=tmp_path / "snapshot-artifacts",
        trial_runner=fake_runner,
    )

    assert report["manifest_sha256"] == sha256(source_bytes).hexdigest()


def test_default_semantic_manifest_is_task_disjoint() -> None:
    manifest = load_semantic_manifest(Path("configs/mujoco_semantic_benchmark.json"))

    assert manifest.train_task_ids == ("train_block_home",)
    assert manifest.dataset_task == "synthetic-validation-place"
    assert manifest.dataset_task_index == 201
    assert tuple(task.task_id for task in manifest.heldout_tasks) == (
        "heldout_block_lift",
    )


def test_object_position_checker_uses_euclidean_metres() -> None:
    task = _task(target=(0.34, 0.0, 0.50))

    passed = score_object_position(task, (0.34, 0.0, 0.48))
    failed = score_object_position(task, (0.34, 0.0, 0.46))

    assert passed.success is True
    assert passed.object_position_error_m == pytest.approx(0.02)
    assert passed.failure_reason is None
    assert failed.success is False
    assert failed.object_position_error_m == pytest.approx(0.04)
    assert failed.failure_reason == "object_body_position_tolerance"


def test_object_position_checker_rejects_non_finite_measurement() -> None:
    with pytest.raises(MujocoSemanticBenchmarkError, match="finite"):
        score_object_position(_task(), (0.34, 0.0, float("nan")))


def test_semantic_benchmark_records_object_evidence_per_seed(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    calls: list[int] = []

    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config
        calls.append(seed)
        success = seed != 13
        initial = task.initial_position(seed)
        final = task.target_object_position if success else initial
        error = score_object_position(task, final).object_position_error_m
        return SemanticTrialResult(
            success=success,
            object_position_error_m=error,
            failure_reason=None if success else "object_body_position_tolerance",
            rollout={"policy_steps": task.policy_steps, "final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=initial,
            final_object_position=final,
            executability_trace=_scored_trace(task, seed),
        )

    report = run_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        trial_runner=fake_runner,
    )

    assert calls == [7, 13, 29]
    assert report["result"] == "fail"
    assert report["policy"] == SEMANTIC_BENCHMARK_POLICY
    assert report["benchmark_scope"] == SEMANTIC_BENCHMARK_SCOPE
    assert report["semantic_mujoco_object_state_evaluated"] is True
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["real_world_success_claimed"] is False
    assert report["official_zero_wam_claimed"] is False
    assert report["summary"]["trial_count"] == 3
    assert report["summary"]["success_count"] == 2
    assert report["summary"]["executability_trace_trial_count"] == 3
    assert report["executability_trace_present"] is True
    assert report["future_visual_success_claimed"] is False
    assert report["runtime_future_latent_proxy_available"] is False
    assert report["runtime_future_latent_success_claimed"] is False
    assert report["summary"]["executability_contact_progression_count"] == 3
    assert report["summary"]["executability_collision_failure_count"] == 0
    assert report["summary"]["failure_counts"] == {"object_body_position_tolerance": 1}
    for trial in report["trials"]:
        artifact_path = artifact_dir / trial["artifact"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        trace_path = artifact_dir / trial["executability_trace_artifact"]
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trial["artifact_sha256"] == file_sha256(artifact_path)
        assert trial["rollout_id"] == f"{trial['task_id']}:seed:{trial['seed']}"
        assert trial["executability_trace_sha256"] == file_sha256(trace_path)
        assert artifact["executability_trace_sha256"] == file_sha256(trace_path)
        assert trace["rollout_id"] == trial["rollout_id"]
        assert artifact["criterion"] == "object_body_position"
        assert artifact["target_object_position"] == [0.34, 0.0, 0.58]
        assert artifact["semantic_heldout_success_claimed"] is False
        assert artifact["future_visual_success_claimed"] is False

    with pytest.raises(MujocoSemanticBenchmarkError, match="already exists"):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=manifest_path,
            artifact_dir=artifact_dir,
            trial_runner=fake_runner,
        )
    assert len(calls) == 3


def test_semantic_benchmark_aggregates_future_latent_proxy(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["heldout_tasks"][0]["policy_steps"] = 2
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"policy_steps": task.policy_steps, "final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=_scored_trace(task, seed),
            mujoco_model_identity=_model_identity(),
            model_identity_source="rollout_session",
        )

    artifact_dir = tmp_path / "future-latent-artifacts"
    report = run_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        trial_runner=fake_runner,
    )

    assert report["future_visual_metric_available"] is False
    assert report["runtime_future_latent_proxy_available"] is True
    assert report["runtime_future_latent_success_claimed"] is False
    assert report["summary"]["future_latent_trace_trial_count"] == 3
    assert report["summary"]["future_latent_prediction_count"] == 6
    assert report["summary"]["future_latent_observation_count"] == 6
    assert report["summary"]["future_latent_aligned_pair_count"] == 3
    assert report["summary"]["future_latent_censored_pair_count"] == 9
    assert report["summary"]["future_latent_element_count"] == 18
    assert report["summary"]["future_latent_squared_error_sum"] == pytest.approx(18.0)
    assert report["summary"]["future_latent_mse_mean"] == pytest.approx(1.0)
    for trial in report["trials"]:
        artifact = json.loads(
            (artifact_dir / trial["artifact"]).read_text(encoding="utf-8")
        )
        assert artifact["future_visual_metric_available"] is False
        assert artifact["runtime_future_latent_proxy_available"] is True
        assert artifact["runtime_future_latent_success_claimed"] is False


def test_semantic_benchmark_rejects_existing_trace_before_trials(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    trace_path = artifact_dir / "heldout_block_lift-seed-7.executability.json"
    trace_path.write_text("{}", encoding="utf-8")
    calls: list[int] = []

    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config, task
        calls.append(seed)
        raise AssertionError("runner must not be called")

    with pytest.raises(MujocoSemanticBenchmarkError, match="already exists"):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=artifact_dir,
            trial_runner=fake_runner,
        )

    assert calls == []


@pytest.mark.parametrize(
    "include_empty_trace",
    [False, True],
    ids=("missing", "empty"),
)
def test_scored_trial_requires_complete_executability_trace(
    tmp_path: Path,
    include_empty_trace: bool,
) -> None:
    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config
        trace = None
        if include_empty_trace:
            trace = ExecutabilityTrace(f"{task.task_id}:seed:{seed}").snapshot()
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"execution_status": "complete"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=trace,
        )

    with pytest.raises(
        MujocoSemanticBenchmarkError,
        match="scored trial requires complete executability trace",
    ):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=tmp_path / "incomplete-trace-artifacts",
            trial_runner=fake_runner,
        )


def test_checkpoint_trial_seeds_object_and_scores_terminal_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(Path("configs/mujoco.toml"))
    task = _task()
    seen_initial: tuple[float, float, float] | None = None

    def fake_session(
        session_config: ProjectConfig,
        *,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        manifest_path: str | Path | None,
        policy_steps: int,
        device: str,
        semantic_object_body: str,
        initial_object_position: tuple[float, float, float],
        rollout_observer: ExecutabilityTrace,
    ) -> dict[str, object]:
        nonlocal seen_initial
        del session_config, checkpoint_path, prompt_path, manifest_path, device
        seen_initial = initial_object_position
        assert rollout_observer.snapshot()["rollout_id"] == (
            "heldout_block_lift:seed:7"
        )
        return {
            "rollout": {"policy_steps": policy_steps, "final_state": "rollout_ready"},
            "object_body": semantic_object_body,
            "object_physical_profile": OBJECT_PROFILE,
            "object_physical_profile_sha256": OBJECT_PROFILE_SHA256,
            "mujoco_model_identity": _model_identity(),
            "terminal_joint_position": [0.0] * ACTION_DIM,
            "terminal_object_position": [0.34, 0.0, 0.475],
        }

    monkeypatch.setattr(
        semantic_module,
        "run_mujoco_checkpoint_session",
        fake_session,
    )

    result = run_checkpoint_task_trial(
        config,
        task,
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        prompt_path=tmp_path / "prompt.npz",
    )

    assert seen_initial == task.initial_position(7)
    assert result.success is False
    assert result.failure_reason == "object_body_position_tolerance"
    assert result.final_object_position == (0.34, 0.0, 0.475)
    assert result.executability_trace is not None
    assert result.executability_trace["rollout_id"] == "heldout_block_lift:seed:7"


@pytest.mark.parametrize(
    ("error", "failure_reason"),
    (
        (
            MujocoCollisionError(
                "servo transition",
                (
                    CollisionContact(
                        geom1="left_link",
                        geom2="table",
                        body1="left_arm",
                        body2="table",
                        category1="left_arm",
                        category2="table",
                        distance=-0.001,
                    ),
                ),
            ),
            "mujoco_collision",
        ),
        (SafetyRejectedError(("action[0]_joint_limit:left_shoulder:above",)), "safety_joint_limit"),
        (SafetyRejectedError(("stale_action",)), "safety_watchdog"),
        (SafetyRejectedError(("stale_observation",)), "safety_rejection"),
        (PolicyError("model action output is non-finite"), "policy_model_error"),
    ),
)
def test_checkpoint_trial_records_recoverable_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    failure_reason: str,
) -> None:
    def fail_session(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise error

    monkeypatch.setattr(semantic_module, "run_mujoco_checkpoint_session", fail_session)

    result = run_checkpoint_task_trial(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        _task(),
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        prompt_path=tmp_path / "prompt.npz",
    )

    assert result.status is SemanticTrialStatus.EXECUTION_FAILURE
    assert result.success is False
    assert result.failure_reason == failure_reason
    assert result.failure_evidence
    assert result.object_position_error_m is None
    assert result.final_joint_position is None
    assert result.final_object_position is None
    assert result.executability_trace is not None
    assert result.executability_trace["rollout_id"] == "heldout_block_lift:seed:7"


def test_joint_adapter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_session(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise MujocoAdapterError("observation outside physical joint limits")

    monkeypatch.setattr(semantic_module, "run_mujoco_checkpoint_session", fail_session)

    result = run_checkpoint_task_trial(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        _task(),
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        prompt_path=tmp_path / "prompt.npz",
    )

    assert result.status is SemanticTrialStatus.EXECUTION_FAILURE
    assert result.success is False
    assert result.failure_reason == "mujoco_adapter_error"
    assert result.failure_evidence == {
        "error_type": "MujocoAdapterError",
        "message": "observation outside physical joint limits",
    }
    assert result.final_joint_position is None
    assert result.final_object_position is None


def test_spec_adapter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_session(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise MujocoAdapterError("observation outside physical joint limits")

    monkeypatch.setattr(
        semantic_module,
        "run_mujoco_task_spec_checkpoint_session",
        fail_session,
    )

    result = run_task_spec_checkpoint_task_trial(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        _task(),
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        task_spec=object(),
    )

    assert result.status is SemanticTrialStatus.EXECUTION_FAILURE
    assert result.success is False
    assert result.failure_reason == "mujoco_adapter_error"
    assert result.failure_evidence == {
        "error_type": "MujocoAdapterError",
        "message": "observation outside physical joint limits",
    }
    assert result.final_joint_position is None
    assert result.final_object_position is None


@pytest.mark.parametrize(
    "evidence",
    (
        {"error_type": "PolicyError", "message": "wrong type"},
        {"error_type": "MujocoAdapterError"},
        {"error_type": "MujocoAdapterError", "message": ""},
    ),
)
def test_bad_adapter_evidence(evidence: dict[str, object]) -> None:
    with pytest.raises(MujocoSemanticBenchmarkError, match="adapter failure"):
        validate_failure_evidence("mujoco_adapter_error", evidence)


@pytest.mark.parametrize(
    "error",
    (
        MujocoCLIError("invalid session contract"),
        RolloutError("rollout clock did not advance"),
    ),
)
def test_checkpoint_trial_keeps_unknown_failures_hard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail_session(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise error

    monkeypatch.setattr(semantic_module, "run_mujoco_checkpoint_session", fail_session)

    with pytest.raises(type(error), match=str(error)):
        run_checkpoint_task_trial(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            _task(),
            seed=7,
            checkpoint_path=tmp_path / "candidate.pt",
            prompt_path=tmp_path / "prompt.npz",
        )


def test_semantic_benchmark_continues_after_execution_failure(tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config
        calls.append(seed)
        initial = task.initial_position(seed)
        if seed == 7:
            trace = ExecutabilityTrace(f"{task.task_id}:seed:{seed}")
            return SemanticTrialResult(
                success=False,
                object_position_error_m=None,
                failure_reason="mujoco_collision",
                rollout={"execution_status": "aborted"},
                initial_joint_position=(0.0,) * ACTION_DIM,
                final_joint_position=None,
                initial_object_position=initial,
                final_object_position=None,
                status=SemanticTrialStatus.EXECUTION_FAILURE,
                failure_evidence={
                    "error_type": "MujocoCollisionError",
                    "phase": "target preflight",
                    "contacts": [],
                },
                executability_trace=trace.snapshot(),
            )
        final = task.target_object_position if seed == 29 else initial
        outcome = score_object_position(task, final)
        return SemanticTrialResult(
            success=outcome.success,
            object_position_error_m=outcome.object_position_error_m,
            failure_reason=outcome.failure_reason,
            rollout={"execution_status": "complete"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=initial,
            final_object_position=final,
            executability_trace=_scored_trace(task, seed),
        )

    artifact_dir = tmp_path / "recoverable-artifacts"
    report = run_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=_manifest(tmp_path),
        artifact_dir=artifact_dir,
        trial_runner=fake_runner,
    )

    assert calls == [7, 13, 29]
    assert report["result"] == "fail"
    assert report["summary"]["trial_count"] == 3
    assert report["summary"]["scored_trial_count"] == 2
    assert report["summary"]["execution_failure_count"] == 1
    assert report["summary"]["executability_trace_trial_count"] == 3
    assert report["summary"]["executability_policy_step_count"] == 2
    assert report["summary"]["executability_servo_step_count"] == 2
    assert report["summary"]["success_count"] == 1
    assert report["summary"]["object_position_error_mean_m"] == pytest.approx(0.0525)
    assert report["summary"]["object_position_error_max_m"] == pytest.approx(0.105)
    assert report["summary"]["failure_counts"] == {
        "mujoco_collision": 1,
        "object_body_position_tolerance": 1,
    }
    failed = report["trials"][0]
    assert failed["status"] == "execution_failure"
    assert failed["object_position_error_m"] is None
    artifact = json.loads((artifact_dir / failed["artifact"]).read_text(encoding="utf-8"))
    trace_path = artifact_dir / failed["executability_trace_artifact"]
    assert artifact["final_joint_position"] is None
    assert artifact["final_object_position"] is None
    assert trace_path.is_file()
    assert failed["executability_trace_sha256"] == file_sha256(trace_path)


def test_semantic_benchmark_rejects_unknown_execution_failure(tmp_path: Path) -> None:
    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason="unexpected_failure",
            rollout={"execution_status": "aborted"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=None,
            initial_object_position=task.initial_position(seed),
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence={"message": "unexpected"},
        )

    with pytest.raises(MujocoSemanticBenchmarkError, match="recoverable failure"):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=tmp_path / "invalid-failure-artifacts",
            trial_runner=fake_runner,
        )


def test_semantic_benchmark_marks_unobserved_terminal_criterion(tmp_path: Path) -> None:
    def fake_runner(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
    ) -> SemanticTrialResult:
        del config
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason="policy_model_error",
            rollout={"execution_status": "aborted"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=None,
            initial_object_position=task.initial_position(seed),
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence={
                "error_type": "PolicyError",
                "message": "model output failed",
            },
        )

    artifact_dir = tmp_path / "unobserved-artifacts"
    report = run_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=_manifest(tmp_path),
        artifact_dir=artifact_dir,
        trial_runner=fake_runner,
    )

    assert report["semantic_mujoco_object_state_evaluated"] is False
    assert report["summary"]["scored_trial_count"] == 0
    assert report["summary"]["object_position_error_mean_m"] is None
    for trial in report["trials"]:
        artifact = json.loads((artifact_dir / trial["artifact"]).read_text(encoding="utf-8"))
        assert artifact["semantic_mujoco_object_state_evaluated"] is False


def test_checkpoint_benchmark_binds_candidate_and_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="semantic-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"policy_steps": task.policy_steps, "final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=_scored_trace(task, seed),
            mujoco_model_identity=_model_identity(),
            model_identity_source="rollout_session",
        )

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)

    report = run_checkpoint_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=_manifest(tmp_path),
        artifact_dir=tmp_path / "checkpoint-artifacts",
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert report["gate"] == CHECKPOINT_SEMANTIC_GATE
    assert report["policy"] == CHECKPOINT_SEMANTIC_POLICY
    assert report["benchmark_scope"] == CHECKPOINT_SEMANTIC_SCOPE
    assert report["checkpoint_sha256"] == file_sha256(checkpoint)
    assert report["checkpoint_id"] == "semantic-candidate"
    assert report["training_report_sha256"] == file_sha256(training_report)
    assert report["prompt_npz_sha256"] == file_sha256(prompt)
    assert report["prompt_manifest_sha256"] == file_sha256(prompt_manifest)
    assert report["checkpoint_task_split"] == "training_report_verified"
    assert report["prompt_task_match"] == "dataset_identity_verified"
    assert report["task_disjoint_basis"] == "checkpoint_bound_task_inventory"
    assert report["semantic_task_mapping"] == "manifest_declared"
    assert report["semantic_heldout_success_claimed"] is False
    assert report["mujoco_model_identity"] == _model_identity()
    assert report["model_identity_sources"] == ["rollout_session"]
    assert report["summary"]["model_identity_sources"] == ["rollout_session"]
    assert {trial["mujoco_model_identity"]["compiled_model_sha256"] for trial in report["trials"]} == {"f" * 64}
    assert {trial["model_identity_source"] for trial in report["trials"]} == {
        "rollout_session"
    }


def test_checkpoint_benchmark_profiles_execution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="profiled-failure-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    profile_reads: list[str] = []

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason="safety_watchdog",
            rollout={"execution_status": "aborted"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=None,
            initial_object_position=task.initial_position(seed),
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence={
                "error_type": "SafetyRejectedError",
                "reasons": ["stale_action"],
            },
        )

    def fake_scene(
        config: ProjectConfig,
        *,
        body_name: str,
    ) -> tuple[dict[str, object], str, dict[str, object]]:
        del config
        profile_reads.append(body_name)
        return OBJECT_PROFILE, OBJECT_PROFILE_SHA256, _model_identity()

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)
    monkeypatch.setattr(
        semantic_module,
        "inspect_mujoco_scene",
        fake_scene,
        raising=False,
    )

    report = run_checkpoint_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=_manifest(tmp_path),
        artifact_dir=tmp_path / "profiled-failure-artifacts",
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert profile_reads == ["task_block"]
    assert report["object_physical_profile"] == OBJECT_PROFILE
    assert report["object_physical_profile_sha256"] == OBJECT_PROFILE_SHA256
    assert {
        trial["object_physical_profile_sha256"] for trial in report["trials"]
    } == {OBJECT_PROFILE_SHA256}
    assert report["mujoco_model_identity"] == _model_identity()
    assert report["model_identity_sources"] == ["post_failure_inspection"]
    assert {
        trial["model_identity_source"] for trial in report["trials"]
    } == {"post_failure_inspection"}


def test_failure_keeps_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="failure-rollout-identity-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    profile_reads: list[str] = []

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason="safety_watchdog",
            rollout={"execution_status": "aborted"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=None,
            initial_object_position=task.initial_position(seed),
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence={
                "error_type": "SafetyRejectedError",
                "reasons": ["stale_action"],
            },
            mujoco_model_identity=_model_identity(),
            model_identity_source="rollout_session",
        )

    def fake_scene(
        config: ProjectConfig,
        *,
        body_name: str,
    ) -> tuple[dict[str, object], str, dict[str, object]]:
        del config
        profile_reads.append(body_name)
        return OBJECT_PROFILE, OBJECT_PROFILE_SHA256, _model_identity()

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)
    monkeypatch.setattr(semantic_module, "inspect_mujoco_scene", fake_scene)

    report = run_checkpoint_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=_manifest(tmp_path),
        artifact_dir=tmp_path / "failure-rollout-identity-artifacts",
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert profile_reads == ["task_block"]
    assert report["mujoco_model_identity"] == _model_identity()
    assert report["model_identity_sources"] == ["rollout_session"]


def test_failure_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="failure-inspected-drift-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason="safety_watchdog",
            rollout={"execution_status": "aborted"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=None,
            initial_object_position=task.initial_position(seed),
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence={
                "error_type": "SafetyRejectedError",
                "reasons": ["stale_action"],
            },
            mujoco_model_identity=_model_identity(),
            model_identity_source="rollout_session",
        )

    def fake_scene(
        config: ProjectConfig,
        *,
        body_name: str,
    ) -> tuple[dict[str, object], str, dict[str, object]]:
        del config, body_name
        return OBJECT_PROFILE, OBJECT_PROFILE_SHA256, _model_identity("e" * 64)

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)
    monkeypatch.setattr(semantic_module, "inspect_mujoco_scene", fake_scene)

    with pytest.raises(MujocoSemanticBenchmarkError, match="inspected model identity"):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=tmp_path / "failure-inspected-drift-artifacts",
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            prompt_path=prompt,
            prompt_manifest_path=prompt_manifest,
        )


def test_failure_partial_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="source-only-identity-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        return SemanticTrialResult(
            success=False,
            object_position_error_m=None,
            failure_reason="safety_watchdog",
            rollout={"execution_status": "aborted"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=None,
            initial_object_position=task.initial_position(seed),
            final_object_position=None,
            status=SemanticTrialStatus.EXECUTION_FAILURE,
            failure_evidence={
                "error_type": "SafetyRejectedError",
                "reasons": ["stale_action"],
            },
            object_physical_profile=OBJECT_PROFILE,
            object_physical_profile_sha256=OBJECT_PROFILE_SHA256,
            model_identity_source="rollout_session",
        )

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)

    with pytest.raises(MujocoSemanticBenchmarkError, match="source requires"):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=tmp_path / "source-only-identity-artifacts",
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            prompt_path=prompt,
            prompt_manifest_path=prompt_manifest,
        )


def test_missing_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="missing-identity-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=_scored_trace(task, seed),
        )

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)

    with pytest.raises(MujocoSemanticBenchmarkError, match="model identity"):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=tmp_path / "missing-identity-artifacts",
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            prompt_path=prompt,
            prompt_manifest_path=prompt_manifest,
        )


def test_mixed_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="mixed-identity-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        identity = _model_identity("e" * 64 if seed == 13 else "f" * 64)
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=_scored_trace(task, seed),
            mujoco_model_identity=identity,
            model_identity_source="rollout_session",
        )

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)

    with pytest.raises(MujocoSemanticBenchmarkError, match="mixed MuJoCo model identity"):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=tmp_path / "mixed-identity-artifacts",
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            prompt_path=prompt,
            prompt_manifest_path=prompt_manifest,
        )


def test_checkpoint_benchmark_runs_and_hashes_frozen_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="frozen-input-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    checkpoint_bytes = checkpoint.read_bytes()
    training_report_bytes = training_report.read_bytes()
    prompt_bytes = prompt.read_bytes()
    prompt_manifest_bytes = prompt_manifest.read_bytes()
    real_bundle_loader = semantic_module.load_compact_wam_bundle
    real_episode_loader = semantic_module.load_episode

    def mutate_checkpoint(path: str | Path, *, device: str) -> object:
        bundle = real_bundle_loader(path, device=device)
        checkpoint.write_bytes(b"mutated-checkpoint")
        training_report.write_text("{}", encoding="utf-8")
        return bundle

    def mutate_prompt(
        npz_path: str | Path,
        manifest_path: str | Path,
    ) -> object:
        episode = real_episode_loader(npz_path, manifest_path)
        prompt.write_bytes(b"mutated-prompt")
        prompt_manifest.write_text("{}", encoding="utf-8")
        return episode

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, device
        assert Path(checkpoint_path).read_bytes() == checkpoint_bytes
        assert Path(prompt_path).read_bytes() == prompt_bytes
        assert prompt_manifest_path is not None
        assert Path(prompt_manifest_path).read_bytes() == prompt_manifest_bytes
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=_scored_trace(task, seed),
            mujoco_model_identity=_model_identity(),
            model_identity_source="rollout_session",
        )

    monkeypatch.setattr(semantic_module, "load_compact_wam_bundle", mutate_checkpoint)
    monkeypatch.setattr(semantic_module, "load_episode", mutate_prompt)
    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)

    report = run_checkpoint_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=_manifest(tmp_path),
        artifact_dir=tmp_path / "frozen-artifacts",
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert report["checkpoint_sha256"] == sha256(checkpoint_bytes).hexdigest()
    assert report["training_report_sha256"] == sha256(training_report_bytes).hexdigest()
    assert report["prompt_npz_sha256"] == sha256(prompt_bytes).hexdigest()
    assert report["prompt_manifest_sha256"] == sha256(prompt_manifest_bytes).hexdigest()


def test_checkpoint_benchmark_rejects_prompt_task_mismatch_before_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="prompt-mismatch-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(
        tmp_path,
        task="wrong-task",
        task_index=999,
    )

    def unexpected_trial(*args: object, **kwargs: object) -> SemanticTrialResult:
        del args, kwargs
        raise AssertionError("task mismatch must fail before trials")

    monkeypatch.setattr(
        semantic_module,
        "run_checkpoint_task_trial",
        unexpected_trial,
    )
    artifact_dir = tmp_path / "prompt-mismatch-artifacts"

    with pytest.raises(MujocoSemanticBenchmarkError, match="prompt task identity"):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=artifact_dir,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            prompt_path=prompt,
            prompt_manifest_path=prompt_manifest,
        )

    assert not artifact_dir.exists()


def test_checkpoint_benchmark_runs_expected_prompt_task_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="expected-mismatch-candidate",
    )
    prompt, prompt_manifest = _prompt_episode(
        tmp_path,
        task="wrong-task",
        task_index=999,
    )

    def fake_trial(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> SemanticTrialResult:
        del config, checkpoint_path, prompt_path, prompt_manifest_path, device
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            executability_trace=_scored_trace(task, seed),
            mujoco_model_identity=_model_identity(),
            model_identity_source="rollout_session",
        )

    monkeypatch.setattr(semantic_module, "run_checkpoint_task_trial", fake_trial)

    report = run_checkpoint_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=_manifest(tmp_path),
        artifact_dir=tmp_path / "expected-mismatch-artifacts",
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
        prompt_task_expectation=PromptTaskExpectation.EXPECTED_MISMATCH,
    )

    assert report["prompt_task"] == "wrong-task"
    assert report["prompt_task_index"] == 999
    assert report["prompt_task_expectation"] == "expected_mismatch"
    assert report["prompt_task_identity_matches_manifest"] is False
    assert report["prompt_task_match"] == "control_mismatch_verified"
    assert report["semantic_heldout_success_claimed"] is False


def test_checkpoint_benchmark_rejects_unheldout_training_inventory(
    tmp_path: Path,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="split-mismatch-candidate",
        validation_tasks=[{"task_index": 301, "task": "different-validation-task"}],
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    artifact_dir = tmp_path / "split-mismatch-artifacts"

    with pytest.raises(
        MujocoSemanticBenchmarkError,
        match="checkpoint validation task inventory",
    ):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=artifact_dir,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            prompt_path=prompt,
            prompt_manifest_path=prompt_manifest,
        )

    assert not artifact_dir.exists()


def test_checkpoint_benchmark_rejects_training_report_core_tamper(
    tmp_path: Path,
) -> None:
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id="report-tamper-candidate",
    )
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    payload["data"]["validation_tasks"][0]["task"] = "forged-task"
    training_report.write_text(json.dumps(payload), encoding="utf-8")
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    artifact_dir = tmp_path / "report-tamper-artifacts"

    with pytest.raises(MujocoSemanticBenchmarkError, match="core digest"):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=artifact_dir,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            prompt_path=prompt,
            prompt_manifest_path=prompt_manifest,
        )

    assert not artifact_dir.exists()


def test_checkpoint_benchmark_rejects_one_prompt_for_multiple_tasks(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    second_task = dict(payload["heldout_tasks"][0])
    second_task["task_id"] = "heldout_block_shift"
    payload["heldout_tasks"].append(second_task)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MujocoSemanticBenchmarkError, match="exactly one held-out task"):
        run_checkpoint_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=manifest_path,
            artifact_dir=tmp_path / "multiple-task-artifacts",
            checkpoint_path=tmp_path / "missing.pt",
            training_report_path=tmp_path / "missing.training.json",
            prompt_path=tmp_path / "missing.npz",
        )


@requires_mujoco
@pytest.mark.parametrize(
    ("object_body", "initial", "geom_type"),
    (
        ("task_block", (0.34, 0.0, 0.475), "box"),
        ("task_cylinder", (0.36, 0.10, 0.49), "cylinder"),
    ),
)
def test_checkpoint_trial_reads_real_terminal_object_state(
    tmp_path: Path,
    object_body: str,
    initial: tuple[float, float, float],
    geom_type: str,
) -> None:
    config = ProjectConfig.load(Path("configs/mujoco_robot_free.toml"))
    config = replace(
        config,
        mujoco=replace(config.mujoco, camera_width=64, camera_height=48),
    )
    checkpoint = tmp_path / "candidate.pt"
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    model = CompactWAM(latent_dim=8, transformer_layers=1, transformer_heads=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    axis_mean = torch.zeros(ACTION_DIM)
    axis_mean[5] = 1.0
    axis_mean[11] = 1.0
    model.set_axis_normalization(axis_mean, torch.ones(ACTION_DIM))
    save_compact_wam_checkpoint(
        model,
        checkpoint,
        metadata={"checkpoint_id": "object-state-integration-test"},
    )
    task = SemanticTask(
        task_id=f"{object_body}_stability_smoke",
        label=f"{object_body} stability smoke",
        policy_steps=2,
        seeds=(7, 13, 29),
        object_body=object_body,
        initial_object_positions=((7, initial), (13, initial), (29, initial)),
        target_object_position=initial,
        position_tolerance_m=0.03,
    )

    result = run_checkpoint_task_trial(
        config,
        task,
        seed=7,
        checkpoint_path=checkpoint,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert result.success is True
    assert np.isfinite(result.final_object_position).all()
    assert result.object_position_error_m < task.position_tolerance_m
    assert result.rollout["sent_actions"] > 0
    assert result.object_physical_profile is not None
    assert result.object_physical_profile["geom"]["type"] == geom_type
    assert result.object_physical_profile_sha256 == canonical_json_sha256(
        result.object_physical_profile
    )
    assert result.executability_trace is not None
    assert result.executability_trace["policy_steps"] == task.policy_steps
    assert result.executability_trace["servo_steps"] > 0
    assert set(result.executability_trace["joint_limit_margins"]) == {
        "decoded_policy_target_min",
        "servo_safe_target_min",
        "measured_joint_min",
        "executed_action_min",
    }
    assert all(
        value >= 0.0
        for value in result.executability_trace["joint_limit_margins"].values()
    )
    assert result.executability_trace["contact_progression_count"] == (
        result.executability_trace["servo_steps"]
    )
    assert result.executability_trace["forbidden_contact_observation_count"] == 0
    assert result.executability_trace["future_latent_prediction_count"] == 2
    assert result.executability_trace["future_latent_observation_count"] == 2
    assert result.executability_trace["future_latent_aligned_pair_count"] == 1
    assert result.executability_trace["future_latent_mse_mean"] is not None
    assert result.executability_trace["runtime_future_latent_proxy_available"] is True
    assert result.executability_trace["runtime_future_latent_success_claimed"] is False


@requires_mujoco
@pytest.mark.parametrize(
    ("object_body", "initial", "geom_type"),
    (
        ("task_block", (0.34, 0.0, 0.475), "box"),
        ("task_cylinder", (0.36, 0.10, 0.49), "cylinder"),
    ),
)
def test_checkpoint_benchmark_binds_real_named_object_profile(
    tmp_path: Path,
    object_body: str,
    initial: tuple[float, float, float],
    geom_type: str,
) -> None:
    config = ProjectConfig.load(Path("configs/mujoco_robot_free.toml"))
    config = replace(
        config,
        mujoco=replace(config.mujoco, camera_width=64, camera_height=48),
    )
    checkpoint, training_report = _candidate_with_training_report(
        tmp_path,
        checkpoint_id=f"{object_body}-benchmark-integration",
    )
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    task = payload["heldout_tasks"][0]
    task["task_id"] = f"heldout_{object_body}_stability"
    task["label"] = f"held-out {object_body} stability"
    task["seeds"] = [7, 13, 29]
    task["object_body"] = object_body
    task["initial_object_positions"] = [
        {"seed": seed, "position": list(initial)}
        for seed in (7, 13, 29)
    ]
    task["target_object_position"] = list(initial)
    task["position_tolerance_m"] = 0.05
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = run_checkpoint_benchmark(
        config,
        manifest_path=manifest,
        artifact_dir=tmp_path / f"{object_body}-artifacts",
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert report["object_body"] == object_body
    assert report["summary"]["trial_count"] == 3
    assert report["checkpoint_sha256"] == file_sha256(checkpoint)
    assert report["training_report_sha256"] == file_sha256(training_report)
    assert report["prompt_npz_sha256"] == file_sha256(prompt)
    assert report["prompt_manifest_sha256"] == file_sha256(prompt_manifest)
    assert report["checkpoint_task_split"] == "training_report_verified"
    assert report["prompt_task_match"] == "dataset_identity_verified"
    assert report["object_physical_profile"]["geom"]["type"] == geom_type
    assert report["object_physical_profile_sha256"] == canonical_json_sha256(
        report["object_physical_profile"]
    )
    assert {trial["object_body"] for trial in report["trials"]} == {object_body}
    assert {
        trial["object_physical_profile_sha256"] for trial in report["trials"]
    } == {report["object_physical_profile_sha256"]}


@requires_mujoco
def test_checkpoint_trial_records_real_collision_event(tmp_path: Path) -> None:
    config = ProjectConfig.load(Path("configs/mujoco_robot_free.toml"))
    prompt, prompt_manifest = _prompt_episode(
        tmp_path,
        image_height=config.mujoco.camera_height,
        image_width=config.mujoco.camera_width,
    )
    checkpoint = _constant_target_checkpoint(
        tmp_path,
        checkpoint_id="collision-event-integration",
        target=COLLISION_TARGET,
    )
    task = replace(
        _task(),
        task_id="collision_event",
        label="collision event",
        policy_steps=COLLISION_POLICY_STEPS,
    )

    result = run_checkpoint_task_trial(
        config,
        task,
        seed=7,
        checkpoint_path=checkpoint,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert result.status is SemanticTrialStatus.EXECUTION_FAILURE
    assert result.failure_reason == "mujoco_collision"
    assert result.failure_evidence is not None
    assert result.failure_evidence["phase"] == "target preflight"
    contacts = result.failure_evidence["contacts"]
    assert isinstance(contacts, list) and contacts
    assert any(
        {contact["category1"], contact["category2"]}
        == {"torso", "right_arm"}
        for contact in contacts
    )
    assert result.final_joint_position is None
    assert result.final_object_position is None
    assert result.executability_trace is not None
    assert result.executability_trace["rollout_id"] == "collision_event:seed:7"
    assert result.executability_trace["collision_failure_phase"] == (
        "target preflight"
    )
    assert result.executability_trace["collision_failure_contacts"] == contacts
    assert result.executability_trace["contact_progression_count"] >= 1


@requires_mujoco
def test_checkpoint_trial_records_real_watchdog_event(tmp_path: Path) -> None:
    base = ProjectConfig.load(Path("configs/mujoco_robot_free.toml"))
    config = replace(
        base,
        runtime=replace(base.runtime, policy_hz=WATCHDOG_POLICY_HZ),
    )
    prompt, prompt_manifest = _prompt_episode(
        tmp_path,
        image_height=config.mujoco.camera_height,
        image_width=config.mujoco.camera_width,
    )
    hold_target = (0.0,) * 5 + (1.0,) + (0.0,) * 5 + (1.0,)
    checkpoint = _constant_target_checkpoint(
        tmp_path,
        checkpoint_id="watchdog-event-integration",
        target=hold_target,
    )
    task = replace(
        _task(),
        task_id="watchdog_event",
        label="watchdog event",
    )

    result = run_checkpoint_task_trial(
        config,
        task,
        seed=7,
        checkpoint_path=checkpoint,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert result.status is SemanticTrialStatus.EXECUTION_FAILURE
    assert result.failure_reason == "safety_watchdog"
    assert result.failure_evidence == {
        "error_type": "SafetyRejectedError",
        "reasons": ["stale_action"],
    }
    assert result.final_joint_position is None
    assert result.final_object_position is None
    assert result.executability_trace is not None
    assert result.executability_trace["safety_rejected_steps"] == 1
    assert result.executability_trace["safety_reason_counts"] == {"stale_action": 1}
