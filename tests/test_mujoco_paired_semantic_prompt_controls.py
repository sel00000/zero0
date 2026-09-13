from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import so101_wam.mujoco_semantic_benchmark as semantic_module
import so101_wam.mujoco_cli as mujoco_cli_module
import so101_wam.mujoco_paired_semantic_prompt_controls as paired_controls_module
from so101_wam.checkpoint import (
    compact_wam_architecture,
    save_compact_wam_checkpoint,
)
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.deployment import canonical_json_sha256, file_sha256
from so101_wam.model import CompactWAM
from so101_wam.mujoco_paired_semantic_prompt_controls import (
    FUTURE_LATENT_OUTCOME_CLASSIFIER,
    MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_GATE,
    MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_SCOPE,
    PairedSemanticPromptCondition,
    PairedSemanticPromptControlError,
    main as paired_semantic_main,
    run_paired_semantic_prompt_controls,
)
from so101_wam.mujoco_semantic_benchmark import (
    TASK_SPEC_CHECKPOINT_SEMANTIC_GATE,
    TASK_SPEC_CHECKPOINT_SEMANTIC_POLICY,
    TASK_SPEC_CHECKPOINT_SEMANTIC_SCOPE,
    TERMINAL_FAILURE_REASON,
    ModelIdentitySource,
    SemanticTask,
    SemanticTrialResult,
    load_semantic_manifest,
    run_task_spec_checkpoint_task_trial,
)
from so101_wam.paired_data import (
    HUMAN_TASK_SPEC_FORMAT,
    LEROBOT_V3_COMPATIBLE,
    PAIRED_DATA_KIND,
    PAIRED_DATA_SCHEMA,
    SemanticMatchStatus,
    load_human_robot_pairs,
    paired_data_audit,
)
from so101_wam.rollout import ExecutabilityTrace
from so101_wam.training import TRAINING_REPORT_SCHEMA
from so101_wam.paired_task_specs import human_prompt_from_pair


ROOT = Path(__file__).resolve().parents[1]
DATASET_TASK = "synthetic-validation-place"
DATASET_TASK_INDEX = 201
WRONG_TASK = "synthetic-control-reach"
WRONG_TASK_INDEX = 202
CONDITIONS = tuple(condition.value for condition in PairedSemanticPromptCondition)
OBJECT_PROFILE = {"schema_version": 1, "body": "task_block"}
OBJECT_PROFILE_SHA256 = canonical_json_sha256(OBJECT_PROFILE)
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
IMAGE_HEIGHT = 24
IMAGE_WIDTH = 32
TRACE_LOWER = (-10.0,) * ACTION_DIM
TRACE_UPPER = (10.0,) * ACTION_DIM
MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
requires_mujoco = pytest.mark.skipif(
    not MUJOCO_AVAILABLE,
    reason="optional mujoco dependency is absent",
)


def _scored_trace(
    task: SemanticTask,
    seed: int,
    *,
    future_value: float = 1.0,
) -> dict[str, object]:
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
            future_latents=np.full(
                (1, 2, 3),
                future_value,
                dtype=np.float32,
            ),
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


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _pair_item(
    root: Path,
    *,
    pair_id: str,
    task: str,
    task_index: int,
    episode_index: int,
    value: int,
    semantic_match: str = SemanticMatchStatus.HUMAN_REVIEWED.value,
    include_task_spec: bool = True,
) -> dict[str, object]:
    episode = EpisodeBuffer(
        fps=30.0,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
        metadata={"action_source": "imported_lerobot_v3_action"},
    )
    for frame_index in range(91):
        timestamp_s = frame_index / 30.0
        axes = np.full(ACTION_DIM, value + frame_index, dtype=np.float32)
        episode.append(
            SensorimotorFrame(
                timestamp_s=timestamp_s,
                images={
                    "left_wrist": np.full(
                        (IMAGE_HEIGHT, IMAGE_WIDTH, 3), value, dtype=np.uint8
                    ),
                    "right_wrist": np.full(
                        (IMAGE_HEIGHT, IMAGE_WIDTH, 3), value + 1, dtype=np.uint8
                    ),
                },
                joint_position=axes,
                executed_action=axes,
            )
        )
    robot_path, _ = episode.save(root / "episodes", stem=pair_id)

    human_dir = root / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    video_path = human_dir / f"{pair_id}.mp4"
    video_path.write_bytes(f"human-video:{pair_id}".encode())
    task_spec_path = human_dir / f"{pair_id}.task_spec.npz"
    np.savez_compressed(
        task_spec_path,
        timestamp=np.array([0.0, 3.0], dtype=np.float64),
        rgb=np.stack(
            (
                np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), value, dtype=np.uint8),
                np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), value + 1, dtype=np.uint8),
            )
        ),
    )

    human_video: dict[str, object] = {
        "path": video_path.relative_to(root).as_posix(),
        "sha256": _sha256(video_path),
        "view": "third_person",
    }
    if include_task_spec:
        human_video["task_spec"] = {
            "format": HUMAN_TASK_SPEC_FORMAT,
            "path": task_spec_path.relative_to(root).as_posix(),
            "sha256": _sha256(task_spec_path),
        }
    return {
        "pair_id": pair_id,
        "task": task,
        "task_index": task_index,
        "semantic_match": semantic_match,
        "robot_episode": robot_path.relative_to(root).as_posix(),
        "robot_episode_sha256": _sha256(robot_path),
        "human_video": human_video,
        "provenance": {
            "source_dataset": "example/repo",
            "source_layout": "data/,videos/",
            "license": "apache-2.0",
            "transformation": "human RGB extraction and robot conversion",
        },
    }


def _pair_manifest(
    tmp_path: Path,
    *,
    matched_task: str = DATASET_TASK,
    matched_task_index: int = DATASET_TASK_INDEX,
    wrong_task: str = WRONG_TASK,
    wrong_task_index: int = WRONG_TASK_INDEX,
    semantic_match: str = SemanticMatchStatus.HUMAN_REVIEWED.value,
    include_task_spec: bool = True,
) -> Path:
    root = tmp_path / "pair-bundle"
    root.mkdir()
    payload = {
        "schema_version": PAIRED_DATA_SCHEMA,
        "artifact_kind": PAIRED_DATA_KIND,
        "source_format": LEROBOT_V3_COMPATIBLE,
        "source_url": "hf://datasets/example/repo",
        "pairs": [
            _pair_item(
                root,
                pair_id="validation-matched",
                task=matched_task,
                task_index=matched_task_index,
                episode_index=1,
                value=3,
                semantic_match=semantic_match,
                include_task_spec=include_task_spec,
            ),
            _pair_item(
                root,
                pair_id="validation-wrong",
                task=wrong_task,
                task_index=wrong_task_index,
                episode_index=2,
                value=17,
            ),
        ],
    }
    path = root / "validation-pairs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _semantic_manifest(tmp_path: Path, *, policy_steps: int = 1) -> Path:
    path = tmp_path / "semantic-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "benchmark_id": "paired-human-object-state-v1",
                "dataset_task": {
                    "task_index": DATASET_TASK_INDEX,
                    "task": DATASET_TASK,
                },
                "train_task_ids": ["synthetic-train-home"],
                "heldout_tasks": [
                    {
                        "task_id": "heldout-block-lift",
                        "label": "held-out block lift",
                        "policy_steps": policy_steps,
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


def _candidate(
    tmp_path: Path,
    pair_manifest: Path,
) -> tuple[Path, Path]:
    pairs = load_human_robot_pairs(pair_manifest)
    pair_digest = str(paired_data_audit(pairs)["pair_digest"])
    validation_pairs = [
        {
            "pair_id": pair.pair_id,
            "pair_fingerprint": pair.fingerprint,
            "task": pair.task,
            "task_index": pair.task_index,
            "semantic_match": pair.semantic_match.value,
            "human_video_sha256": pair.human_video_sha256,
            "human_task_spec_sha256": (
                pair.human_task_spec.sha256 if pair.human_task_spec else None
            ),
            "robot_episode_sha256": pair.robot_episode_sha256,
            "robot_episode_fingerprint": pair.robot_episode.fingerprint,
        }
        for pair in sorted(pairs, key=lambda item: item.pair_id)
    ]
    model = CompactWAM(
        latent_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        future_steps=1,
        action_horizon=10,
        action_history_steps=1,
        ifp_steps=0,
    )
    report_core = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "evidence_level": "offline",
        "artifact_kind": "compact_wam_candidate",
        "result": "pass",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": "paired-semantic-candidate",
        "protocol": {
            "split": "task_disjoint",
            "prompt_pairing": "human_video_reviewed_pair",
            "prompt_modality": "human_video_task_spec",
            "real_output_authorized": False,
        },
        "optimization": {
            "policy_hz": 10.0,
            "servo_hz": 50.0,
        },
        "data": {
            "train_task_count": 1,
            "validation_task_count": 2,
            "train_tasks": [{"task_index": 101, "task": "synthetic-train-home"}],
            "validation_tasks": [
                {"task_index": DATASET_TASK_INDEX, "task": DATASET_TASK},
                {"task_index": WRONG_TASK_INDEX, "task": WRONG_TASK},
            ],
            "validation_pair_count": len(pairs),
            "validation_pair_digest": pair_digest,
            "validation_pairs": validation_pairs,
        },
        "model": compact_wam_architecture(model),
    }
    training_evidence = canonical_json_sha256(report_core)
    checkpoint = tmp_path / "paired-semantic-candidate.pt"
    save_compact_wam_checkpoint(
        model,
        checkpoint,
        metadata={
            "artifact_kind": "compact_wam_candidate",
            "evidence_level": "offline",
            "checkpoint_id": "paired-semantic-candidate",
            "trained": False,
            "offline_trained": True,
            "deployment_ready": False,
            "paired_human_video": True,
            "training_evidence_sha256": training_evidence,
        },
    )
    report = {
        **report_core,
        "training_evidence_sha256": training_evidence,
        "artifacts": {"checkpoint_sha256": file_sha256(checkpoint)},
    }
    report_path = tmp_path / "paired-semantic-candidate.training.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return checkpoint, report_path


def _fake_task_trial(
    calls: list[tuple[str, str]],
):
    def run(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        task_spec: object,
        device: str = "cpu",
    ) -> SemanticTrialResult:
        del config, checkpoint_path, device
        fingerprint = str(getattr(task_spec, "fingerprint"))
        text_metadata = str(getattr(task_spec, "text_metadata"))
        calls.append((fingerprint, text_metadata))
        return SemanticTrialResult(
            success=True,
            object_position_error_m=0.0,
            failure_reason=None,
            rollout={"policy_steps": task.policy_steps},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=task.initial_position(seed),
            final_object_position=task.target_object_position,
            object_physical_profile=OBJECT_PROFILE,
            object_physical_profile_sha256=OBJECT_PROFILE_SHA256,
            mujoco_model_identity=MODEL_IDENTITY,
            model_identity_source=ModelIdentitySource.ROLLOUT_SESSION,
            executability_trace=_scored_trace(task, seed),
        )

    return run


def _fake_outcome_task_trial(
    calls: list[tuple[str, str]],
    *,
    wrong_future_value: float,
):
    def run(
        config: ProjectConfig,
        task: SemanticTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        task_spec: object,
        device: str = "cpu",
    ) -> SemanticTrialResult:
        del config, checkpoint_path, device
        fingerprint = str(getattr(task_spec, "fingerprint"))
        text_metadata = str(getattr(task_spec, "text_metadata"))
        calls.append((fingerprint, text_metadata))
        wrong_task = text_metadata == WRONG_TASK
        initial = task.initial_position(seed)
        final = initial if wrong_task else task.target_object_position
        error = float(
            np.linalg.norm(np.asarray(final) - np.asarray(task.target_object_position))
        )
        return SemanticTrialResult(
            success=not wrong_task,
            object_position_error_m=error,
            failure_reason=TERMINAL_FAILURE_REASON if wrong_task else None,
            rollout={"policy_steps": task.policy_steps},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=(0.0,) * ACTION_DIM,
            initial_object_position=initial,
            final_object_position=final,
            object_physical_profile=OBJECT_PROFILE,
            object_physical_profile_sha256=OBJECT_PROFILE_SHA256,
            mujoco_model_identity=MODEL_IDENTITY,
            model_identity_source=ModelIdentitySource.ROLLOUT_SESSION,
            executability_trace=_scored_trace(
                task,
                seed,
                future_value=wrong_future_value if wrong_task else 2.0,
            ),
        )

    return run


def _run_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pair_manifest: Path | None = None,
    policy_steps: int = 1,
    wrong_future_value: float | None = None,
) -> tuple[dict[str, object], list[tuple[str, str]], Path, Path, Path]:
    pair_manifest = pair_manifest or _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path, policy_steps=policy_steps)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    artifact_dir = tmp_path / "paired-semantic-artifacts"
    report_path = tmp_path / "paired-semantic-report.json"
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        (
            _fake_task_trial(calls)
            if wrong_future_value is None
            else _fake_outcome_task_trial(
                calls,
                wrong_future_value=wrong_future_value,
            )
        ),
    )

    report = run_paired_semantic_prompt_controls(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        validation_pair_manifest_path=pair_manifest,
        semantic_manifest_path=semantic_manifest,
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        artifact_dir=artifact_dir,
        report_path=report_path,
    )
    return report, calls, artifact_dir, report_path, training_report


def _rewrite_nested_identity(
    report: dict[str, object],
    artifact_dir: Path,
    identity: dict[str, object],
    source: ModelIdentitySource = ModelIdentitySource.ROLLOUT_SESSION,
) -> None:
    report["mujoco_model_identity"] = identity
    report["model_identity_sources"] = [source.value]
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["mujoco_model_identity"] = identity
    summary["model_identity_sources"] = [source.value]
    tasks = report["tasks"]
    assert isinstance(tasks, list)
    for task in tasks:
        task["mujoco_model_identity"] = identity
        task["model_identity_sources"] = [source.value]
    trials = report["trials"]
    assert isinstance(trials, list)
    for trial in trials:
        trial["mujoco_model_identity"] = identity
        trial["model_identity_source"] = source.value
        artifact_path = artifact_dir / str(trial["artifact"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["mujoco_model_identity"] = identity
        artifact["model_identity_source"] = source.value
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        trial["artifact_sha256"] = file_sha256(artifact_path)


def test_paired_semantic_controls_run_three_task_spec_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, calls, artifact_dir, report_path, _ = _run_bundle(
        tmp_path,
        monkeypatch,
    )

    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["gate"] == MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_GATE
    assert report["schema_version"] == 4
    assert report["scope"] == MUJOCO_PAIRED_SEMANTIC_PROMPT_CONTROL_SCOPE
    assert report["result"] == "complete"
    assert report["prompt_modality"] == "human_video_task_spec"
    assert report["paired_human_video_checkpoint"] is True
    assert report["robot_used"] is False
    assert report["future_visual_metric_available"] is False
    assert report["future_visual_success_claimed"] is False
    assert report["runtime_future_latent_proxy_available"] is False
    assert report["runtime_future_latent_success_claimed"] is False
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["real_world_success_claimed"] is False
    assert report["official_zero_wam_claimed"] is False
    assert report["mujoco_model_identity"] == MODEL_IDENTITY
    assert report["model_identity_sources"] == [ModelIdentitySource.ROLLOUT_SESSION.value]
    assert report["summary"] == {
        "condition_count": 3,
        "trials_per_condition": 3,
        "total_trial_count": 9,
        "scored_trial_count": 9,
        "execution_failure_count": 0,
        "executability_trace_trial_count": 9,
        "executability_policy_step_count": 9,
        "executability_servo_step_count": 9,
        "executability_sent_action_count": 9,
        "executability_shadow_step_count": 0,
        "executability_safety_accepted_count": 9,
        "executability_safety_rejected_count": 0,
        "executability_safety_clipped_count": 0,
        "executability_joint_limit_margins": {
            "decoded_policy_target_min": 0.5,
            "servo_safe_target_min": 0.5,
            "measured_joint_min": 0.5,
            "executed_action_min": 0.5,
        },
        "executability_contact_progression_count": 9,
        "executability_contact_observation_count": 0,
        "executability_forbidden_contact_observation_count": 0,
        "executability_object_contact_observation_count": 0,
        "executability_minimum_contact_distance_m": None,
        "executability_collision_failure_count": 0,
        "future_latent_trace_trial_count": 9,
        "future_latent_prediction_count": 9,
        "future_latent_observation_count": 9,
        "future_latent_aligned_pair_count": 0,
        "future_latent_censored_pair_count": 9,
        "future_latent_element_count": 0,
        "future_latent_squared_error_sum": 0.0,
        "future_latent_mse_mean": None,
        "future_latent_proxy_outcome_mismatch_count": 0,
        "mujoco_model_identity": MODEL_IDENTITY,
        "model_identity_sources": [ModelIdentitySource.ROLLOUT_SESSION.value],
        "matched_success_rate": 1.0,
        "matched_object_position_error_mean_m": 0.0,
    }

    conditions = report["conditions"]
    assert [item["condition"] for item in conditions] == list(CONDITIONS)
    assert [item["prompt_task_expectation"] for item in conditions] == [
        "match_manifest",
        "expected_mismatch",
        "match_manifest",
    ]
    assert [item["prompt_task_identity_matches_manifest"] for item in conditions] == [
        True,
        False,
        True,
    ]
    assert all(item["success_rate_delta_vs_matched"] == 0.0 for item in conditions)
    assert all(
        item["object_position_error_mean_delta_vs_matched_m"] == 0.0
        for item in conditions
    )
    assert all(item["future_latent_mse_delta_vs_matched"] is None for item in conditions)
    assert all(
        item["future_latent_proxy_outcome_mismatch"] is False
        for item in conditions
    )
    assert len(calls) == 9
    assert len({fingerprint for fingerprint, _ in calls}) == 3
    assert [metadata for _, metadata in calls[0::3]] == [
        DATASET_TASK,
        WRONG_TASK,
        DATASET_TASK,
    ]
    for condition in conditions:
        nested_path = artifact_dir / condition["benchmark_report"]
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        assert condition["benchmark_report_sha256"] == file_sha256(nested_path)
        assert nested["gate"] == TASK_SPEC_CHECKPOINT_SEMANTIC_GATE
        assert nested["policy"] == TASK_SPEC_CHECKPOINT_SEMANTIC_POLICY
        assert nested["benchmark_scope"] == TASK_SPEC_CHECKPOINT_SEMANTIC_SCOPE
        assert nested["runtime_placeholder_prompt_used_by_model"] is False
        assert nested["executability_trace_present"] is True
        assert nested["mujoco_model_identity"] == MODEL_IDENTITY
        assert nested["model_identity_sources"] == [
            ModelIdentitySource.ROLLOUT_SESSION.value
        ]
        assert nested["summary"]["mujoco_model_identity"] == MODEL_IDENTITY
        assert nested["summary"]["model_identity_sources"] == [
            ModelIdentitySource.ROLLOUT_SESSION.value
        ]
        assert nested["tasks"][0]["mujoco_model_identity"] == MODEL_IDENTITY
        assert nested["tasks"][0]["model_identity_sources"] == [
            ModelIdentitySource.ROLLOUT_SESSION.value
        ]
        assert condition["mujoco_model_identity"] == MODEL_IDENTITY
        assert condition["model_identity_sources"] == [
            ModelIdentitySource.ROLLOUT_SESSION.value
        ]
        assert nested["summary"]["executability_trace_trial_count"] == 3
        assert nested["summary"]["executability_contact_progression_count"] == 3
        for trial in nested["trials"]:
            assert trial["rollout_id"] == f"{trial['task_id']}:seed:{trial['seed']}"
            assert len(trial["executability_trace_sha256"]) == 64
            assert trial["mujoco_model_identity"] == MODEL_IDENTITY
            assert trial["model_identity_source"] == (
                ModelIdentitySource.ROLLOUT_SESSION.value
            )


def test_paired_semantic_controls_aggregate_future_latent_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _, _, _, _ = _run_bundle(
        tmp_path,
        monkeypatch,
        policy_steps=2,
    )

    assert report["future_visual_metric_available"] is False
    assert report["runtime_future_latent_proxy_available"] is True
    assert report["runtime_future_latent_success_claimed"] is False
    assert report["summary"]["future_latent_trace_trial_count"] == 9
    assert report["summary"]["future_latent_prediction_count"] == 18
    assert report["summary"]["future_latent_observation_count"] == 18
    assert report["summary"]["future_latent_aligned_pair_count"] == 9
    assert report["summary"]["future_latent_censored_pair_count"] == 9
    assert report["summary"]["future_latent_element_count"] == 54
    assert report["summary"]["future_latent_squared_error_sum"] == pytest.approx(
        54.0
    )
    assert report["summary"]["future_latent_mse_mean"] == pytest.approx(1.0)
    assert all(
        condition["future_latent_mse_mean"] == pytest.approx(1.0)
        for condition in report["conditions"]
    )
    assert report["summary"]["future_latent_proxy_outcome_mismatch_count"] == 0
    assert all(
        condition["future_latent_mse_delta_vs_matched"] == pytest.approx(0.0)
        for condition in report["conditions"]
    )
    assert all(
        condition["future_latent_proxy_outcome_mismatch"] is False
        for condition in report["conditions"]
    )


def test_paired_semantic_controls_flag_latent_outcome_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _, artifact_dir, _, _ = _run_bundle(
        tmp_path,
        monkeypatch,
        policy_steps=2,
        wrong_future_value=1.0,
    )

    conditions = {
        condition["condition"]: condition for condition in report["conditions"]
    }
    matched = conditions[PairedSemanticPromptCondition.MATCHED_HUMAN_VIDEO.value]
    wrong = conditions[PairedSemanticPromptCondition.WRONG_TASK_HUMAN_VIDEO.value]
    null = conditions[PairedSemanticPromptCondition.NULL_HUMAN_VIDEO.value]
    assert matched["future_latent_proxy_outcome_mismatch"] is False
    assert null["future_latent_proxy_outcome_mismatch"] is False
    assert wrong["future_latent_mse_delta_vs_matched"] == pytest.approx(-3.0)
    assert wrong["success_rate_delta_vs_matched"] == pytest.approx(-1.0)
    assert wrong["future_latent_proxy_outcome_mismatch"] is True
    assert report["future_latent_outcome_classifier"] == (
        FUTURE_LATENT_OUTCOME_CLASSIFIER
    )
    assert report["summary"]["future_latent_proxy_outcome_mismatch_count"] == 1
    assert report["runtime_future_latent_success_claimed"] is False
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["real_world_success_claimed"] is False
    nested_path = artifact_dir / wrong["benchmark_report"]
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    assert wrong["benchmark_report_sha256"] == file_sha256(nested_path)
    assert all(
        len(trial["artifact_sha256"]) == 64
        and len(trial["executability_trace_sha256"]) == 64
        for trial in nested["trials"]
    )


def test_paired_semantic_controls_do_not_flag_latent_mse_tie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _, _, _, _ = _run_bundle(
        tmp_path,
        monkeypatch,
        policy_steps=2,
        wrong_future_value=2.0,
    )

    wrong = next(
        condition
        for condition in report["conditions"]
        if condition["condition"]
        == PairedSemanticPromptCondition.WRONG_TASK_HUMAN_VIDEO.value
    )
    assert wrong["future_latent_mse_delta_vs_matched"] == pytest.approx(0.0)
    assert wrong["success_rate_delta_vs_matched"] == pytest.approx(-1.0)
    assert wrong["future_latent_proxy_outcome_mismatch"] is False
    assert report["summary"]["future_latent_proxy_outcome_mismatch_count"] == 0


@requires_mujoco
def test_paired_semantic_controls_run_real_mujoco_task_spec_sessions(
    tmp_path: Path,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)

    report = run_paired_semantic_prompt_controls(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        validation_pair_manifest_path=pair_manifest,
        semantic_manifest_path=_semantic_manifest(tmp_path),
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        artifact_dir=tmp_path / "real-artifacts",
        report_path=tmp_path / "real-report.json",
    )

    assert report["result"] == "complete"
    assert report["summary"]["total_trial_count"] == 9
    assert report["summary"]["executability_trace_trial_count"] == 9
    assert report["runtime_placeholder_prompt_used_by_model"] is False
    for condition in report["conditions"]:
        assert condition["runtime_placeholder_prompt_used_by_model"] is False
        assert condition["executability_trace_trial_count"] == 3


def test_paired_semantic_controls_bind_validation_manifest_and_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    expected_pairs = load_human_robot_pairs(pair_manifest)
    expected_digest = paired_data_audit(expected_pairs)["pair_digest"]
    report, _, _, _, training_report = _run_bundle(
        tmp_path,
        monkeypatch,
        pair_manifest=pair_manifest,
    )

    assert report["validation_pair_manifest_sha256"] == file_sha256(pair_manifest)
    assert report["validation_pair_digest"] == expected_digest
    assert report["training_report_sha256"] == file_sha256(training_report)
    matched = report["conditions"][0]
    pair = expected_pairs[0]
    assert matched["source_pair_id"] == pair.pair_id
    assert matched["source_pair_fingerprint"] == pair.fingerprint
    assert matched["human_video_sha256"] == pair.human_video_sha256
    assert matched["human_task_spec_sha256"] == pair.human_task_spec.sha256
    null = report["conditions"][2]
    assert null["source_pair_id"] is None
    assert null["source_pair_fingerprint"] is None
    assert null["derived_from_pair_fingerprint"] == pair.fingerprint


def test_paired_semantic_controls_reject_digest_mismatch_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    payload["data"]["validation_pair_digest"] = "0" * 64
    training_report.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="digest"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=artifact_dir,
            report_path=report_path,
        )

    assert calls == []
    assert not artifact_dir.exists()
    assert not report_path.exists()


@pytest.mark.parametrize(
    ("manifest_kwargs", "message"),
    (
        (
            {"semantic_match": SemanticMatchStatus.UNVERIFIED.value},
            "human_reviewed",
        ),
        ({"include_task_spec": False}, "task-spec"),
        (
            {
                "matched_task": "different-task",
                "matched_task_index": 301,
            },
            "matching semantic dataset task",
        ),
        (
            {
                "wrong_task": DATASET_TASK,
                "wrong_task_index": DATASET_TASK_INDEX,
            },
            "different validation task",
        ),
    ),
)
def test_paired_semantic_controls_reject_invalid_controls_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_kwargs: dict[str, object],
    message: str,
) -> None:
    pair_manifest = _pair_manifest(tmp_path, **manifest_kwargs)
    semantic_manifest = _semantic_manifest(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match=message):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=tmp_path / "unused.pt",
            training_report_path=tmp_path / "unused.json",
            artifact_dir=artifact_dir,
            report_path=report_path,
        )

    assert calls == []
    assert not artifact_dir.exists()
    assert not report_path.exists()


def test_paired_semantic_controls_do_not_overwrite_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    with pytest.raises(PairedSemanticPromptControlError, match="already exists"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=artifact_dir,
            report_path=report_path,
        )


def test_paired_semantic_controls_reject_camera_resolution_before_outputs(
    tmp_path: Path,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)

    with pytest.raises(
        PairedSemanticPromptControlError,
        match="resolution must match MuJoCo cameras",
    ):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=_semantic_manifest(tmp_path),
            checkpoint_path=tmp_path / "unused.pt",
            training_report_path=tmp_path / "unused.json",
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "report.json").exists()


def test_paired_semantic_controls_reject_runtime_rate_mismatch_before_outputs(
    tmp_path: Path,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    base_config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    config = replace(
        base_config,
        runtime=replace(base_config.runtime, policy_hz=5.0),
    )

    with pytest.raises(
        PairedSemanticPromptControlError,
        match="policy_hz does not match MuJoCo runtime",
    ):
        run_paired_semantic_prompt_controls(
            config,
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=_semantic_manifest(tmp_path),
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "report.json").exists()


def test_paired_semantic_controls_reject_nested_evidence_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        return {**report, "training_report_sha256": "0" * 64}

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(
        PairedSemanticPromptControlError,
        match="training_report_sha256 mismatch",
    ):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_missing_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        del report["mujoco_model_identity"]
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="model identity"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        if kwargs["prompt_condition"] == (
            PairedSemanticPromptCondition.WRONG_TASK_HUMAN_VIDEO.value
        ):
            _rewrite_nested_identity(
                report,
                Path(str(kwargs["artifact_dir"])),
                DRIFT_MODEL_IDENTITY,
            )
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="model identity"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 6
    assert not report_path.exists()


def test_paired_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        trial = report["trials"][0]
        artifact_path = Path(str(kwargs["artifact_dir"])) / str(trial["artifact"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
        artifact["model_identity_source"] = (
            ModelIdentitySource.POST_FAILURE_INSPECTION.value
        )
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        trial["artifact_sha256"] = file_sha256(artifact_path)
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="model identity"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_artifact_one_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    watched: set[Path] = set()
    reads: dict[Path, int] = {}
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark
    real_read_bytes = Path.read_bytes

    def racing_read_bytes(path: Path) -> bytes:
        source = real_read_bytes(path)
        key = path.resolve()
        if key not in watched:
            return source
        reads[key] = reads.get(key, 0) + 1
        if reads[key] == 1:
            return source
        payload = json.loads(source.decode("utf-8"))
        payload["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
        return json.dumps(payload).encode("utf-8")

    def watched_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        if not watched:
            artifact_dir = Path(str(kwargs["artifact_dir"]))
            watched.add((artifact_dir / str(report["trials"][0]["artifact"])).resolve())
        return report

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        watched_run,
    )

    report = run_paired_semantic_prompt_controls(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        validation_pair_manifest_path=pair_manifest,
        semantic_manifest_path=semantic_manifest,
        checkpoint_path=checkpoint,
        training_report_path=training_report,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    assert report["mujoco_model_identity"] == MODEL_IDENTITY
    assert list(reads.values()) == [1]


def test_paired_invalid_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        report["trials"][0]["model_identity_source"] = [
            ModelIdentitySource.ROLLOUT_SESSION.value
        ]
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="model identity source"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_summary_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        report["summary"]["mujoco_model_identity"] = DRIFT_MODEL_IDENTITY
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="summary model identity"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_summary_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        report["summary"]["model_identity_sources"] = [
            ModelIdentitySource.POST_FAILURE_INSPECTION.value
        ]
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="summary model identity"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_top_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        sources = [ModelIdentitySource.POST_FAILURE_INSPECTION.value]
        report["model_identity_sources"] = sources
        report["summary"]["model_identity_sources"] = sources
        for task in report["tasks"]:
            task["model_identity_sources"] = sources
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="identity sources"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_task_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        report["tasks"] = []
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(PairedSemanticPromptControlError, match="task summaries"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_semantic_controls_reject_trace_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        trial = report["trials"][0]
        trace_path = Path(str(kwargs["artifact_dir"])) / str(
            trial["executability_trace_artifact"]
        )
        trace_path.write_text("{}", encoding="utf-8")
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(
        PairedSemanticPromptControlError,
        match="executability trace hash mismatch",
    ):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_semantic_controls_recompute_joint_margin_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        margins = report["summary"]["executability_joint_limit_margins"]
        margins["executed_action_min"] = -1.0
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(
        PairedSemanticPromptControlError,
        match="executability_joint_limit_margins mismatch",
    ):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_semantic_controls_recompute_future_latent_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path, policy_steps=2)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        report["summary"]["future_latent_prediction_count"] -= 1
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(
        PairedSemanticPromptControlError,
        match="future_latent_prediction_count mismatch",
    ):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_semantic_controls_reject_empty_scored_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        semantic_module,
        "run_task_spec_checkpoint_task_trial",
        _fake_task_trial(calls),
    )
    real_run = paired_controls_module.run_task_spec_checkpoint_benchmark

    def tampered_run(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_run(*args, **kwargs)
        trial = report["trials"][0]
        artifact_dir = Path(str(kwargs["artifact_dir"]))
        trace_path = artifact_dir / str(trial["executability_trace_artifact"])
        trace = ExecutabilityTrace(str(trial["rollout_id"])).snapshot()
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        trace_sha256 = file_sha256(trace_path)
        trial["executability_trace_sha256"] = trace_sha256

        artifact_path = artifact_dir / str(trial["artifact"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["executability_trace_sha256"] = trace_sha256
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        trial["artifact_sha256"] = file_sha256(artifact_path)

        summary = report["summary"]
        for field in (
            "executability_policy_step_count",
            "executability_servo_step_count",
            "executability_sent_action_count",
            "executability_safety_accepted_count",
        ):
            summary[field] -= 1
        return report

    monkeypatch.setattr(
        paired_controls_module,
        "run_task_spec_checkpoint_benchmark",
        tampered_run,
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(
        PairedSemanticPromptControlError,
        match="scored executability policy step count mismatch",
    ):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=checkpoint,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
        )

    assert len(calls) == 3
    assert not report_path.exists()


def test_paired_semantic_controls_reject_training_pair_inventory_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_manifest = _pair_manifest(tmp_path)
    semantic_manifest = _semantic_manifest(tmp_path)
    checkpoint, training_report = _candidate(tmp_path, pair_manifest)
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    payload["data"]["validation_pairs"][0]["human_video_sha256"] = "0" * 64
    report_core = {
        key: value
        for key, value in payload.items()
        if key not in {"training_evidence_sha256", "artifacts"}
    }
    training_evidence = canonical_json_sha256(report_core)
    payload["training_evidence_sha256"] = training_evidence
    training_report.write_text(json.dumps(payload), encoding="utf-8")
    bundle = semantic_module.load_compact_wam_bundle(checkpoint)
    metadata = {
        **dict(bundle.metadata),
        "training_evidence_sha256": training_evidence,
    }
    replacement = tmp_path / "replacement.pt"
    save_compact_wam_checkpoint(bundle.model, replacement, metadata=metadata)
    payload["artifacts"]["checkpoint_sha256"] = file_sha256(replacement)
    training_report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedSemanticPromptControlError, match="pair inventory"):
        run_paired_semantic_prompt_controls(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            validation_pair_manifest_path=pair_manifest,
            semantic_manifest_path=semantic_manifest,
            checkpoint_path=replacement,
            training_report_path=training_report,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_task_spec_snapshot_policy_excludes_placeholder_frames(
    tmp_path: Path,
) -> None:
    from so101_wam.mujoco_cli import _TaskSpecSnapshotPolicy

    pair = load_human_robot_pairs(_pair_manifest(tmp_path))[0]
    prompt = human_prompt_from_pair(pair)
    action = ActionChunk(
        target_joint_position=np.zeros((2, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=5.0,
    )
    calls: list[tuple[object, object, float]] = []

    class Base:
        required_history_steps = 4

        def predict_task(
            self,
            task_spec: object,
            live_frames: object,
            *,
            now_s: float,
        ) -> ActionChunk:
            calls.append((task_spec, live_frames, now_s))
            return action

    live_frames = (object(), object())
    placeholder_frames = (object(), object())
    snapshot = SimpleNamespace(
        live_frames=live_frames,
        prompt_frames=placeholder_frames,
    )
    policy = _TaskSpecSnapshotPolicy(Base(), prompt)  # type: ignore[arg-type]

    assert policy.predict(snapshot, now_s=5.0) is action  # type: ignore[arg-type]
    assert policy.required_history_steps == 4
    assert calls == [(prompt, live_frames, 5.0)]


def test_task_spec_session_marks_physical_prompt_as_model_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = load_human_robot_pairs(_pair_manifest(tmp_path))[0]
    prompt = human_prompt_from_pair(pair)
    config = ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    model = CompactWAM(
        latent_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        action_horizon=config.runtime.action_horizon,
    )
    bundle = SimpleNamespace(
        model=model,
        metadata={"checkpoint_id": "task-spec-candidate"},
    )
    axes = np.zeros(ACTION_DIM, dtype=np.float32)
    images = {
        "left_wrist": np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
        "right_wrist": np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
    }
    placeholder = PhysicalPrompt(
        (
            SensorimotorFrame(0.0, images, axes, axes),
            SensorimotorFrame(3.0, images, axes, axes),
        )
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        mujoco_cli_module, "load_compact_wam_bundle", lambda *a, **k: bundle
    )
    monkeypatch.setattr(mujoco_cli_module, "_adapter", lambda *a, **k: object())
    monkeypatch.setattr(
        mujoco_cli_module,
        "_capture_home_prompt",
        lambda adapter: placeholder,
    )

    def fake_session(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        captured.update(kwargs)
        return {"result": "pass"}

    monkeypatch.setattr(
        mujoco_cli_module,
        "_run_checkpoint_policy_session",
        fake_session,
    )

    report = mujoco_cli_module.run_mujoco_task_spec_checkpoint_session(
        config,
        checkpoint_path=checkpoint,
        task_spec=prompt,
        policy_steps=1,
    )

    assert isinstance(captured["policy"], mujoco_cli_module._TaskSpecSnapshotPolicy)
    assert captured["prompt"] is placeholder
    assert report["task_spec_kind"] == "human_video"
    assert report["task_spec_fingerprint"] == prompt.fingerprint
    assert report["runtime_placeholder_prompt_fingerprint"] == placeholder.fingerprint
    assert report["runtime_placeholder_prompt_used_by_model"] is False


def test_task_spec_checkpoint_trial_uses_task_spec_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = load_human_robot_pairs(_pair_manifest(tmp_path))[0]
    prompt = human_prompt_from_pair(pair)
    task = load_semantic_manifest(_semantic_manifest(tmp_path)).heldout_tasks[0]
    seen: list[object] = []

    def fake_session(
        config: ProjectConfig,
        *,
        checkpoint_path: str | Path,
        task_spec: object,
        policy_steps: int,
        device: str,
        semantic_object_body: str,
        initial_object_position: tuple[float, float, float],
        rollout_observer: ExecutabilityTrace,
    ) -> dict[str, object]:
        del config, checkpoint_path, policy_steps, device, initial_object_position
        seen.append(task_spec)
        assert rollout_observer.snapshot()["rollout_id"] == (f"{task.task_id}:seed:7")
        return {
            "runtime_placeholder_prompt_used_by_model": False,
            "task_spec_fingerprint": prompt.fingerprint,
            "rollout": {"final_state": "rollout_ready"},
            "terminal_joint_position": [0.0] * ACTION_DIM,
            "object_body": semantic_object_body,
            "object_physical_profile": OBJECT_PROFILE,
            "object_physical_profile_sha256": OBJECT_PROFILE_SHA256,
            "mujoco_model_identity": MODEL_IDENTITY,
            "terminal_object_position": list(task.target_object_position),
        }

    monkeypatch.setattr(
        semantic_module,
        "run_mujoco_task_spec_checkpoint_session",
        fake_session,
    )

    result = run_task_spec_checkpoint_task_trial(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        task,
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        task_spec=prompt,
    )

    assert seen == [prompt]
    assert result.success is True
    assert result.object_position_error_m == 0.0


def test_paired_semantic_cli_routes_all_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}
    expected = {"result": "complete", "robot_used": False}

    def fake_run(config: ProjectConfig, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        assert config.runtime.backend == "mujoco"
        return expected

    monkeypatch.setattr(
        "so101_wam.mujoco_paired_semantic_prompt_controls."
        "run_paired_semantic_prompt_controls",
        fake_run,
    )
    args = {
        "validation_pair_manifest_path": tmp_path / "pairs.json",
        "semantic_manifest_path": tmp_path / "semantic.json",
        "checkpoint_path": tmp_path / "candidate.pt",
        "training_report_path": tmp_path / "training.json",
        "artifact_dir": tmp_path / "artifacts",
        "report_path": tmp_path / "report.json",
    }

    assert (
        paired_semantic_main(
            [
                "--config",
                str(ROOT / "configs/mujoco_robot_free.toml"),
                "--validation-pair-manifest",
                str(args["validation_pair_manifest_path"]),
                "--semantic-manifest",
                str(args["semantic_manifest_path"]),
                "--checkpoint",
                str(args["checkpoint_path"]),
                "--training-report",
                str(args["training_report_path"]),
                "--artifact-dir",
                str(args["artifact_dir"]),
                "--report",
                str(args["report_path"]),
            ]
        )
        == 0
    )
    assert seen == {**args, "device": "cpu"}
    assert json.loads(capsys.readouterr().out) == expected
