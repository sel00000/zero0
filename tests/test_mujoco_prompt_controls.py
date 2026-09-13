from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer, load_episode
from so101_wam.deployment import file_sha256, project_config_sha256
from so101_wam.mujoco_benchmark import (
    CHECKPOINT_BENCHMARK_GATE,
    CHECKPOINT_BENCHMARK_POLICY,
    CHECKPOINT_BENCHMARK_SCOPE,
)
from so101_wam.mujoco_prompt_controls import (
    MUJOCO_PROMPT_CONTROL_SCOPE,
    run_mujoco_prompt_control_suite,
)
from so101_wam.prompt_directionality import (
    DIRECTION_CLAIM_SCOPE,
    DIRECTION_PLAN_SCHEMA,
    DirectionPlan,
    load_direction_plan_bytes,
)
from so101_wam.prompt_controls import PromptCondition, PromptControlError


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = tuple(condition.value for condition in PromptCondition)
SUCCESS_COUNTS = {
    PromptCondition.MATCHED.value: 3,
    PromptCondition.SAME_TASK_ALTERNATE.value: 2,
    PromptCondition.WRONG_TASK.value: 0,
    PromptCondition.TEMPORAL_SHUFFLE.value: 1,
    PromptCondition.IMAGE_FRAME_SHUFFLE.value: 1,
    PromptCondition.NULL.value: 0,
    PromptCondition.COUNTERFACTUAL.value: 0,
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
        fps=10.0,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
    )
    for frame_index in range(31):
        value = float(offset + frame_index % 3)
        joints = np.full(ACTION_DIM, value, dtype=np.float32)
        joints[5] = 50.0 + value
        joints[11] = 50.0 - value
        images = {
            PRIMARY_CAMERA_KEYS[0]: np.full(
                (8, 8, 3), offset + frame_index, dtype=np.uint8
            ),
            PRIMARY_CAMERA_KEYS[1]: np.full(
                (8, 8, 3), offset + frame_index + 1, dtype=np.uint8
            ),
        }
        buffer.append(
            SensorimotorFrame(
                timestamp_s=frame_index / 10.0,
                images=images,
                joint_position=joints,
                executed_action=joints,
            )
        )
    return buffer.save(root, stem=stem)[0]


def _prompt_control_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    episodes = tmp_path / "episodes"
    paths = {
        "live": _episode(
            episodes,
            stem="live",
            task="place",
            task_index=2,
            episode_index=1,
            offset=1,
        ),
        "matched": _episode(
            episodes,
            stem="matched",
            task="place",
            task_index=2,
            episode_index=2,
            offset=4,
        ),
        "alternate": _episode(
            episodes,
            stem="alternate",
            task="place",
            task_index=2,
            episode_index=3,
            offset=8,
        ),
        "wrong": _episode(
            episodes,
            stem="wrong",
            task="reach",
            task_index=5,
            episode_index=4,
            offset=12,
        ),
    }
    config = tmp_path / "source.toml"
    config.write_bytes((ROOT / "configs/fake.toml").read_bytes())
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"same-checkpoint-for-all-conditions")
    manifest = tmp_path / "prompt-controls.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "mujoco-prompt-control-test",
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
    return manifest, paths


def _benchmark_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": "prompt-control-joint-proxy",
                "train_task_ids": ["train-home"],
                "heldout_tasks": [
                    {
                        "task_id": "heldout-delta",
                        "label": "held-out joint delta",
                        "policy_steps": 1,
                        "seeds": [7, 13, 29],
                        "target_joint_position": [0.0] * ACTION_DIM,
                        "tolerance": 0.25,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _directionality_manifest(
    tmp_path: Path,
    *,
    prompt_manifest: Path,
    benchmark_manifest: Path,
    checkpoint: Path,
    negative_success_margin: float = 0.34,
    negative_error_margin: float = 0.2,
    negative_conditions: list[str] | None = None,
) -> Path:
    path = tmp_path / "directionality.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": DIRECTION_PLAN_SCHEMA,
                "suite_id": "mujoco-prompt-control-test",
                "hypothesis_id": "expected-matched-drop-v1",
                "claim_scope": DIRECTION_CLAIM_SCOPE,
                "prompt_control_manifest_sha256": file_sha256(prompt_manifest),
                "benchmark_manifest_sha256": file_sha256(benchmark_manifest),
                "checkpoint_sha256": file_sha256(checkpoint),
                "mujoco_config_sha256": project_config_sha256(
                    ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml")
                ),
                "minimum_trials_per_condition": 3,
                "primary_endpoint": "success_rate",
                "secondary_endpoint": "final_error_mean",
                "same_task_noninferiority_margin": 0.34,
                "negative_success_margin": negative_success_margin,
                "negative_error_margin": negative_error_margin,
                "negative_conditions": negative_conditions
                or [
                    PromptCondition.WRONG_TASK.value,
                    PromptCondition.NULL.value,
                    PromptCondition.COUNTERFACTUAL.value,
                ],
                "descriptive_conditions": [
                    PromptCondition.TEMPORAL_SHUFFLE.value,
                    PromptCondition.IMAGE_FRAME_SHUFFLE.value,
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_benchmark(
    calls: list[dict[str, object]],
    *,
    corrupt_prompt_hash: bool = False,
):
    def run(
        config: ProjectConfig,
        *,
        manifest_path: str | Path,
        artifact_dir: str | Path,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None = None,
        device: str = "cpu",
    ) -> dict[str, object]:
        prompt_target = Path(prompt_path)
        prompt_manifest = Path(
            prompt_manifest_path or prompt_target.with_suffix(".json")
        )
        condition = prompt_target.stem
        prompt = load_episode(prompt_target, prompt_manifest)
        success_count = SUCCESS_COUNTS[condition]
        checkpoint_target = Path(checkpoint_path)
        trial_root = Path(artifact_dir)
        trial_root.mkdir(parents=True)
        trials = []
        for trial_index, seed in enumerate((7, 13, 29)):
            success = trial_index < success_count
            final_error = 0.1 if success else 0.5
            artifact = trial_root / f"heldout-delta-seed-{seed}.json"
            artifact.write_text(
                json.dumps({"condition": condition, "seed": seed}),
                encoding="utf-8",
            )
            trials.append(
                {
                    "trial_id": f"heldout-delta:seed:{seed}",
                    "task_id": "heldout-delta",
                    "seed": seed,
                    "success": success,
                    "final_error": final_error,
                    "failure_reason": None if success else "terminal_tolerance",
                    "artifact": artifact.name,
                    "artifact_sha256": file_sha256(artifact),
                }
            )
        calls.append(
            {
                "condition": condition,
                "checkpoint_path": checkpoint_target,
                "manifest_path": Path(manifest_path),
                "prompt_path": prompt_target,
                "device": device,
            }
        )
        prompt_sha256 = file_sha256(prompt_target)
        if corrupt_prompt_hash and condition == PromptCondition.MATCHED.value:
            prompt_sha256 = "0" * 64
        return {
            "schema_version": 1,
            "gate": CHECKPOINT_BENCHMARK_GATE,
            "result": "pass" if success_count == 3 else "fail",
            "mode": "mujoco",
            "evidence_level": "simulation",
            "benchmark_id": "prompt-control-joint-proxy",
            "config_sha256": project_config_sha256(config),
            "manifest_sha256": file_sha256(manifest_path),
            "policy": CHECKPOINT_BENCHMARK_POLICY,
            "benchmark_scope": CHECKPOINT_BENCHMARK_SCOPE,
            "task_disjoint": True,
            "train_task_ids": ["train-home"],
            "heldout_task_ids": ["heldout-delta"],
            "checkpoint_sha256": file_sha256(checkpoint_target),
            "checkpoint_id": "candidate-test",
            "prompt_npz_sha256": prompt_sha256,
            "prompt_manifest_sha256": file_sha256(prompt_manifest),
            "prompt_fingerprint": prompt.fingerprint,
            "task_disjoint_basis": "manifest_ids_only",
            "checkpoint_task_split": "not_verified",
            "prompt_task_match": "not_verified",
            "summary": {
                "task_count": 1,
                "trial_count": 3,
                "success_count": success_count,
                "success_rate": success_count / 3,
                "success_rate_95ci": [0.0, 1.0],
                "all_trials_successful": success_count == 3,
                "failure_counts": (
                    {}
                    if success_count == 3
                    else {"terminal_tolerance": 3 - success_count}
                ),
            },
            "tasks": [],
            "trials": trials,
        }

    return run


def test_mujoco_prompt_controls_run_same_checkpoint_and_seeds_for_all_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest, source_paths = _prompt_control_bundle(tmp_path)
    benchmark_manifest = _benchmark_manifest(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.run_checkpoint_benchmark",
        _fake_benchmark(calls),
    )

    report = run_mujoco_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        benchmark_manifest_path=benchmark_manifest,
        artifact_dir=artifact_dir,
        report_path=report_path,
        device="cpu",
    )

    assert tuple(call["condition"] for call in calls) == CONDITIONS
    assert {call["checkpoint_path"] for call in calls} == {tmp_path / "candidate.pt"}
    assert {call["manifest_path"] for call in calls} == {benchmark_manifest}
    assert report["result"] == "complete"
    assert report["scope"] == MUJOCO_PROMPT_CONTROL_SCOPE
    assert report["robot_used"] is False
    assert report["mujoco_terminal_success_evaluated"] is True
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["scientific_pass_fail_evaluated"] is False
    assert report["directionality_preregistered"] is False
    assert report["checkpoint_task_split"] == "not_verified"
    assert report["prompt_task_match"] == "not_verified"
    assert report["summary"]["condition_count"] == len(CONDITIONS)
    assert report["summary"]["total_trial_count"] == 3 * len(CONDITIONS)
    assert report_path.exists()

    condition_reports = {
        condition["condition"]: condition for condition in report["conditions"]
    }
    assert tuple(condition_reports) == CONDITIONS
    assert condition_reports["matched"]["success_rate_delta_vs_matched"] == 0.0
    assert condition_reports["wrong_task"]["success_rate_delta_vs_matched"] == -1.0
    assert condition_reports["wrong_task"]["final_error_mean_delta_vs_matched"] > 0
    assert all(
        condition["checkpoint_sha256"] == file_sha256(tmp_path / "candidate.pt")
        for condition in report["conditions"]
    )

    generated = {
        condition: load_episode(
            artifact_dir / "prompts" / f"{condition}.npz",
            artifact_dir / "prompts" / f"{condition}.json",
        )
        for condition in CONDITIONS
    }
    matched = load_episode(source_paths["matched"])
    alternate = load_episode(source_paths["alternate"])
    wrong = load_episode(source_paths["wrong"])
    assert generated["matched"].fingerprint == matched.fingerprint
    assert generated["same_task_alternate"].fingerprint == alternate.fingerprint
    assert generated["wrong_task"].fingerprint == wrong.fingerprint
    assert not np.any(generated["null"].wrist_rgb)
    assert not np.any(generated["null"].joint_state)
    np.testing.assert_array_equal(
        generated["image_frame_shuffle"].joint_state,
        matched.joint_state,
    )
    assert not np.array_equal(
        generated["image_frame_shuffle"].wrist_rgb,
        matched.wrist_rgb,
    )
    np.testing.assert_array_equal(
        generated["counterfactual"].wrist_rgb,
        matched.wrist_rgb,
    )
    assert not np.array_equal(
        generated["counterfactual"].joint_state,
        matched.joint_state,
    )

    with pytest.raises(PromptControlError, match="already exists"):
        run_mujoco_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            benchmark_manifest_path=benchmark_manifest,
            artifact_dir=artifact_dir,
            report_path=tmp_path / "second-report.json",
        )
    assert len(calls) == len(CONDITIONS)


def test_mujoco_prompt_controls_reject_benchmark_prompt_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest, _ = _prompt_control_bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.run_checkpoint_benchmark",
        _fake_benchmark(calls, corrupt_prompt_hash=True),
    )

    with pytest.raises(PromptControlError, match="prompt_npz_sha256"):
        run_mujoco_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            benchmark_manifest_path=_benchmark_manifest(tmp_path),
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert len(calls) == 1
    assert not (tmp_path / "report.json").exists()


def test_mujoco_prompt_controls_reject_invalid_directionality_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest, _ = _prompt_control_bundle(tmp_path)
    benchmark_manifest = _benchmark_manifest(tmp_path)
    directionality = _directionality_manifest(
        tmp_path,
        prompt_manifest=prompt_manifest,
        benchmark_manifest=benchmark_manifest,
        checkpoint=tmp_path / "candidate.pt",
        negative_conditions=[
            PromptCondition.MATCHED.value,
            PromptCondition.NULL.value,
            PromptCondition.COUNTERFACTUAL.value,
        ],
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.run_checkpoint_benchmark",
        _fake_benchmark(calls),
    )

    with pytest.raises(PromptControlError, match="condition"):
        run_mujoco_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            benchmark_manifest_path=benchmark_manifest,
            direction_preregistration_path=directionality,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert calls == []
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "report.json").exists()


def test_mujoco_prompt_controls_bind_directionality_sha_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest, _ = _prompt_control_bundle(tmp_path)
    benchmark_manifest = _benchmark_manifest(tmp_path)
    directionality = _directionality_manifest(
        tmp_path,
        prompt_manifest=prompt_manifest,
        benchmark_manifest=benchmark_manifest,
        checkpoint=tmp_path / "candidate.pt",
    )
    payload = json.loads(directionality.read_text(encoding="utf-8"))
    payload["prompt_control_manifest_sha256"] = "0" * 64
    directionality.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.run_checkpoint_benchmark",
        _fake_benchmark(calls),
    )

    with pytest.raises(PromptControlError, match="prompt_control_manifest_sha256"):
        run_mujoco_prompt_control_suite(
            ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
            prompt_control_manifest_path=prompt_manifest,
            benchmark_manifest_path=benchmark_manifest,
            direction_preregistration_path=directionality,
            artifact_dir=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )

    assert calls == []
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "report.json").exists()


def test_mujoco_prompt_controls_pass_preregistered_directionality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest, _ = _prompt_control_bundle(tmp_path)
    benchmark_manifest = _benchmark_manifest(tmp_path)
    directionality = _directionality_manifest(
        tmp_path,
        prompt_manifest=prompt_manifest,
        benchmark_manifest=benchmark_manifest,
        checkpoint=tmp_path / "candidate.pt",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.run_checkpoint_benchmark",
        _fake_benchmark(calls),
    )

    report = run_mujoco_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        benchmark_manifest_path=benchmark_manifest,
        direction_preregistration_path=directionality,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    assert report["result"] == "complete"
    assert report["directionality_preregistered"] is True
    assert report["scientific_pass_fail_evaluated"] is True
    assert report["prompt_causality_claimed"] is False
    assert report["semantic_heldout_success_claimed"] is False
    assert report["preregistration_level"] == "local_input_hash_bound_before_rollout"
    assert report["external_preregistration_timestamp_verified"] is False
    assert report["directionality_evaluation"]["result"] == "pass"
    artifact = (
        tmp_path / "artifacts" / report["directionality_preregistration_artifact"]
    )
    assert artifact.is_file()
    assert report["directionality_preregistration_sha256"] == file_sha256(
        directionality
    )
    assert report["directionality_preregistration_artifact_sha256"] == file_sha256(
        artifact
    )


def test_mujoco_prompt_controls_fail_negative_directionality_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest, _ = _prompt_control_bundle(tmp_path)
    benchmark_manifest = _benchmark_manifest(tmp_path)
    directionality = _directionality_manifest(
        tmp_path,
        prompt_manifest=prompt_manifest,
        benchmark_manifest=benchmark_manifest,
        checkpoint=tmp_path / "candidate.pt",
        negative_error_margin=0.5,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.run_checkpoint_benchmark",
        _fake_benchmark(calls),
    )

    report = run_mujoco_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        benchmark_manifest_path=benchmark_manifest,
        direction_preregistration_path=directionality,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    assert report["result"] == "complete"
    assert report["scientific_pass_fail_evaluated"] is True
    assert report["prompt_causality_claimed"] is False
    assert report["directionality_evaluation"]["result"] == "fail"


def test_mujoco_prompt_controls_freeze_preregistration_before_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_manifest, _ = _prompt_control_bundle(tmp_path)
    benchmark_manifest = _benchmark_manifest(tmp_path)
    directionality = _directionality_manifest(
        tmp_path,
        prompt_manifest=prompt_manifest,
        benchmark_manifest=benchmark_manifest,
        checkpoint=tmp_path / "candidate.pt",
    )
    original = directionality.read_bytes()

    def mutate_and_load(
        source: bytes,
        *,
        expected_suite_id: str,
        expected_prompt_control_manifest_sha256: str,
        expected_benchmark_manifest_sha256: str,
        expected_checkpoint_sha256: str,
        expected_mujoco_config_sha256: str,
    ) -> DirectionPlan:
        payload = json.loads(directionality.read_text(encoding="utf-8"))
        payload["negative_error_margin"] = 0.9
        directionality.write_text(json.dumps(payload), encoding="utf-8")
        return load_direction_plan_bytes(
            source,
            expected_suite_id=expected_suite_id,
            expected_prompt_control_manifest_sha256=(
                expected_prompt_control_manifest_sha256
            ),
            expected_benchmark_manifest_sha256=expected_benchmark_manifest_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_mujoco_config_sha256=expected_mujoco_config_sha256,
        )

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.load_direction_plan_bytes",
        mutate_and_load,
    )
    monkeypatch.setattr(
        "so101_wam.mujoco_prompt_controls.run_checkpoint_benchmark",
        _fake_benchmark(calls),
    )

    report = run_mujoco_prompt_control_suite(
        ProjectConfig.load(ROOT / "configs/mujoco_robot_free.toml"),
        prompt_control_manifest_path=prompt_manifest,
        benchmark_manifest_path=benchmark_manifest,
        direction_preregistration_path=directionality,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "report.json",
    )

    artifact = (
        tmp_path / "artifacts" / report["directionality_preregistration_artifact"]
    )
    assert directionality.read_bytes() != original
    assert artifact.read_bytes() == original
    assert (
        report["directionality_preregistration_sha256"] == sha256(original).hexdigest()
    )
    assert report["directionality_evaluation"]["result"] == "pass"
