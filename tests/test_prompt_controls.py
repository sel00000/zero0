from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_wam.checkpoint import save_compact_wam_checkpoint
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import PhysicalPrompt, SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer, EpisodeData
from so101_wam.deployment import file_sha256
from so101_wam.model import CompactWAM
from so101_wam.prompt_controls import (
    PromptCondition,
    PromptControlError,
    build_prompt_conditions,
    load_prompt_control_manifest,
    run_prompt_control_suite,
    summarize_action_divergence,
    validate_prompt_control_sources,
)


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "matched",
    "same_task_alternate",
    "wrong_task",
    "temporal_shuffle",
    "image_frame_shuffle",
    "null",
    "counterfactual",
}


def _frame(timestamp_s: float, value: int) -> SensorimotorFrame:
    joints = np.full(ACTION_DIM, float(value), dtype=np.float32)
    joints[5] = 50.0 + value
    joints[11] = 50.0 - value
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={
            PRIMARY_CAMERA_KEYS[0]: np.full((8, 8, 3), value, dtype=np.uint8),
            PRIMARY_CAMERA_KEYS[1]: np.full(
                (8, 8, 3), value + 1, dtype=np.uint8
            ),
        },
        joint_position=joints,
        executed_action=joints,
    )


def _prompt(offset: int) -> PhysicalPrompt:
    return PhysicalPrompt(
        tuple(_frame(float(index), offset + index) for index in range(4))
    )


def _episode(
    *,
    task: str,
    task_index: int,
    episode_index: int,
    offset: int,
) -> EpisodeData:
    frames = tuple(_frame(index / 10.0, offset + index % 5) for index in range(31))
    buffer = EpisodeBuffer(
        fps=10.0,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
    )
    buffer.extend(iter(frames))
    return buffer.to_episode_data()


def _save_episode(root: Path, data: EpisodeData, stem: str) -> Path:
    buffer = EpisodeBuffer(
        fps=data.fps,
        task=data.task,
        task_index=data.task_index,
        episode_index=data.episode_index,
    )
    buffer.extend(data.frames())
    return buffer.save(root, stem=stem)[0]


def test_prompt_conditions_are_deterministic_and_isolate_modalities() -> None:
    config = ProjectConfig.load(ROOT / "configs/fake.toml")
    matched = _prompt(1)
    alternate = _prompt(11)
    wrong_task = _prompt(21)

    first = build_prompt_conditions(
        matched,
        alternate,
        wrong_task,
        safety=config.safety,
        seed=7,
    )
    repeated = build_prompt_conditions(
        matched,
        alternate,
        wrong_task,
        safety=config.safety,
        seed=7,
    )

    assert {condition.value for condition in first} == CONDITIONS
    assert {
        condition: prompt.fingerprint for condition, prompt in first.items()
    } == {
        condition: prompt.fingerprint for condition, prompt in repeated.items()
    }
    assert first[PromptCondition.MATCHED].fingerprint == matched.fingerprint
    assert (
        first[PromptCondition.SAME_TASK_ALTERNATE].fingerprint
        == alternate.fingerprint
    )
    assert first[PromptCondition.WRONG_TASK].fingerprint == wrong_task.fingerprint

    timestamps = tuple(frame.timestamp_s for frame in matched.frames)
    for condition in (
        PromptCondition.TEMPORAL_SHUFFLE,
        PromptCondition.IMAGE_FRAME_SHUFFLE,
        PromptCondition.NULL,
        PromptCondition.COUNTERFACTUAL,
    ):
        assert tuple(
            frame.timestamp_s for frame in first[condition].frames
        ) == timestamps

    image_shuffle = first[PromptCondition.IMAGE_FRAME_SHUFFLE]
    for original, changed in zip(matched.frames, image_shuffle.frames, strict=True):
        np.testing.assert_array_equal(changed.joint_position, original.joint_position)
        np.testing.assert_array_equal(changed.executed_action, original.executed_action)
    assert image_shuffle.fingerprint != matched.fingerprint

    null = first[PromptCondition.NULL]
    assert all(not np.any(frame.joint_position) for frame in null.frames)
    assert all(not np.any(frame.executed_action) for frame in null.frames)
    assert all(
        not np.any(image)
        for frame in null.frames
        for image in frame.primary_images
    )

    counterfactual = first[PromptCondition.COUNTERFACTUAL]
    assert all(
        np.array_equal(changed.primary_images[0], original.primary_images[0])
        for original, changed in zip(
            matched.frames, counterfactual.frames, strict=True
        )
    )
    assert counterfactual.fingerprint != matched.fingerprint


def test_prompt_control_sources_require_three_distinct_same_task_episodes() -> None:
    target = _episode(task="place", task_index=2, episode_index=1, offset=1)
    matched = _episode(task="place", task_index=2, episode_index=2, offset=2)
    alternate = _episode(task="place", task_index=2, episode_index=3, offset=3)
    wrong = _episode(task="reach", task_index=1, episode_index=4, offset=4)

    validate_prompt_control_sources(target, matched, alternate, wrong)

    with pytest.raises(PromptControlError, match="distinct"):
        validate_prompt_control_sources(target, target, alternate, wrong)
    with pytest.raises(PromptControlError, match="wrong-task"):
        validate_prompt_control_sources(target, matched, alternate, matched)

    relabeled_duplicate = EpisodeData(
        timestamps_s=target.timestamps_s,
        wrist_rgb=target.wrist_rgb,
        joint_state=target.joint_state,
        action=target.action,
        fps=target.fps,
        task=target.task,
        task_index=target.task_index,
        episode_index=99,
    )
    with pytest.raises(PromptControlError, match="content-distinct"):
        validate_prompt_control_sources(
            target,
            relabeled_duplicate,
            alternate,
            wrong,
        )


@pytest.mark.parametrize("unsafe_path", ["/tmp/outside.pt", "../outside.pt"])
def test_prompt_control_manifest_rejects_paths_outside_bundle(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "prompt-control-local-v1",
                "seed": 7,
                "config": "config.toml",
                "checkpoint": unsafe_path,
                "live_episode": "live.npz",
                "matched_prompt": "matched.npz",
                "same_task_alternate_prompt": "alternate.npz",
                "wrong_task_prompt": "wrong.npz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PromptControlError, match="bundle-relative"):
        load_prompt_control_manifest(manifest)


def test_prompt_control_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside"
    bundle.mkdir()
    outside.mkdir()
    (bundle / "escape").symlink_to(outside, target_is_directory=True)
    manifest = bundle / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "prompt-control-local-v1",
                "seed": 7,
                "config": "config.toml",
                "checkpoint": "escape/outside.pt",
                "live_episode": "live.npz",
                "matched_prompt": "matched.npz",
                "same_task_alternate_prompt": "alternate.npz",
                "wrong_task_prompt": "wrong.npz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PromptControlError, match="bundle-relative"):
        load_prompt_control_manifest(manifest)


def test_action_divergence_uses_matched_as_the_only_baseline() -> None:
    matched = np.zeros((2, ACTION_DIM), dtype=np.float32)
    alternate = np.ones((2, ACTION_DIM), dtype=np.float32)
    metrics = summarize_action_divergence(
        {
            PromptCondition.MATCHED: matched,
            PromptCondition.SAME_TASK_ALTERNATE: alternate,
        }
    )

    assert metrics["matched"]["l2_vs_matched"] == 0.0
    assert metrics["same_task_alternate"]["mean_abs_vs_matched"] == 1.0
    assert metrics["same_task_alternate"]["max_abs_vs_matched"] == 1.0
    assert metrics["same_task_alternate"]["per_axis_max_abs_vs_matched"] == [
        1.0
    ] * ACTION_DIM
    assert metrics["same_task_alternate"]["action_shape"] == [2, ACTION_DIM]
    assert metrics["same_task_alternate"]["all_actions_finite"] is True


def test_prompt_control_suite_writes_immutable_negative_safe_evidence(
    tmp_path: Path,
) -> None:
    torch.manual_seed(5)
    checkpoint = tmp_path / "candidate.pt"
    save_compact_wam_checkpoint(
        CompactWAM(
            latent_dim=8,
            transformer_heads=2,
            future_steps=1,
            action_horizon=10,
            action_history_steps=1,
            ifp_steps=0,
        ),
        checkpoint,
        metadata={"checkpoint_id": "prompt-control-test"},
    )
    episode_dir = tmp_path / "episodes"
    target = _save_episode(
        episode_dir,
        _episode(task="place", task_index=2, episode_index=1, offset=1),
        "target",
    )
    matched = _save_episode(
        episode_dir,
        _episode(task="place", task_index=2, episode_index=2, offset=2),
        "matched",
    )
    alternate = _save_episode(
        episode_dir,
        _episode(task="place", task_index=2, episode_index=3, offset=3),
        "alternate",
    )
    wrong = _save_episode(
        episode_dir,
        _episode(task="reach", task_index=1, episode_index=4, offset=4),
        "wrong",
    )
    manifest = tmp_path / "prompt_controls.json"
    config = tmp_path / "fake.toml"
    config.write_bytes((ROOT / "configs/fake.toml").read_bytes())
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "prompt-control-local-v1",
                "seed": 7,
                "config": config.name,
                "checkpoint": checkpoint.name,
                "live_episode": target.relative_to(tmp_path).as_posix(),
                "matched_prompt": matched.relative_to(tmp_path).as_posix(),
                "same_task_alternate_prompt": alternate.relative_to(
                    tmp_path
                ).as_posix(),
                "wrong_task_prompt": wrong.relative_to(tmp_path).as_posix(),
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    artifact_dir = tmp_path / "artifacts"

    collision_dir = tmp_path / "collision"
    with pytest.raises(PromptControlError, match="distinct"):
        run_prompt_control_suite(
            manifest,
            artifact_dir=collision_dir,
            report_path=collision_dir / "matched.json",
            device="cpu",
        )
    assert not collision_dir.exists()

    report = run_prompt_control_suite(
        manifest,
        artifact_dir=artifact_dir,
        report_path=report_path,
        device="cpu",
    )

    assert report["result"] == "complete"
    assert report["evidence_level"] == "offline"
    assert report["scope"] == "diagnostic_prompt_control_action_divergence"
    assert report["robot_used"] is False
    assert report["mujoco_terminal_success_evaluated"] is False
    assert report["real_world_success_claimed"] is False
    assert report["causality_claimed"] is False
    assert report["summary"]["condition_count"] == len(CONDITIONS)
    assert report["summary"]["matched_repeat_max_abs"] == 0.0
    assert {condition["condition"] for condition in report["conditions"]} == CONDITIONS
    assert report["source_path_policy"] == "manifest_bundle_relative"
    assert set(report["source_episodes"]) == {
        "live",
        "matched",
        "same_task_alternate",
        "wrong_task",
    }
    assert report["source_episodes"]["matched"]["file_sha256"] == file_sha256(
        matched
    )
    assert report["source_episodes"]["wrong_task"]["manifest_path"] == (
        wrong.relative_to(tmp_path).as_posix()
    )
    assert len(tuple(artifact_dir.glob("*.json"))) == len(CONDITIONS)
    assert report_path.exists()

    with pytest.raises(PromptControlError, match="already exists"):
        run_prompt_control_suite(
            manifest,
            artifact_dir=artifact_dir,
            report_path=report_path,
            device="cpu",
        )
