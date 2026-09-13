from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so101_wam.checkpoint import load_compact_wam_bundle
from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.deployment import canonical_json_sha256
from so101_wam.hardware_cli import HardwareCLIError, _require_deployment_checkpoint
from so101_wam.model import ActionDecoder
from so101_wam.training import (
    CompactWAMTrainingConfig,
    SamplingStrategy,
    TrainingArtifactExistsError,
    _draw_stage_schedules,
    _draw_window_schedule,
    _task_inventory,
    train_offline_candidate,
)
from so101_wam.training_data import build_training_windows, load_episode_records
from so101_wam.train_cli import main as train_main


def _save_pair(
    directory: Path,
    *,
    task: str,
    task_index: int,
    first_episode_index: int,
    base: float,
) -> None:
    for pair_index in range(2):
        episode_index = first_episode_index + pair_index
        buffer = EpisodeBuffer(
            fps=30.0,
            task=task,
            task_index=task_index,
            episode_index=episode_index,
            metadata={"action_source": "measured_present_position_no_goal_write"},
        )
        for frame_index in range(91):
            timestamp_s = frame_index / 30.0
            value = base + pair_index + timestamp_s
            axes = np.full(ACTION_DIM, value, dtype=np.float32)
            buffer.append(
                SensorimotorFrame(
                    timestamp_s=timestamp_s,
                    images={
                        "left_wrist": np.full(
                            (8, 8, 3), frame_index % 255, dtype=np.uint8
                        ),
                        "right_wrist": np.full(
                            (8, 8, 3), (frame_index + 3) % 255, dtype=np.uint8
                        ),
                    },
                    joint_position=axes,
                    executed_action=axes,
                )
            )
        buffer.save(directory, stem=f"episode_{episode_index:06d}")


def _tiny_config() -> CompactWAMTrainingConfig:
    return CompactWAMTrainingConfig(
        latent_dim=8,
        transformer_heads=2,
        future_steps=1,
        action_horizon=2,
        action_history_steps=1,
        ifp_steps=1,
        stage1_steps=1,
        stage2_steps=1,
        seed=11,
    )


def test_task_inventory_is_sorted_and_deduplicated(tmp_path: Path) -> None:
    _save_pair(
        tmp_path,
        task="stack",
        task_index=2,
        first_episode_index=3,
        base=10.0,
    )
    _save_pair(
        tmp_path,
        task="pick",
        task_index=1,
        first_episode_index=1,
        base=0.0,
    )

    inventory = _task_inventory(load_episode_records([tmp_path]))

    assert inventory == [
        {"task_index": 1, "task": "pick"},
        {"task_index": 2, "task": "stack"},
    ]


def test_offline_training_publishes_candidate_with_auditable_false_real_gate(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    _save_pair(
        train_dir,
        task="pick",
        task_index=1,
        first_episode_index=1,
        base=0.0,
    )
    _save_pair(
        validation_dir,
        task="stack",
        task_index=2,
        first_episode_index=3,
        base=10.0,
    )
    checkpoint_path = tmp_path / "candidate.pt"
    report_path = tmp_path / "candidate.training.json"

    artifacts = train_offline_candidate(
        load_episode_records([train_dir]),
        load_episode_records([validation_dir]),
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        checkpoint_id="candidate-test-1",
        config=_tiny_config(),
    )

    assert artifacts.trained is False
    assert artifacts.deployment_ready is False
    assert artifacts.train_windows > 0 and artifacts.validation_windows > 0
    assert np.isfinite(artifacts.validation_action_mse_normalized)
    bundle = load_compact_wam_bundle(checkpoint_path)
    assert bundle.metadata["offline_trained"] is True
    assert bundle.metadata["artifact_kind"] == "compact_wam_candidate"
    assert bundle.metadata["evidence_level"] == "offline"
    assert bundle.metadata["trained"] is False
    assert bundle.metadata["deployment_ready"] is False
    assert bundle.metadata["training_evidence_sha256"] == artifacts.training_evidence_sha256
    assert not np.allclose(bundle.model.axis_mean.numpy(), 0.0)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["protocol"]["split"] == "task_disjoint"
    assert report["protocol"]["sampling_strategy"] == "task_balanced"
    assert report["protocol"]["real_output_authorized"] is False
    assert report["data"]["train_tasks"] == [{"task_index": 1, "task": "pick"}]
    assert report["data"]["validation_tasks"] == [{"task_index": 2, "task": "stack"}]
    report_core = {
        key: value
        for key, value in report.items()
        if key not in {"training_evidence_sha256", "artifacts"}
    }
    assert canonical_json_sha256(report_core) == artifacts.training_evidence_sha256
    sampling = report["data"]["sampling_audit"]
    assert sampling["strategy"] == "task_balanced"
    assert sampling["start_draw"] == 0
    assert sampling["next_draw"] == 2
    assert sampling["task_draw_counts"] == {"1:pick": 2}
    assert len(sampling["schedule_sha256"]) == 64
    assert report["training_evidence_sha256"] == artifacts.training_evidence_sha256
    assert report["artifacts"]["checkpoint_filename"] == checkpoint_path.name
    with pytest.raises(HardwareCLIError, match="rejects offline"):
        _require_deployment_checkpoint(bundle.metadata)

    with pytest.raises(TrainingArtifactExistsError, match="immutable"):
        train_offline_candidate(
            load_episode_records([train_dir]),
            load_episode_records([validation_dir]),
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            checkpoint_id="candidate-test-2",
            config=_tiny_config(),
        )


@pytest.mark.parametrize(
    "mode",
    [
        ActionDecoder.LEGACY_MEAN,
        ActionDecoder.ORDERED_CONCAT,
    ],
)
def test_training_cli_runs_one_step_and_prints_candidate_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: ActionDecoder,
) -> None:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    _save_pair(
        train_dir,
        task="pick",
        task_index=1,
        first_episode_index=1,
        base=0.0,
    )
    _save_pair(
        validation_dir,
        task="stack",
        task_index=2,
        first_episode_index=3,
        base=10.0,
    )
    checkpoint_path = tmp_path / "cli_candidate.pt"
    report_path = tmp_path / "cli_candidate.training.json"

    args = [
        "--train-episodes",
        str(train_dir),
        "--validation-episodes",
        str(validation_dir),
        "--output",
        str(checkpoint_path),
        "--report",
        str(report_path),
        "--checkpoint-id",
        "cli-candidate-test",
        "--stage1-steps",
        "0",
        "--stage2-steps",
        "1",
        "--latent-dim",
        "8",
        "--transformer-heads",
        "2",
        "--future-steps",
        "1",
        "--action-horizon",
        "2",
        "--action-history-steps",
        "1",
        "--ifp-steps",
        "1",
        "--sampling-strategy",
        SamplingStrategy.WINDOW_SHUFFLE.value,
    ]
    if mode is not ActionDecoder.LEGACY_MEAN:
        args.extend(["--action-decoder", mode.value])

    result = train_main(args)

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["trained"] is False
    assert payload["deployment_ready"] is False
    assert checkpoint_path.exists() and report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["protocol"]["sampling_strategy"] == "window_shuffle"
    assert report["model"]["action_decoder"] == mode.value
    bundle = load_compact_wam_bundle(checkpoint_path)
    assert bundle.metadata["action_decoder"] == mode.value


def test_window_shuffle_preserves_legacy_stage_boundaries(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    _save_pair(
        train_dir,
        task="pick",
        task_index=1,
        first_episode_index=1,
        base=0.0,
    )
    windows = build_training_windows(
        load_episode_records([train_dir]),
        policy_hz=10.0,
        servo_hz=50.0,
        action_history_steps=1,
        future_steps=1,
        action_horizon=2,
        ifp_steps=0,
    )
    expected_rng = np.random.default_rng(19)
    expected = (
        _draw_window_schedule(windows, steps=2, rng=expected_rng),
        _draw_window_schedule(windows, steps=3, rng=expected_rng),
    )

    actual = _draw_stage_schedules(
        windows,
        stage1_steps=2,
        stage2_steps=3,
        strategy=SamplingStrategy.WINDOW_SHUFFLE,
        rng=np.random.default_rng(19),
    )

    assert actual == expected
