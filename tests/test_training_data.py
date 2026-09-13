from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer, save_episode
from so101_wam.training_data import (
    TrainingDataError,
    build_training_windows,
    draw_task_balanced_windows,
    episode_split_digest,
    load_episode_records,
    materialize_training_window,
    task_draw_counts,
    training_axis_statistics,
    validate_task_disjoint_split,
)
from so101_wam.vision import compact_rgb_image


def _save_episode(
    directory: Path,
    *,
    task: str,
    task_index: int,
    episode_index: int,
    offset: float,
    resolution: tuple[int, int] = (8, 8),
    frame_count: int = 91,
) -> Path:
    buffer = EpisodeBuffer(
        fps=30.0,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
        metadata={"action_source": "measured_present_position_no_goal_write"},
    )
    for frame_index in range(frame_count):
        timestamp_s = frame_index / 30.0
        joint = np.array(
            [offset + 10.0 * timestamp_s + axis for axis in range(ACTION_DIM)],
            dtype=np.float32,
        )
        height, width = resolution
        buffer.append(
            SensorimotorFrame(
                timestamp_s=timestamp_s,
                images={
                    "left_wrist": np.full(
                        (height, width, 3), frame_index % 255, dtype=np.uint8
                    ),
                    "right_wrist": np.full(
                        (height, width, 3), (frame_index + 1) % 255, dtype=np.uint8
                    ),
                },
                joint_position=joint,
                executed_action=joint,
            )
        )
    path, _ = buffer.save(directory, stem=f"episode_{episode_index:06d}")
    return path


def test_task_disjoint_split_rejects_episode_and_task_leakage(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train = _save_episode(
        train_dir, task="pick", task_index=1, episode_index=1, offset=0.0
    )
    _save_episode(train_dir, task="pick", task_index=1, episode_index=2, offset=1.0)
    _save_episode(val_dir, task="pick", task_index=2, episode_index=3, offset=2.0)
    _save_episode(val_dir, task="pick", task_index=2, episode_index=4, offset=3.0)

    train_records = load_episode_records([train_dir])
    validation_records = load_episode_records([val_dir])
    with pytest.raises(TrainingDataError, match="task labels must be disjoint"):
        validate_task_disjoint_split(train_records, validation_records)

    with pytest.raises(TrainingDataError, match="fingerprint leakage"):
        validate_task_disjoint_split(train_records, (train_records[0], train_records[1]))
    assert train.exists()


def test_split_content_leak(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _save_episode(train_dir, task="pick", task_index=1, episode_index=1, offset=0.0)
    _save_episode(train_dir, task="pick", task_index=1, episode_index=2, offset=1.0)
    _save_episode(val_dir, task="stack", task_index=2, episode_index=3, offset=0.0)
    _save_episode(val_dir, task="stack", task_index=2, episode_index=4, offset=2.0)

    train_records = load_episode_records([train_dir])
    validation_records = load_episode_records([val_dir])

    assert train_records[0].fingerprint != validation_records[0].fingerprint
    assert train_records[0].data.content_fingerprint == validation_records[0].data.content_fingerprint
    with pytest.raises(TrainingDataError, match="content leakage"):
        validate_task_disjoint_split(train_records, validation_records)


def test_group_content_clone(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _save_episode(train_dir, task="pick", task_index=1, episode_index=1, offset=0.0)
    original = load_episode_records([train_dir])[0]
    save_episode(
        replace(original.data, episode_index=2, metadata={"operator": "copy"}),
        train_dir,
    )
    _save_episode(val_dir, task="stack", task_index=2, episode_index=3, offset=2.0)
    _save_episode(val_dir, task="stack", task_index=2, episode_index=4, offset=3.0)
    records = load_episode_records([train_dir])

    with pytest.raises(TrainingDataError, match="duplicate episode content"):
        validate_task_disjoint_split(records, load_episode_records([val_dir]))

    with pytest.raises(TrainingDataError, match="duplicate episode content"):
        build_training_windows(
            records,
            policy_hz=10.0,
            servo_hz=50.0,
            action_history_steps=4,
            future_steps=3,
            action_horizon=10,
            ifp_steps=0,
        )


def test_content_hash_digest(tmp_path: Path) -> None:
    data_dir = tmp_path / "episodes"
    _save_episode(data_dir, task="pick", task_index=1, episode_index=1, offset=0.0)
    _save_episode(data_dir, task="pick", task_index=1, episode_index=2, offset=1.0)
    records = load_episode_records([data_dir])

    assert episode_split_digest(records) == episode_split_digest(tuple(reversed(records)))
    assert records[0].data.content_fingerprint != records[1].data.content_fingerprint


def test_windows_use_different_prompt_episode_and_align_video_action_targets(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "episodes"
    _save_episode(data_dir, task="pick", task_index=1, episode_index=1, offset=0.0)
    _save_episode(data_dir, task="pick", task_index=1, episode_index=2, offset=100.0)
    records = load_episode_records([data_dir])

    windows = build_training_windows(
        records,
        policy_hz=10.0,
        servo_hz=50.0,
        action_history_steps=4,
        future_steps=3,
        action_horizon=10,
        ifp_steps=2,
        ifp_stride=2,
    )
    first = windows[0]
    assert first.pair.prompt.fingerprint != first.pair.target.fingerprint
    assert first.pair.prompt.data.task_index == first.pair.target.data.task_index

    batch = materialize_training_window(first, image_max_side=8)
    assert batch.prompt_images.shape == (1, 31, 2, 3, 8, 8)
    assert batch.live_images.shape == (1, 4, 2, 3, 8, 8)
    assert batch.target_future_images.shape == (1, 3, 2, 3, 8, 8)
    assert batch.target_ifp_images is not None
    assert batch.target_ifp_images.shape == (1, 2, 2, 3, 8, 8)
    assert batch.target_actions.shape == (1, 10, ACTION_DIM)

    anchor_raw_index = first.pair.target_policy_indices[first.anchor_policy_position]
    anchor_time = float(first.pair.target.data.timestamps_s[anchor_raw_index])
    expected_first_action = first.pair.target.data.action[0, 0] + 10.0 * anchor_time
    assert float(batch.target_actions[0, 0, 0]) == pytest.approx(
        expected_first_action, abs=1e-5
    )

    next_policy_raw = first.pair.target_policy_indices[first.anchor_policy_position + 1]
    assert int(batch.target_future_images[0, 0, 0, 0, 0, 0]) == next_policy_raw
    first_ifp_raw = first.pair.target_policy_indices[first.anchor_policy_position + 2]
    second_ifp_raw = first.pair.target_policy_indices[first.anchor_policy_position + 4]
    assert int(batch.target_ifp_images[0, 0, 0, 0, 0, 0]) == first_ifp_raw
    assert int(batch.target_ifp_images[0, 1, 0, 0, 0, 0]) == second_ifp_raw


def test_task_balanced_sampler_equalizes_task_draws(tmp_path: Path) -> None:
    data_dir = tmp_path / "episodes"
    _save_episode(
        data_dir,
        task="short",
        task_index=1,
        episode_index=1,
        offset=0.0,
        frame_count=91,
    )
    _save_episode(
        data_dir,
        task="short",
        task_index=1,
        episode_index=2,
        offset=10.0,
        frame_count=91,
    )
    _save_episode(
        data_dir,
        task="long",
        task_index=2,
        episode_index=3,
        offset=100.0,
        frame_count=121,
    )
    _save_episode(
        data_dir,
        task="long",
        task_index=2,
        episode_index=4,
        offset=200.0,
        frame_count=121,
    )
    windows = build_training_windows(
        load_episode_records([data_dir]),
        policy_hz=10.0,
        servo_hz=50.0,
        action_history_steps=4,
        future_steps=3,
        action_horizon=10,
        ifp_steps=0,
    )

    assert task_draw_counts(windows) == {"1:short": 50, "2:long": 70}

    schedule = draw_task_balanced_windows(
        windows,
        steps=17,
        rng=np.random.default_rng(7),
    )
    repeated = draw_task_balanced_windows(
        windows,
        steps=17,
        rng=np.random.default_rng(7),
    )
    resumed = draw_task_balanced_windows(
        windows,
        steps=11,
        start_draw=6,
        rng=np.random.default_rng(7),
    )

    assert task_draw_counts(schedule) == {"1:short": 9, "2:long": 8}
    assert repeated == schedule
    assert resumed == schedule[6:]


def test_split_accepts_task_holdout_and_statistics_use_train_only(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _save_episode(train_dir, task="pick", task_index=1, episode_index=1, offset=0.0)
    _save_episode(train_dir, task="pick", task_index=1, episode_index=2, offset=2.0)
    _save_episode(val_dir, task="stack", task_index=2, episode_index=3, offset=500.0)
    _save_episode(val_dir, task="stack", task_index=2, episode_index=4, offset=600.0)

    split = validate_task_disjoint_split(
        load_episode_records([train_dir]), load_episode_records([val_dir])
    )
    mean, scale = training_axis_statistics(split.train)

    assert split.train_digest != split.validation_digest
    assert mean.shape == scale.shape == (ACTION_DIM,)
    assert float(mean.max()) < 100.0
    assert np.all(scale >= 1.0)


def test_compact_rgb_preprocessing_preserves_aspect_and_small_inputs() -> None:
    large = np.zeros((96, 128, 3), dtype=np.uint8)
    small = np.zeros((8, 10, 3), dtype=np.uint8)

    assert compact_rgb_image(large).shape == (48, 64, 3)
    compact_small = compact_rgb_image(small)
    assert compact_small.shape == small.shape
    assert not np.shares_memory(compact_small, small)
