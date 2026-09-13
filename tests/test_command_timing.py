from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.dataset import physical_prompt_from_episode
from so101_wam.training_data import (
    ACTION_TIMING_KEY,
    INITIAL_PREVIOUS_ACTION_KEY,
    OBSERVATION_THEN_COMMAND,
    EpisodeRecord,
    TrainingDataError,
    build_training_windows,
    episode_action_history,
    load_episode_records,
    materialize_training_window,
)


def test_command_history_shifts(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        metadata={
            ACTION_TIMING_KEY: OBSERVATION_THEN_COMMAND,
            INITIAL_PREVIOUS_ACTION_KEY: _axis(-1.0).tolist(),
        },
    )
    window = build_training_windows(
        records,
        policy_hz=10.0,
        servo_hz=10.0,
        action_history_steps=3,
        future_steps=2,
        action_horizon=2,
        ifp_steps=0,
    )[0]

    batch = materialize_training_window(window, image_max_side=8)

    np.testing.assert_allclose(batch.live_actions[0, 0], _axis(-1.0))
    np.testing.assert_allclose(batch.live_actions[0, 1], _axis(0.0))
    np.testing.assert_allclose(batch.live_actions[0, 2], _axis(1.0))
    np.testing.assert_allclose(batch.target_actions[0, 0], _axis(2.0))
    assert not np.array_equal(batch.live_actions[0, -1].numpy(), _axis(2.0))


def test_prompt_actions_stay_raw(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        metadata={
            ACTION_TIMING_KEY: OBSERVATION_THEN_COMMAND,
            INITIAL_PREVIOUS_ACTION_KEY: _axis(-1.0).tolist(),
        },
    )
    window = build_training_windows(
        records,
        policy_hz=10.0,
        servo_hz=10.0,
        action_history_steps=3,
        future_steps=2,
        action_horizon=2,
        ifp_steps=0,
    )[0]

    batch = materialize_training_window(window, image_max_side=8)
    prompt = physical_prompt_from_episode(window.pair.prompt.data, policy_hz=10.0)
    expected = np.stack(
        [frame.executed_action for frame in prompt.frames],
        axis=0,
    ).astype(np.float32)

    np.testing.assert_allclose(batch.prompt_actions[0], expected)


def test_helper_shifts_index_zero(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        metadata={
            ACTION_TIMING_KEY: OBSERVATION_THEN_COMMAND,
            INITIAL_PREVIOUS_ACTION_KEY: _axis(-2.0).tolist(),
        },
    )

    history = episode_action_history(records[0].data, (0, 1))

    np.testing.assert_allclose(history[0], _axis(-2.0))
    np.testing.assert_allclose(history[1], _axis(0.0))


def test_legacy_timing_unchanged(tmp_path: Path) -> None:
    records = _records(tmp_path, metadata={})
    window = build_training_windows(
        records,
        policy_hz=10.0,
        servo_hz=10.0,
        action_history_steps=3,
        future_steps=2,
        action_horizon=2,
        ifp_steps=0,
    )[0]

    batch = materialize_training_window(window, image_max_side=8)

    np.testing.assert_allclose(batch.live_actions[0, 0], _axis(0.0))
    np.testing.assert_allclose(batch.live_actions[0, 1], _axis(1.0))
    np.testing.assert_allclose(batch.live_actions[0, 2], _axis(2.0))
    np.testing.assert_allclose(batch.target_actions[0, 0], _axis(2.0))


def test_load_rejects_bad_timing(tmp_path: Path) -> None:
    _save_episode(
        tmp_path,
        episode_index=0,
        action_offset=0.0,
        metadata={ACTION_TIMING_KEY: "command_then_observation"},
    )

    with pytest.raises(TrainingDataError, match="action_timing"):
        load_episode_records((tmp_path,))


@pytest.mark.parametrize(
    "initial",
    [
        None,
        [0.0],
        [float("nan")] * ACTION_DIM,
        [1e100] * ACTION_DIM,
        ["0.0"] * ACTION_DIM,
    ],
)
def test_load_rejects_bad_initial(tmp_path: Path, initial: object) -> None:
    metadata = {ACTION_TIMING_KEY: OBSERVATION_THEN_COMMAND}
    if initial is not None:
        metadata[INITIAL_PREVIOUS_ACTION_KEY] = initial
    _save_episode(tmp_path, episode_index=0, action_offset=0.0, metadata=metadata)

    with pytest.raises(TrainingDataError, match="initial_previous_action"):
        load_episode_records((tmp_path,))


def test_materialize_rejects_bad_initial(tmp_path: Path) -> None:
    records = _records(
        tmp_path,
        metadata={
            ACTION_TIMING_KEY: OBSERVATION_THEN_COMMAND,
            INITIAL_PREVIOUS_ACTION_KEY: _axis(-1.0).tolist(),
        },
    )
    damaged = EpisodeRecord(
        path=records[0].path,
        data=replace(
            records[0].data,
            metadata={ACTION_TIMING_KEY: OBSERVATION_THEN_COMMAND},
        ),
    )

    with pytest.raises(TrainingDataError, match="initial_previous_action"):
        episode_action_history(damaged.data, (0,))


def _records(
    root: Path,
    *,
    metadata: dict[str, object],
) -> tuple[EpisodeRecord, ...]:
    _save_episode(root, episode_index=0, action_offset=0.0, metadata=metadata)
    _save_episode(root, episode_index=1, action_offset=100.0, metadata=metadata)
    return load_episode_records((root,))


def _save_episode(
    root: Path,
    *,
    episode_index: int,
    action_offset: float,
    metadata: dict[str, object],
) -> None:
    buffer = EpisodeBuffer(
        fps=10.0,
        task="command",
        task_index=1,
        episode_index=episode_index,
        metadata=metadata,
    )
    for index in range(41):
        action = _axis(action_offset + index)
        buffer.append(
            SensorimotorFrame(
                timestamp_s=index / 10.0,
                images={
                    "left_wrist": np.full((8, 8, 3), index, dtype=np.uint8),
                    "right_wrist": np.full((8, 8, 3), index + 1, dtype=np.uint8),
                },
                joint_position=action + 0.5,
                executed_action=action,
            )
        )
    buffer.save(root, stem=f"episode_{episode_index}")


def _axis(base: float) -> np.ndarray:
    return np.array([base + axis for axis in range(ACTION_DIM)], dtype=np.float32)
