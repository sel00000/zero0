from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.training import CompactWAMTrainingConfig
from so101_wam.training_data import EpisodeRecord, load_episode_records


@pytest.fixture(scope="module")
def decoder_records(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]]:
    root = tmp_path_factory.mktemp("decoder-data")
    episodes_per_task = 2
    for split, task_index in (("train", 1), ("validation", 2)):
        split_path = root / split
        for repeat in range(episodes_per_task):
            buffer = EpisodeBuffer(
                fps=30.0,
                task=split,
                task_index=task_index,
                episode_index=task_index * 10 + repeat,
                metadata={
                    "action_source": "measured_present_position_no_goal_write",
                },
            )
            for index in range(91):
                axes = np.full(
                    12,
                    task_index * episodes_per_task + repeat + index / 30,
                    dtype=np.float32,
                )
                buffer.append(
                    SensorimotorFrame(
                        timestamp_s=index / 30,
                        images={
                            "left_wrist": np.full((8, 8, 3), index, dtype=np.uint8),
                            "right_wrist": np.full((8, 8, 3), index + 3, dtype=np.uint8),
                        },
                        joint_position=axes,
                        executed_action=axes,
                    )
                )
            buffer.save(split_path, stem=f"episode_{repeat}")

    return (
        load_episode_records((Path(root) / "train",)),
        load_episode_records((Path(root) / "validation",)),
    )


@pytest.fixture(scope="module")
def decoder_config() -> CompactWAMTrainingConfig:
    return CompactWAMTrainingConfig(
        latent_dim=8,
        transformer_heads=2,
        future_steps=3,
        action_horizon=2,
        action_history_steps=1,
        ifp_steps=2,
        stage1_steps=1,
        stage2_steps=1,
        seed=7,
    )
