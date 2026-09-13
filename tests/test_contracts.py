from __future__ import annotations

import numpy as np
import pytest

from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import (
    ActionChunk,
    ContractError,
    PhysicalPrompt,
    SensorimotorFrame,
)


def make_frame(timestamp_s: float, *, include_action: bool = True) -> SensorimotorFrame:
    images = {
        "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "right_wrist": np.ones((8, 8, 3), dtype=np.uint8),
    }
    action = np.zeros(ACTION_DIM, dtype=np.float32) if include_action else None
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images=images,
        joint_position=np.zeros(ACTION_DIM, dtype=np.float32),
        executed_action=action,
    )


def test_frame_requires_both_wrist_views() -> None:
    with pytest.raises(ContractError, match="right_wrist"):
        SensorimotorFrame(
            timestamp_s=0.0,
            images={"left_wrist": np.zeros((8, 8, 3), dtype=np.uint8)},
            joint_position=np.zeros(ACTION_DIM),
        )


def test_frame_validates_image_timestamps_and_exposes_primary_skew() -> None:
    frame = SensorimotorFrame(
        timestamp_s=1.0,
        images={
            "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_wrist": np.ones((8, 8, 3), dtype=np.uint8),
        },
        joint_position=np.zeros(ACTION_DIM),
        image_timestamps_s={"left_wrist": 10.0, "right_wrist": 10.012},
    )

    assert frame.primary_image_skew_s == pytest.approx(0.012)
    with pytest.raises(ContractError, match="timestamp keys"):
        SensorimotorFrame(
            timestamp_s=1.0,
            images=frame.images,
            joint_position=frame.joint_position,
            image_timestamps_s={"left_wrist": 10.0},
        )
    with pytest.raises(ContractError, match="finite"):
        SensorimotorFrame(
            timestamp_s=1.0,
            images=frame.images,
            joint_position=frame.joint_position,
            image_timestamps_s={
                "left_wrist": 10.0,
                "right_wrist": float("nan"),
            },
        )


def test_prompt_accepts_three_to_twelve_seconds_and_has_stable_fingerprint() -> None:
    prompt = PhysicalPrompt((make_frame(1.0), make_frame(4.0)))
    same_prompt = PhysicalPrompt((make_frame(1.0), make_frame(4.0)))

    assert prompt.duration_s == 3.0
    assert prompt.fingerprint == same_prompt.fingerprint


@pytest.mark.parametrize("duration", [2.999, 12.001])
def test_prompt_rejects_duration_outside_contract(duration: float) -> None:
    with pytest.raises(ContractError, match="duration"):
        PhysicalPrompt((make_frame(0.0), make_frame(duration)))


def test_prompt_requires_executed_actions() -> None:
    with pytest.raises(ContractError, match="executed action"):
        PhysicalPrompt((make_frame(0.0), make_frame(3.0, include_action=False)))


def test_action_chunk_requires_bimanual_width() -> None:
    with pytest.raises(ContractError, match="shape"):
        ActionChunk(
            target_joint_position=np.zeros((10, ACTION_DIM - 1), dtype=np.float32),
            dt_s=0.02,
            created_at_s=1.0,
        )
