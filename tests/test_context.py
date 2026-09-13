from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from so101_wam.constants import ACTION_DIM
from so101_wam.context import Gen15Context, LIVE_SEGMENT, PROMPT_SEGMENT
from so101_wam.contracts import ContractError, PhysicalPrompt, SensorimotorFrame


def make_frame(timestamp_s: float, *, value: int = 0, include_action: bool = True) -> SensorimotorFrame:
    images = {
        "left_wrist": np.full((8, 8, 3), value, dtype=np.uint8),
        "right_wrist": np.full((8, 8, 3), value + 1, dtype=np.uint8),
    }
    action = np.full(ACTION_DIM, value, dtype=np.float32) if include_action else None
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images=images,
        joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
        executed_action=action,
    )


def make_prompt(start_s: float = 0.0, end_s: float = 3.0, *, value: int = 0) -> PhysicalPrompt:
    return PhysicalPrompt((make_frame(start_s, value=value), make_frame(end_s, value=value + 1)))


def test_snapshot_pins_prompt_as_ordered_prefix_with_model_masks() -> None:
    prompt = make_prompt()
    context = Gen15Context(prompt, policy_hz=1.0)
    live_frame = make_frame(10.0, value=10, include_action=False)

    context.append_live(live_frame)
    snapshot = context.snapshot()

    assert snapshot.prompt_fingerprint == prompt.fingerprint
    assert snapshot.prompt_frames == prompt.frames
    assert snapshot.live_frames == (live_frame,)
    assert snapshot.ordered_frames == prompt.frames + (live_frame,)
    assert snapshot.frames == snapshot.ordered_frames
    assert snapshot.segment_labels == (PROMPT_SEGMENT, PROMPT_SEGMENT, LIVE_SEGMENT)
    assert snapshot.attention_mask.tolist() == [True, True, True]
    assert snapshot.prompt_mask.tolist() == [True, True, False]
    assert snapshot.live_mask.tolist() == [False, False, True]
    assert snapshot.causal_mask.tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_snapshot_rejects_fingerprint_that_does_not_match_prompt() -> None:
    snapshot = Gen15Context(make_prompt(), policy_hz=1.0).snapshot()

    with pytest.raises(ContractError, match="prompt_fingerprint"):
        replace(snapshot, prompt_fingerprint="0" * 32)


def test_live_fifo_uses_remaining_total_capacity_after_prompt_frames() -> None:
    context = Gen15Context(make_prompt(), policy_hz=1.0)

    assert context.total_capacity == 30
    assert context.live_capacity == 28

    frames = [make_frame(10.0 + index, value=index, include_action=False) for index in range(30)]
    context.extend_live(frames)

    snapshot = context.snapshot()
    assert snapshot.live_frames == tuple(frames[-28:])
    assert snapshot.ordered_frames == context.prompt.frames + tuple(frames[-28:])
    assert len(snapshot.ordered_frames) == 30


def test_live_timestamps_must_be_strictly_increasing_even_after_fifo_eviction() -> None:
    context = Gen15Context(make_prompt(), policy_hz=1.0)
    context.extend_live([make_frame(10.0 + index, value=index, include_action=False) for index in range(30)])

    with pytest.raises(ContractError, match="strictly increasing"):
        context.append_live(make_frame(39.0, value=99, include_action=False))

    with pytest.raises(ContractError, match="strictly increasing"):
        context.append_live(make_frame(38.5, value=100, include_action=False))

    context.append_live(make_frame(40.0, value=101, include_action=False))
    assert context.snapshot().live_frames[-1].timestamp_s == 40.0


def test_live_frame_requires_wrist_cameras_via_contract() -> None:
    context = Gen15Context(make_prompt(), policy_hz=1.0)

    with pytest.raises(ContractError, match="right_wrist"):
        context.append_live(
            SensorimotorFrame(
                timestamp_s=10.0,
                images={"left_wrist": np.zeros((8, 8, 3), dtype=np.uint8)},
                joint_position=np.zeros(ACTION_DIM, dtype=np.float32),
            )
        )


def test_reset_clears_live_frames_and_allows_new_episode_timestamps() -> None:
    context = Gen15Context(make_prompt(), policy_hz=1.0)
    context.append_live(make_frame(10.0, value=10, include_action=False))

    context.reset_live()
    context.append_live(make_frame(1.0, value=11, include_action=False))

    snapshot = context.snapshot()
    assert snapshot.prompt_frames == context.prompt.frames
    assert [frame.timestamp_s for frame in snapshot.live_frames] == [1.0]


def test_rebind_prompt_replaces_prefix_fingerprint_capacity_and_live_state() -> None:
    first_prompt = make_prompt(0.0, 3.0, value=0)
    second_prompt = PhysicalPrompt(
        (
            make_frame(5.0, value=5),
            make_frame(8.0, value=6),
            make_frame(11.0, value=7),
        )
    )
    context = Gen15Context(first_prompt, policy_hz=1.0)
    context.append_live(make_frame(10.0, value=10, include_action=False))

    context.rebind_prompt(second_prompt)
    snapshot = context.snapshot()

    assert context.prompt == second_prompt
    assert context.live_capacity == 27
    assert snapshot.prompt_fingerprint == second_prompt.fingerprint
    assert snapshot.prompt_frames == second_prompt.frames
    assert snapshot.live_frames == ()
    assert snapshot.segment_labels == (PROMPT_SEGMENT, PROMPT_SEGMENT, PROMPT_SEGMENT)


@pytest.mark.parametrize("policy_hz", [0.0, -1.0, float("inf"), float("nan")])
def test_policy_hz_must_be_finite_and_positive(policy_hz: float) -> None:
    with pytest.raises(ContractError, match="policy_hz"):
        Gen15Context(make_prompt(), policy_hz=policy_hz)


def test_snapshot_arrays_are_readonly() -> None:
    context = Gen15Context(make_prompt(), policy_hz=1.0)
    snapshot = context.snapshot()

    with pytest.raises(ValueError):
        snapshot.attention_mask[0] = False
    with pytest.raises(ValueError):
        snapshot.causal_mask[0, 0] = False
