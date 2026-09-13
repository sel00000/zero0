from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from so101_wam.constants import ACTION_DIM
from so101_wam.context import Gen15Context
from so101_wam.contracts import PhysicalPrompt, SensorimotorFrame
from so101_wam.tensorizer import TensorizerError, tensorize_context


def make_frame(
    timestamp_s: float,
    *,
    value: int,
    resolution: tuple[int, int] = (4, 5),
    include_action: bool = True,
    extra_camera: bool = False,
) -> SensorimotorFrame:
    height, width = resolution
    images = {
        "left_wrist": np.full((height, width, 3), value, dtype=np.uint8),
        "right_wrist": np.full((height, width, 3), value + 10, dtype=np.uint8),
    }
    if extra_camera:
        images["head_optional"] = np.full((height, width, 3), value + 20, dtype=np.uint8)
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images=images,
        joint_position=np.full(ACTION_DIM, value + 0.5, dtype=np.float32),
        executed_action=np.full(ACTION_DIM, value + 1.5, dtype=np.float32) if include_action else None,
    )


def make_snapshot(*, live_count: int = 2, live_action: bool = True, resolution: tuple[int, int] = (4, 5)):
    prompt = PhysicalPrompt((make_frame(0.0, value=1), make_frame(3.0, value=2)))
    context = Gen15Context(prompt, policy_hz=1.0)
    for index in range(live_count):
        context.append_live(
            make_frame(
                4.0 + index,
                value=10 + index,
                resolution=resolution,
                include_action=live_action,
            )
        )
    return context.snapshot()


def test_tensorize_context_shapes_order_and_content() -> None:
    batch = tensorize_context(make_snapshot())

    assert batch.prompt_images.shape == (1, 2, 2, 3, 4, 5)
    assert batch.live_images.shape == (1, 2, 2, 3, 4, 5)
    assert batch.prompt_images.dtype == torch.uint8
    assert batch.prompt_proprio.shape == (1, 2, ACTION_DIM)
    assert batch.prompt_actions.shape == (1, 2, ACTION_DIM)
    assert batch.live_proprio.shape == (1, 2, ACTION_DIM)
    assert batch.live_actions.shape == (1, 2, ACTION_DIM)
    assert batch.prompt_mask.shape == (1, 2)
    assert batch.prompt_mask.tolist() == [[True, True]]

    assert tuple(batch.as_kwargs()) == (
        "prompt_images",
        "prompt_proprio",
        "prompt_actions",
        "live_images",
        "live_proprio",
        "live_actions",
        "prompt_mask",
    )
    assert int(batch.prompt_images[0, 0, 0, 0, 0, 0]) == 1
    assert int(batch.prompt_images[0, 0, 1, 0, 0, 0]) == 11
    assert int(batch.live_images[0, 1, 0, 0, 0, 0]) == 11
    assert int(batch.live_images[0, 1, 1, 0, 0, 0]) == 21
    assert float(batch.live_actions[0, 1, 0]) == pytest.approx(12.5)


def test_tensorize_context_rejects_missing_executed_action() -> None:
    with pytest.raises(TensorizerError, match="missing executed_action"):
        tensorize_context(make_snapshot(live_action=False))


def test_tensorize_context_rejects_missing_live_frames() -> None:
    with pytest.raises(TensorizerError, match="at least one live frame"):
        tensorize_context(make_snapshot(live_count=0))


def test_tensorize_context_rejects_resolution_mismatch_across_segments() -> None:
    with pytest.raises(TensorizerError, match="resolution"):
        tensorize_context(make_snapshot(resolution=(5, 5)))


def test_tensorize_context_ignores_optional_extra_camera_inputs() -> None:
    prompt = PhysicalPrompt((make_frame(0.0, value=1), make_frame(3.0, value=2)))
    context = Gen15Context(prompt, policy_hz=1.0)
    context.append_live(make_frame(4.0, value=10, extra_camera=True))

    batch = tensorize_context(context.snapshot())

    assert batch.live_images.shape[2] == 2
    assert int(batch.live_images[0, 0, 0, 0, 0, 0]) == 10
    assert int(batch.live_images[0, 0, 1, 0, 0, 0]) == 20


def test_tensorize_context_has_no_source_array_aliases_and_frozen_batch() -> None:
    snapshot = make_snapshot()
    batch = tensorize_context(snapshot)

    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.live_images = torch.zeros_like(batch.live_images)  # type: ignore[misc]
    with pytest.raises(TypeError):
        batch.as_kwargs()["live_images"] = batch.live_images

    original = int(batch.live_images[0, 0, 0, 0, 0, 0])
    source_array = snapshot.live_frames[0].images["left_wrist"]
    with pytest.raises(ValueError):
        source_array[0, 0, 0] = 255
    assert int(batch.live_images[0, 0, 0, 0, 0, 0]) == original


def test_tensorize_context_compacts_large_wrist_frames_before_model() -> None:
    resolution = (96, 128)
    prompt = PhysicalPrompt(
        (
            make_frame(0.0, value=1, resolution=resolution),
            make_frame(3.0, value=2, resolution=resolution),
        )
    )
    context = Gen15Context(prompt, policy_hz=1.0)
    context.append_live(make_frame(4.0, value=10, resolution=resolution))

    batch = tensorize_context(context.snapshot())

    assert batch.prompt_images.shape[-2:] == (48, 64)
    assert batch.live_images.shape[-2:] == (48, 64)
