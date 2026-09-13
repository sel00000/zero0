from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch

from so101_wam.constants import ACTION_DIM
from so101_wam.context import Gen15Context
from so101_wam.contracts import PhysicalPrompt, SensorimotorFrame
from so101_wam.model import CompactWAM, ModelContractError
from so101_wam.policy import CompactWAMPolicy
from so101_wam.task_specs import (
    DeterministicLanguageEncoder,
    HumanVideoFrame,
    HumanVideoPrompt,
    LanguagePrompt,
    RobotEpisodePrompt,
    TaskSpecError,
    TaskSpecKind,
    TaskSpecProvenance,
    Utf8Tokenizer,
)
from so101_wam.tensorizer import tensorize_context, tensorize_task_spec


def _rgb(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _human_prompt(value: int, *, source_id: str) -> HumanVideoPrompt:
    return HumanVideoPrompt(
        frames=(
            HumanVideoFrame(timestamp_s=0.0, rgb=_rgb(value)),
            HumanVideoFrame(timestamp_s=3.0, rgb=_rgb(value + 1)),
        ),
        provenance=TaskSpecProvenance(source_id=source_id),
        text_metadata="place the cup",
    )


def _robot_frame(timestamp_s: float, value: int) -> SensorimotorFrame:
    axes = np.full(ACTION_DIM, value, dtype=np.float32)
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={"left_wrist": _rgb(value), "right_wrist": _rgb(value + 1)},
        joint_position=axes,
        executed_action=axes + 0.5,
    )


def _robot_prompt() -> PhysicalPrompt:
    return PhysicalPrompt((_robot_frame(0.0, 1), _robot_frame(3.0, 2)))


def _live_frames() -> tuple[SensorimotorFrame, ...]:
    return tuple(_robot_frame(4.0 + index, 10 + index) for index in range(4))


def test_human_video_contract_structurally_forbids_robot_signals() -> None:
    prompt = _human_prompt(1, source_id="human-demo-001")
    repeated = _human_prompt(1, source_id="human-demo-copy")

    assert {field.name for field in fields(HumanVideoFrame)} == {
        "timestamp_s",
        "rgb",
    }
    forbidden_fields = {
        "joint_position",
        "joint_state",
        "executed_action",
        "action",
    }
    assert not forbidden_fields & {
        field.name for field in fields(HumanVideoPrompt)
    }
    assert not forbidden_fields & {field.name for field in fields(LanguagePrompt)}
    assert prompt.fingerprint == repeated.fingerprint
    assert prompt.provenance.source_id == "human-demo-001"
    assert prompt.duration_s == pytest.approx(3.0)
    assert prompt.frames[0].rgb.flags.writeable is False

    with pytest.raises(TypeError, match="joint_position"):
        HumanVideoFrame(  # type: ignore[call-arg]
            timestamp_s=0.0,
            rgb=_rgb(1),
            joint_position=np.zeros(ACTION_DIM),
        )


def test_task_spec_contracts_reject_invalid_content_and_provenance() -> None:
    with pytest.raises(TaskSpecError, match="strictly increasing"):
        HumanVideoPrompt(
            frames=(
                HumanVideoFrame(timestamp_s=0.0, rgb=_rgb(1)),
                HumanVideoFrame(timestamp_s=0.0, rgb=_rgb(2)),
            ),
            provenance=TaskSpecProvenance(source_id="bad-time"),
        )
    with pytest.raises(TaskSpecError, match="non-empty"):
        LanguagePrompt(
            text="  ",
            provenance=TaskSpecProvenance(source_id="empty-language"),
        )
    with pytest.raises(TaskSpecError, match="SHA-256"):
        TaskSpecProvenance(source_id="bad-hash", source_sha256="1234")
    with pytest.raises(TaskSpecError, match="provenance"):
        LanguagePrompt(text="place the cup", provenance="untracked")  # type: ignore[arg-type]


def test_language_tokenizer_and_encoder_are_deterministic_and_bounded() -> None:
    prompt = LanguagePrompt(
        text="place the red cup",
        provenance=TaskSpecProvenance(
            source_id="operator-text-001",
            source_sha256="a" * 64,
        ),
    )
    tokenizer = Utf8Tokenizer()
    encoder = DeterministicLanguageEncoder()

    tokens = tokenizer.tokenize(prompt.text)
    repeated = tokenizer.tokenize(prompt.text)
    embedding = encoder.encode(tokens, width=16)

    assert tokens == repeated
    assert tokens != tokenizer.tokenize("place the blue cup")
    assert embedding.shape == (16,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert embedding.flags.writeable is False
    assert not np.array_equal(
        embedding,
        encoder.encode(tokenizer.tokenize("place the blue cup"), width=16),
    )


def test_task_spec_tensorizer_uses_neutral_axes_without_human_leakage() -> None:
    neutral = np.arange(ACTION_DIM, dtype=np.float32)
    human = tensorize_task_spec(
        _human_prompt(3, source_id="human-demo-003"),
        _live_frames(),
        neutral_axes=neutral,
    )
    language = tensorize_task_spec(
        LanguagePrompt(
            text="place the cup",
            provenance=TaskSpecProvenance(source_id="language-001"),
        ),
        _live_frames(),
        neutral_axes=neutral,
    )

    assert human.kind is TaskSpecKind.HUMAN_VIDEO
    assert human.batch.prompt_images.shape == (1, 2, 2, 3, 8, 8)
    assert torch.equal(
        human.batch.prompt_images[:, :, 0],
        human.batch.prompt_images[:, :, 1],
    )
    assert torch.equal(
        human.batch.prompt_proprio,
        torch.from_numpy(neutral).view(1, 1, -1).expand(1, 2, -1),
    )
    assert torch.equal(human.batch.prompt_actions, human.batch.prompt_proprio)
    assert human.language_text is None

    assert language.kind is TaskSpecKind.LANGUAGE
    assert language.batch.prompt_images.shape == (1, 1, 2, 3, 8, 8)
    assert not bool(language.batch.prompt_images.any())
    assert language.language_text == "place the cup"
    assert torch.equal(language.batch.prompt_actions, language.batch.prompt_proprio)


def test_robot_task_spec_preserves_existing_tensor_and_policy_behavior() -> None:
    torch.manual_seed(11)
    prompt = _robot_prompt()
    live_frames = _live_frames()
    context = Gen15Context(prompt, policy_hz=1.0)
    context.extend_live(live_frames)
    snapshot = context.snapshot()
    spec = RobotEpisodePrompt(
        prompt=prompt,
        provenance=TaskSpecProvenance(source_id="robot-episode-001"),
    )
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=3,
        action_history_steps=4,
    )
    policy = CompactWAMPolicy(model, servo_hz=50.0)

    old_batch = tensorize_context(snapshot)
    new_batch = tensorize_task_spec(
        spec,
        live_frames,
        neutral_axes=np.zeros(ACTION_DIM, dtype=np.float32),
    ).batch
    old_chunk = policy.predict(snapshot, now_s=8.0)
    new_chunk = policy.predict_task(spec, live_frames, now_s=8.0)

    for name in old_batch.as_kwargs():
        assert torch.equal(old_batch.as_kwargs()[name], new_batch.as_kwargs()[name])
    np.testing.assert_array_equal(
        old_chunk.target_joint_position,
        new_chunk.target_joint_position,
    )


def test_human_and_language_specs_share_one_offline_policy_boundary() -> None:
    torch.manual_seed(17)
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=3,
        action_history_steps=4,
    )
    policy = CompactWAMPolicy(model, servo_hz=50.0)
    live_frames = _live_frames()

    human_matched = policy.predict_task(
        _human_prompt(5, source_id="human-matched"),
        live_frames,
        now_s=8.0,
    )
    human_mismatched = policy.predict_task(
        _human_prompt(25, source_id="human-mismatched"),
        live_frames,
        now_s=8.0,
    )
    language_matched = policy.predict_task(
        LanguagePrompt(
            text="place the red cup",
            provenance=TaskSpecProvenance(source_id="language-matched"),
        ),
        live_frames,
        now_s=8.0,
    )
    language_mismatched = policy.predict_task(
        LanguagePrompt(
            text="open the blue drawer",
            provenance=TaskSpecProvenance(source_id="language-mismatched"),
        ),
        live_frames,
        now_s=8.0,
    )

    for chunk in (
        human_matched,
        human_mismatched,
        language_matched,
        language_mismatched,
    ):
        assert chunk.target_joint_position.shape == (3, ACTION_DIM)
        assert np.isfinite(chunk.target_joint_position).all()
    assert not np.array_equal(
        human_matched.target_joint_position,
        human_mismatched.target_joint_position,
    )
    assert not np.array_equal(
        language_matched.target_joint_position,
        language_mismatched.target_joint_position,
    )


def test_model_rejects_invalid_task_conditioning_without_state_changes() -> None:
    model = CompactWAM(latent_dim=16, transformer_heads=4)
    prompt = torch.zeros((1, 2, 2, 3, 8, 8), dtype=torch.uint8)
    live = torch.zeros((1, 4, 2, 3, 8, 8), dtype=torch.uint8)
    prompt_axes = torch.zeros((1, 2, ACTION_DIM))
    live_axes = torch.zeros((1, 4, ACTION_DIM))
    state_keys = tuple(model.state_dict())

    with pytest.raises(ModelContractError, match="task_conditioning"):
        model.infer_action(
            prompt,
            prompt_axes,
            prompt_axes,
            live,
            live_axes,
            live_axes,
            task_conditioning=torch.zeros((1, 15)),
        )

    assert tuple(model.state_dict()) == state_keys
