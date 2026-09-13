from __future__ import annotations

import inspect
from io import BytesIO

import pytest
import torch

from so101_wam.checkpoint import (
    CHECKPOINT_FORMAT,
    compact_wam_architecture,
    load_compact_wam_bytes,
)
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_COUNT
from so101_wam.decoder_evaluation import named_tensor_hashes
from so101_wam.model import ActionDecoder, CompactWAM, InverseDynamicsActionHead, ModelContractError, wam_loss


def make_batch(
    *,
    batch: int = 2,
    prompt_steps: int = 3,
    live_steps: int = 4,
    views: int = PRIMARY_CAMERA_COUNT,
    size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, 256, (batch, prompt_steps, views, 3, size, size), dtype=torch.uint8)
    live = torch.randint(0, 256, (batch, live_steps, views, 3, size, size), dtype=torch.uint8)
    prompt_proprio = torch.randn(batch, prompt_steps, ACTION_DIM)
    prompt_actions = torch.randn(batch, prompt_steps, ACTION_DIM)
    live_proprio = torch.randn(batch, live_steps, ACTION_DIM)
    live_actions = torch.randn(batch, live_steps, ACTION_DIM)
    prompt_mask = torch.ones(batch, prompt_steps, dtype=torch.bool)
    return prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask


def make_model() -> CompactWAM:
    return CompactWAM(latent_dim=16, transformer_heads=4, future_steps=2, action_horizon=5, action_history_steps=4)


def test_forward_shapes_and_loss_terms() -> None:
    torch.manual_seed(1)
    model = make_model()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask = make_batch()

    outputs = model(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)

    assert outputs["future_latents"].shape == (2, 2, PRIMARY_CAMERA_COUNT, 16)
    assert outputs["actions"].shape == (2, 5, ACTION_DIM)
    assert outputs["ifp_latents"].shape == (2, 2, PRIMARY_CAMERA_COUNT, 16)
    assert model.temporal_position_embedding.num_embeddings >= 300

    losses = wam_loss(
        outputs,
        target_future_latents=torch.zeros_like(outputs["future_latents"]),
        target_actions=torch.zeros_like(outputs["actions"]),
        target_ifp_latents=torch.zeros_like(outputs["ifp_latents"]),
    )
    assert set(losses) == {"future_latent", "action", "ifp", "total"}
    assert losses["total"].requires_grad


def test_axis_normalization_round_trip_and_scaled_action_loss() -> None:
    model = make_model()
    mean = torch.arange(ACTION_DIM, dtype=torch.float32)
    scale = torch.arange(1, ACTION_DIM + 1, dtype=torch.float32)
    model.set_axis_normalization(mean, scale)
    raw = mean.view(1, 1, -1) + 2.0 * scale.view(1, 1, -1)

    normalized = model.normalize_axes(raw)
    assert torch.allclose(normalized, torch.full_like(normalized, 2.0))
    assert torch.allclose(model.denormalize_axes(normalized), raw)

    future = torch.zeros(1, 2, PRIMARY_CAMERA_COUNT, model.latent_dim)
    actions = torch.zeros(1, model.action_horizon, ACTION_DIM)
    targets = scale.view(1, 1, -1).expand_as(actions)
    losses = wam_loss(
        {"future_latents": future, "actions": actions, "ifp_latents": None},
        target_future_latents=future,
        target_actions=targets,
        action_scale=scale,
    )
    assert float(losses["action"]) == pytest.approx(1.0)


@pytest.mark.parametrize("views", [1, 3])
def test_rejects_missing_or_extra_wrist_views(views: int) -> None:
    model = make_model()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask = make_batch(views=views)

    with pytest.raises(ModelContractError, match="exactly 2 wrist views"):
        model(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)


def test_rejects_non_causal_prompt_mask_order() -> None:
    model = make_model()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, _ = make_batch()
    prompt_mask = torch.tensor([[True, False, True], [True, True, True]])

    with pytest.raises(ModelContractError, match="left-packed causal order"):
        model(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)


def test_rejects_wrong_or_unaligned_sensorimotor_tensors() -> None:
    model = make_model()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask = make_batch()

    with pytest.raises(ModelContractError, match=f"prompt_proprio.*{ACTION_DIM}"):
        model(
            prompt,
            torch.randn(2, 3, ACTION_DIM - 1),
            prompt_actions,
            live,
            live_proprio,
            live_actions,
            prompt_mask=prompt_mask,
        )

    with pytest.raises(ModelContractError, match="live_actions must align"):
        model(
            prompt,
            prompt_proprio,
            prompt_actions,
            live,
            live_proprio,
            torch.randn(2, 3, ACTION_DIM),
            prompt_mask=prompt_mask,
        )

    with pytest.raises(ModelContractError, match="live_actions must contain at least 4 steps"):
        short_live = live[:, :3]
        model(
            prompt,
            prompt_proprio,
            prompt_actions,
            short_live,
            live_proprio[:, :3],
            live_actions[:, :3],
            prompt_mask=prompt_mask,
        )


def test_infer_action_uses_no_grad_and_is_deterministic_in_eval() -> None:
    torch.manual_seed(2)
    model = make_model()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask = make_batch()

    first = model.infer_action(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)
    second = model.infer_action(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=prompt_mask,
    )

    assert first.shape == (2, 5, ACTION_DIM)
    assert not first.requires_grad
    assert torch.equal(first, second)

    model.eval()
    eval_first = model(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)[
        "actions"
    ]
    eval_second = model(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)[
        "actions"
    ]
    assert torch.equal(eval_first, eval_second)


def test_cached_prompt_encoding_matches_uncached_inference() -> None:
    torch.manual_seed(6)
    model = make_model()
    inputs = make_batch()
    (
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask,
    ) = inputs
    prompt_mask[:, -1] = False

    prompt_encoding = model.encode_prompt(
        prompt,
        prompt_proprio,
        prompt_actions,
        prompt_mask=prompt_mask,
    )
    cached = model.infer_action_cached(
        prompt_encoding,
        live,
        live_proprio,
        live_actions,
    )
    uncached = model.infer_action(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=prompt_mask,
    )

    assert torch.equal(cached, uncached)

    with pytest.raises(ModelContractError, match="another model"):
        make_model().infer_action_cached(
            prompt_encoding,
            live,
            live_proprio,
            live_actions,
        )


@pytest.mark.parametrize("mode", list(ActionDecoder))
def test_decoder_inference_paths(
    monkeypatch: pytest.MonkeyPatch,
    mode: ActionDecoder,
) -> None:
    torch.manual_seed(8)
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=5,
        action_history_steps=4,
        action_decoder=mode,
    )
    inputs = make_batch()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, mask = inputs
    before = named_tensor_hashes(model)

    def fail_ifp(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("ifp_head must not run during inference")

    assert model.ifp_head is not None
    monkeypatch.setattr(model.ifp_head, "forward", fail_ifp)

    encoding = model.encode_prompt(
        prompt,
        prompt_proprio,
        prompt_actions,
        prompt_mask=mask,
    )
    cached = model.infer_action_cached(encoding, live, live_proprio, live_actions)
    uncached = model.infer_action(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=mask,
    )

    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    assert not cached.requires_grad
    assert not uncached.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    assert named_tensor_hashes(model) == before


def test_v2_inference_parity() -> None:
    torch.manual_seed(9)
    model = make_model()
    architecture = compact_wam_architecture(model)
    architecture.pop("action_decoder")
    assert len(architecture) == 8
    stream = BytesIO()
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "schema_version": 2,
            "model_class": "CompactWAM",
            "architecture": architecture,
            "metadata": {},
            "state_dict": model.state_dict(),
        },
        stream,
    )
    restored = load_compact_wam_bytes(stream.getvalue()).model
    inputs = make_batch()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, mask = inputs

    original = model.infer_action(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=mask,
    )
    loaded = restored.infer_action(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=mask,
    )

    torch.testing.assert_close(original, loaded, rtol=0, atol=0)


def test_cached_prompt_encoding_can_return_full_inference_outputs() -> None:
    torch.manual_seed(6)
    model = make_model()
    (
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask,
    ) = make_batch()
    model.eval()
    prompt_encoding = model.encode_prompt(
        prompt,
        prompt_proprio,
        prompt_actions,
        prompt_mask=prompt_mask,
    )

    cached = model.infer_outputs_cached(
        prompt_encoding,
        live,
        live_proprio,
        live_actions,
    )
    uncached = model(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=prompt_mask,
        compute_ifp=False,
    )

    assert set(cached) == {"future_latents", "actions", "ifp_latents"}
    torch.testing.assert_close(cached["future_latents"], uncached["future_latents"])
    torch.testing.assert_close(cached["actions"], uncached["actions"])
    assert cached["ifp_latents"] is None
    assert not cached["future_latents"].requires_grad


def test_prompt_actions_affect_predicted_future_latents() -> None:
    torch.manual_seed(4)
    model = make_model()
    model.eval()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask = make_batch()

    baseline = model(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=prompt_mask,
    )["future_latents"]
    changed_prompt_actions = prompt_actions.clone()
    changed_prompt_actions[:, 0, :] += 10.0
    changed = model(
        prompt,
        prompt_proprio,
        changed_prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=prompt_mask,
    )["future_latents"]

    assert not torch.allclose(baseline, changed)


def test_action_head_signature_excludes_prompt_embedding() -> None:
    signature = inspect.signature(InverseDynamicsActionHead.forward)

    assert list(signature.parameters) == ["self", "future_latents", "current_proprio", "action_history"]


def test_ifp_is_training_only_and_ignored_in_inference() -> None:
    torch.manual_seed(3)
    model = make_model()
    prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask = make_batch()

    model.train()
    train_outputs = model(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)
    assert train_outputs["ifp_latents"] is not None

    model.eval()
    eval_outputs = model(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)
    assert eval_outputs["ifp_latents"] is None

    action = model.infer_action(prompt, prompt_proprio, prompt_actions, live, live_proprio, live_actions, prompt_mask=prompt_mask)
    assert action.shape == (2, 5, ACTION_DIM)

    no_ifp_outputs = model(
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask=prompt_mask,
        compute_ifp=False,
    )
    losses = wam_loss(
        no_ifp_outputs,
        target_future_latents=torch.zeros_like(no_ifp_outputs["future_latents"]),
        target_actions=torch.zeros_like(no_ifp_outputs["actions"]),
        target_ifp_latents=torch.randn(2, 2, PRIMARY_CAMERA_COUNT, 16),
    )
    assert losses["ifp"].item() == 0.0
