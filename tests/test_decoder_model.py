from __future__ import annotations

from io import BytesIO
import inspect
import json
from pathlib import Path

import pytest
import torch

from so101_wam.checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointError,
    load_compact_wam_bundle,
    load_compact_wam_bytes,
    save_compact_wam_checkpoint,
)
from so101_wam.deployment import canonical_json_sha256, file_sha256
from so101_wam import deployment_issuer
from so101_wam.model import ActionDecoder, CompactWAM, InverseDynamicsActionHead, ModelContractError


def _head(mode: ActionDecoder) -> InverseDynamicsActionHead:
    return InverseDynamicsActionHead(8, 2, 1, future_steps=3, action_decoder=mode).eval()


@pytest.mark.parametrize("mode", list(ActionDecoder))
def test_decoder_contract(mode: ActionDecoder) -> None:
    head = _head(mode)

    assert head(torch.randn(2, 3, 2, 8), torch.zeros(2, 12), torch.zeros(2, 1, 12)).shape == (2, 2, 12)
    assert list(inspect.signature(head.forward).parameters) == ["future_latents", "current_proprio", "action_history"]


def test_default_head_is_legacy() -> None:
    torch.manual_seed(7)
    default = InverseDynamicsActionHead(8, 2, 1)
    explicit = _head(ActionDecoder.LEGACY_MEAN)
    explicit.load_state_dict(default.state_dict())
    z = torch.randn(2, 5, 2, 8)
    current = torch.zeros(2, 12)
    history = torch.zeros(2, 1, 12)

    expected = default.net(torch.cat((z.mean(1).flatten(1), current, history.flatten(1)), -1))

    torch.testing.assert_close(default(z, current, history), expected.view(2, 2, 12), rtol=0, atol=0)
    torch.testing.assert_close(default(z, current, history), explicit(z, current, history), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 0, 2, 8), (1, 3, 2, 7)])
def test_legacy_latent_dimensions(shape: tuple[int, ...]) -> None:
    with pytest.raises(ModelContractError, match="latents"):
        _head(ActionDecoder.LEGACY_MEAN)(
            torch.zeros(shape),
            torch.zeros(1, 12),
            torch.zeros(1, 1, 12),
        )


def test_rejects_non_enum_decoder() -> None:
    with pytest.raises(ModelContractError, match="action_decoder must be an ActionDecoder"):
        InverseDynamicsActionHead(8, 2, 1, action_decoder="legacy_mean")


@pytest.mark.parametrize("future_steps", [None, False, 0])
def test_fixed_requires_horizon(future_steps: int | None) -> None:
    with pytest.raises(ModelContractError, match="fixed decoder requires positive future_steps"):
        InverseDynamicsActionHead(
            8,
            2,
            1,
            future_steps=future_steps,
            action_decoder=ActionDecoder.ORDERED_CONCAT,
        )


@pytest.mark.parametrize("mode", [ActionDecoder.MEAN_REPEAT_CONTROL, ActionDecoder.ORDERED_CONCAT])
@pytest.mark.parametrize("shape", [(1, 2, 2, 8), (1, 3, 2, 7)])
def test_fixed_decoder_dimensions(mode: ActionDecoder, shape: tuple[int, ...]) -> None:
    with pytest.raises(ModelContractError, match="configured future"):
        _head(mode)(torch.zeros(shape), torch.zeros(1, 12), torch.zeros(1, 1, 12))


@pytest.mark.parametrize("mode", [ActionDecoder.LEGACY_MEAN, ActionDecoder.MEAN_REPEAT_CONTROL])
def test_mean_order_invariance(mode: ActionDecoder) -> None:
    head = _head(mode)
    z = torch.randn(2, 3, 2, 8)
    current = torch.randn(2, 12)
    history = torch.randn(2, 1, 12)

    torch.testing.assert_close(head(z, current, history), head(z[:, [2, 0, 1]], current, history), rtol=1e-5, atol=1e-6)


def test_ordered_distinguishes() -> None:
    head = _head(ActionDecoder.ORDERED_CONCAT)
    with torch.no_grad():
        head.net[1].weight.zero_()
        head.net[1].bias.zero_()
        head.net[3].weight.zero_()
        head.net[3].bias.zero_()
        head.net[1].weight[0, 0] = 1
        head.net[3].weight[0, 0] = 1
    z = torch.zeros(1, 3, 2, 8)
    z[:, 0] = 1
    z[:, 2] = -1
    reverse = z.flip(1)
    current = torch.zeros(1, 12)
    history = torch.zeros(1, 1, 12)

    torch.testing.assert_close(z.mean(1), reverse.mean(1), rtol=0, atol=0)
    assert not torch.allclose(head(z, current, history), head(reverse, current, history))


def test_wide_head_shapes_match() -> None:
    control = _head(ActionDecoder.MEAN_REPEAT_CONTROL)
    ordered = _head(ActionDecoder.ORDERED_CONCAT)

    assert {k: v.shape for k, v in control.state_dict().items()} == {k: v.shape for k, v in ordered.state_dict().items()}
    assert sum(p.numel() for p in control.parameters()) == sum(p.numel() for p in ordered.parameters())


@pytest.mark.parametrize("mode", list(ActionDecoder))
def test_single_step_normal_model(mode: ActionDecoder) -> None:
    model = CompactWAM(future_steps=1, action_decoder=mode)

    assert model.action_decoder is mode


@pytest.mark.parametrize("mode", list(ActionDecoder))
def test_v3_roundtrip(tmp_path: Path, mode: ActionDecoder) -> None:
    model = CompactWAM(latent_dim=8, transformer_heads=2, action_decoder=mode)
    path = tmp_path / "model.pt"
    save_compact_wam_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == 3
    assert payload["architecture"]["action_decoder"] == mode.value
    bundle = load_compact_wam_bytes(path.read_bytes(), device="cpu")
    assert bundle.model.action_decoder is mode
    assert dict(bundle.architecture) == payload["architecture"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, bundle.model.state_dict()[name], rtol=0, atol=0)
    args = torch.randn(1, 3, 2, 8), torch.zeros(1, 12), torch.zeros(1, 4, 12)
    torch.testing.assert_close(model.action_head(*args), bundle.model.action_head(*args), rtol=0, atol=0)


def test_genuine_v2_payload(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = CompactWAM(latent_dim=8, transformer_heads=2)
    architecture = dict(latent_dim=8, transformer_layers=1, transformer_heads=2, future_steps=3, action_horizon=10, action_history_steps=4, ifp_steps=2, max_context_steps=300)
    path = tmp_path / "legacy.pt"
    torch.save(dict(format=CHECKPOINT_FORMAT, schema_version=2, model_class="CompactWAM", architecture=architecture, metadata={}, state_dict=model.state_dict()), path)
    bundle = load_compact_wam_bundle(path, device="cpu")
    assert dict(bundle.architecture) == architecture
    assert bundle.model.action_decoder is ActionDecoder.LEGACY_MEAN
    args = torch.randn(1, 3, 2, 8), torch.zeros(1, 12), torch.zeros(1, 4, 12)
    torch.testing.assert_close(model.action_head(*args), bundle.model.action_head(*args), rtol=0, atol=0)


def test_v2_candidate_report(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = CompactWAM(latent_dim=8, transformer_heads=2)
    architecture = {
        "latent_dim": 8,
        "transformer_layers": 1,
        "transformer_heads": 2,
        "future_steps": 3,
        "action_horizon": 10,
        "action_history_steps": 4,
        "ifp_steps": 2,
        "max_context_steps": 300,
    }
    markers = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "checkpoint_id": "legacy-candidate",
    }
    report_core = {
        **markers,
        "schema_version": "so101_wam.offline_training.v1",
        "result": "pass",
        "protocol": {
            "split": "task_disjoint",
            "real_output_authorized": False,
        },
        "data": {
            "train_task_count": 1,
            "validation_task_count": 1,
        },
        "model": architecture,
        "validation": {
            "action_mse_normalized": 1.0,
            "action_mae_native": 1.0,
            "future_latent_mse": 1.0,
        },
    }
    digest = canonical_json_sha256(report_core)
    checkpoint = tmp_path / "candidate.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "schema_version": 2,
            "model_class": "CompactWAM",
            "architecture": architecture,
            "metadata": {
                **markers,
                "training_evidence_sha256": digest,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    report = tmp_path / "training.json"
    report.write_text(
        json.dumps(
            {
                **report_core,
                "training_evidence_sha256": digest,
                "artifacts": {"checkpoint_sha256": file_sha256(checkpoint)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    bundle, training_digest = deployment_issuer._validate_candidate_and_training_report(
        checkpoint,
        report,
        candidate_sha256=file_sha256(checkpoint),
    )

    assert training_digest == digest
    assert dict(bundle.architecture) == architecture


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_mode",
        "unknown_mode",
        "version",
        "dimension",
        "dtype",
        "shape",
        "keys",
        "nan_mean",
        "inf_weight",
        "zero_scale",
        "negative_scale",
    ],
)
def test_v3_rejects_bad_payload(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "bad.pt"
    save_compact_wam_checkpoint(CompactWAM(), path)
    payload = torch.load(path, weights_only=True)
    if mutation == "missing_mode":
        del payload["architecture"]["action_decoder"]
    elif mutation == "unknown_mode":
        payload["architecture"]["action_decoder"] = "guessed_from_shape"
    elif mutation == "version":
        payload["schema_version"] = 4
    elif mutation == "dimension":
        payload["architecture"]["future_steps"] = True
    elif mutation == "dtype":
        payload["state_dict"]["axis_mean"] = torch.zeros(12, dtype=torch.float64)
    elif mutation == "shape":
        payload["state_dict"]["axis_mean"] = torch.zeros(11)
    elif mutation == "nan_mean":
        payload["state_dict"]["axis_mean"][0] = torch.nan
    elif mutation == "inf_weight":
        payload["state_dict"]["action_head.net.1.weight"][0, 0] = torch.inf
    elif mutation == "zero_scale":
        payload["state_dict"]["axis_scale"][0] = 0
    elif mutation == "negative_scale":
        payload["state_dict"]["axis_scale"][0] = -1
    else:
        del payload["state_dict"]["axis_mean"]
    stream = BytesIO()
    torch.save(payload, stream)
    with pytest.raises(CheckpointError):
        load_compact_wam_bytes(stream.getvalue(), device="cpu")
