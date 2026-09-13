from __future__ import annotations

from pathlib import Path

import pytest
import torch

from so101_wam.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    compact_wam_architecture,
    load_compact_wam_bundle,
    load_compact_wam_checkpoint,
    save_compact_wam_checkpoint,
)
from so101_wam.model import CompactWAM


def make_model() -> CompactWAM:
    return CompactWAM(
        latent_dim=16,
        transformer_layers=2,
        transformer_heads=4,
        future_steps=2,
        action_horizon=5,
        action_history_steps=4,
        ifp_steps=1,
        max_context_steps=300,
    )


def test_round_trip_preserves_architecture_state_and_device(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = make_model()
    path = tmp_path / "wam.pt"

    save_compact_wam_checkpoint(model, path, metadata={"run_id": "unit"})
    loaded = load_compact_wam_checkpoint(path, device=torch.device("cpu"))

    assert isinstance(loaded, CompactWAM)
    assert compact_wam_architecture(loaded) == compact_wam_architecture(model)
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)

    bundle = load_compact_wam_bundle(path)
    assert bundle.metadata == {"run_id": "unit"}


def test_load_rejects_non_mapping_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(["not", "a", "checkpoint"], path)

    with pytest.raises(CheckpointError, match="payload must be a mapping"):
        load_compact_wam_checkpoint(path)


def test_load_rejects_wrong_format_and_schema(tmp_path: Path) -> None:
    model = make_model()
    path = tmp_path / "wam.pt"
    save_compact_wam_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)

    wrong_format = tmp_path / "wrong-format.pt"
    payload["format"] = "other"
    torch.save(payload, wrong_format)
    with pytest.raises(CheckpointError, match="format"):
        load_compact_wam_checkpoint(wrong_format)

    wrong_schema = tmp_path / "wrong-schema.pt"
    payload["format"] = CHECKPOINT_FORMAT
    payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    torch.save(payload, wrong_schema)
    with pytest.raises(CheckpointError, match="schema_version"):
        load_compact_wam_checkpoint(wrong_schema)


def test_load_rejects_mismatched_architecture_state_shapes(tmp_path: Path) -> None:
    model = make_model()
    path = tmp_path / "wam.pt"
    save_compact_wam_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    payload["architecture"]["latent_dim"] = 32
    mismatched = tmp_path / "mismatched.pt"
    torch.save(payload, mismatched)

    with pytest.raises(CheckpointError, match="shape mismatch"):
        load_compact_wam_checkpoint(mismatched)


def test_load_rejects_corrupted_state_tensor_shape(tmp_path: Path) -> None:
    model = make_model()
    path = tmp_path / "wam.pt"
    save_compact_wam_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    first_key = next(
        key for key, value in payload["state_dict"].items() if value.ndim > 1
    )
    payload["state_dict"][first_key] = payload["state_dict"][first_key].reshape(-1)
    corrupted = tmp_path / "corrupted.pt"
    torch.save(payload, corrupted)

    with pytest.raises(CheckpointError, match="shape mismatch"):
        load_compact_wam_checkpoint(corrupted)


def test_load_rejects_ambiguous_device_arguments(tmp_path: Path) -> None:
    path = tmp_path / "wam.pt"
    save_compact_wam_checkpoint(make_model(), path)

    with pytest.raises(CheckpointError, match="either map_location or device"):
        load_compact_wam_checkpoint(path, map_location="cpu", device="cpu")


def test_checkpoint_rejects_non_scalar_metadata(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="scalar JSON"):
        save_compact_wam_checkpoint(make_model(), tmp_path / "bad.pt", metadata={"nested": {"x": 1}})
    with pytest.raises(CheckpointError, match="finite"):
        save_compact_wam_checkpoint(make_model(), tmp_path / "nan.pt", metadata={"score": float("nan")})


def test_load_rejects_state_tensor_dtype_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "wam.pt"
    save_compact_wam_checkpoint(make_model(), path)
    payload = torch.load(path, weights_only=True)
    first_key = next(iter(payload["state_dict"]))
    payload["state_dict"][first_key] = payload["state_dict"][first_key].double()
    corrupted = tmp_path / "wrong-dtype.pt"
    torch.save(payload, corrupted)

    with pytest.raises(CheckpointError, match="dtype mismatch"):
        load_compact_wam_checkpoint(corrupted)
