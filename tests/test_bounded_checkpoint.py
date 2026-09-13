from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import torch

from so101_wam.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    compact_wam_architecture,
    load_compact_wam_bytes,
    save_compact_wam_checkpoint,
)
from so101_wam.constants import ACTION_DIM
from so101_wam.model import ActionRangeConstraint, CompactWAM


LOWER = tuple(float(-index - 1) for index in range(ACTION_DIM))
UPPER = tuple(float(index + 1) for index in range(ACTION_DIM))
BOUNDED_MODES = ("affine_tanh", "normalized_clamp")


def _bounded_model(mode_value: str = "affine_tanh") -> CompactWAM:
    return CompactWAM(
        latent_dim=8,
        transformer_heads=2,
        future_steps=2,
        action_horizon=3,
        action_history_steps=1,
        ifp_steps=0,
        action_range_constraint=ActionRangeConstraint(mode_value),
        action_lower=LOWER,
        action_upper=UPPER,
    )


def _save_payload(payload: object) -> bytes:
    stream = BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


@pytest.mark.parametrize("mode_value", BOUNDED_MODES)
def test_bounded_roundtrip(tmp_path: Path, mode_value: str) -> None:
    model = _bounded_model(mode_value)
    path = tmp_path / "bounded.pt"
    raw = torch.linspace(-8.0, 8.0, model.action_horizon * ACTION_DIM).view(1, model.action_horizon, ACTION_DIM)

    save_compact_wam_checkpoint(model, path, metadata={"action_range_constraint": mode_value})
    payload = torch.load(path, weights_only=True)
    bundle = load_compact_wam_bytes(path.read_bytes(), device="cpu")

    assert CHECKPOINT_SCHEMA_VERSION == 4
    assert payload["schema_version"] == 4
    assert payload["architecture"]["action_range_constraint"] == mode_value
    assert dict(bundle.architecture) == payload["architecture"]
    assert bundle.model.action_range_constraint is ActionRangeConstraint(mode_value)
    torch.testing.assert_close(bundle.model.action_lower, torch.tensor(LOWER, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(bundle.model.action_upper, torch.tensor(UPPER, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(bundle.model.decode_actions(raw), model.decode_actions(raw), rtol=0, atol=0)


def test_unbounded_save_schema3(tmp_path: Path) -> None:
    path = tmp_path / "unbounded.pt"

    save_compact_wam_checkpoint(CompactWAM(latent_dim=8, transformer_heads=2), path)
    payload = torch.load(path, weights_only=True)
    bundle = load_compact_wam_bytes(path.read_bytes())

    assert payload["schema_version"] == 3
    assert "action_range_constraint" not in payload["architecture"]
    assert "action_lower" not in payload["state_dict"]
    assert "action_upper" not in payload["state_dict"]
    assert dict(bundle.architecture) == payload["architecture"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_arch_constraint",
        "unbounded_arch_constraint",
        "unknown_arch_constraint",
        "list_arch_constraint",
        "dict_arch_constraint",
        "none_arch_constraint",
        "missing_lower",
        "missing_upper",
        "lower_shape",
        "upper_dtype",
        "nan_lower",
        "inverted_bounds",
        "huge_bounds",
        "metadata_contradiction",
        "mode_metadata",
    ],
)
def test_v4_rejects_bad_bounds(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "bounded.pt"
    save_compact_wam_checkpoint(_bounded_model(), path, metadata={"action_range_constraint": "affine_tanh"})
    payload = torch.load(path, weights_only=True)
    if mutation == "missing_arch_constraint":
        del payload["architecture"]["action_range_constraint"]
    elif mutation == "unbounded_arch_constraint":
        payload["architecture"]["action_range_constraint"] = "unbounded"
    elif mutation == "unknown_arch_constraint":
        payload["architecture"]["action_range_constraint"] = "new_unreviewed_mode"
    elif mutation == "list_arch_constraint":
        payload["architecture"]["action_range_constraint"] = ["normalized_clamp"]
    elif mutation == "dict_arch_constraint":
        payload["architecture"]["action_range_constraint"] = {"mode": "normalized_clamp"}
    elif mutation == "none_arch_constraint":
        payload["architecture"]["action_range_constraint"] = None
    elif mutation == "missing_lower":
        del payload["state_dict"]["action_lower"]
    elif mutation == "missing_upper":
        del payload["state_dict"]["action_upper"]
    elif mutation == "lower_shape":
        payload["state_dict"]["action_lower"] = torch.zeros(ACTION_DIM - 1, dtype=torch.float64)
    elif mutation == "upper_dtype":
        payload["state_dict"]["action_upper"] = torch.ones(ACTION_DIM, dtype=torch.float32)
    elif mutation == "nan_lower":
        payload["state_dict"]["action_lower"][0] = torch.nan
    elif mutation == "inverted_bounds":
        payload["state_dict"]["action_lower"][0] = payload["state_dict"]["action_upper"][0]
    elif mutation == "huge_bounds":
        payload["state_dict"]["action_lower"] = torch.full((ACTION_DIM,), -1e40, dtype=torch.float64)
        payload["state_dict"]["action_upper"] = torch.full((ACTION_DIM,), 1e40, dtype=torch.float64)
    elif mutation == "metadata_contradiction":
        payload["metadata"]["action_range_constraint"] = "unbounded"
    else:
        payload["architecture"]["action_range_constraint"] = "normalized_clamp"
        payload["metadata"]["action_range_constraint"] = "affine_tanh"

    with pytest.raises(CheckpointError):
        load_compact_wam_bytes(_save_payload(payload))


@pytest.mark.parametrize("schema_version", [2, 3])
def test_legacy_rejects_bounds(schema_version: int) -> None:
    model = CompactWAM(latent_dim=8, transformer_heads=2)
    architecture = compact_wam_architecture(model)
    if schema_version == 2:
        del architecture["action_decoder"]
    payload = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": schema_version,
        "model_class": "CompactWAM",
        "architecture": architecture,
        "metadata": {},
        "state_dict": model.state_dict(),
    }

    bounded_arch = {
        **payload,
        "architecture": {**architecture, "action_range_constraint": "affine_tanh"},
    }
    with pytest.raises(CheckpointError):
        load_compact_wam_bytes(_save_payload(bounded_arch))

    bounded_state = {
        **payload,
        "state_dict": {
            **model.state_dict(),
            "action_lower": torch.tensor(LOWER, dtype=torch.float64),
            "action_upper": torch.tensor(UPPER, dtype=torch.float64),
        },
    }
    with pytest.raises(CheckpointError):
        load_compact_wam_bytes(_save_payload(bounded_state))
