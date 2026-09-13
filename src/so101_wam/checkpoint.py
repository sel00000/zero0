"""Versioned checkpoint I/O for :class:`so101_wam.model.CompactWAM`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from math import isfinite
from pathlib import Path
import os
import tempfile
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from .constants import ACTION_DIM
from .model import ActionDecoder, ActionRangeConstraint, CompactWAM, ModelContractError


CHECKPOINT_SCHEMA_VERSION = 4
CHECKPOINT_FORMAT = "so101_wam.compact_wam"
_LEGACY_SCHEMA_VERSION = 2
_UNBOUNDED_SCHEMA_VERSION = 3
_BOUNDED_SCHEMA_VERSION = 4
_ACTION_RANGE_CONSTRAINT_KEY = "action_range_constraint"
_ACTION_LOWER_KEY = "action_lower"
_ACTION_UPPER_KEY = "action_upper"
_UNBOUNDED_CONSTRAINT = "unbounded"
_AFFINE_TANH_CONSTRAINT = "affine_tanh"
_NORMALIZED_CLAMP_CONSTRAINT = "normalized_clamp"
_BOUNDED_CONSTRAINTS = {
    _AFFINE_TANH_CONSTRAINT,
    _NORMALIZED_CLAMP_CONSTRAINT,
}

_ARCHITECTURE_KEYS = (
    "latent_dim",
    "transformer_layers",
    "transformer_heads",
    "future_steps",
    "action_horizon",
    "action_history_steps",
    "ifp_steps",
    "max_context_steps",
)


class CheckpointError(ValueError):
    """Raised when a CompactWAM checkpoint cannot be trusted or loaded."""


@dataclass(frozen=True, slots=True)
class CompactWAMCheckpoint:
    model: CompactWAM
    metadata: Mapping[str, str | int | float | bool | None]
    architecture: Mapping[str, int | str]


def _torch_map_location(
    value: str | torch.device | Mapping[str, str] | None,
) -> str | torch.device | dict[str, str] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return value


def compact_wam_architecture(model: CompactWAM) -> dict[str, int | str]:
    """Return constructor metadata required to recreate ``model`` exactly."""

    architecture: dict[str, int | str] = {
        "latent_dim": _positive_int(model.latent_dim, "latent_dim"),
        "transformer_layers": _positive_int(len(model.temporal.layers), "transformer_layers"),
        "transformer_heads": _positive_int(model.temporal.layers[0].self_attn.num_heads, "transformer_heads"),
        "future_steps": _positive_int(model.future_steps, "future_steps"),
        "action_horizon": _positive_int(model.action_horizon, "action_horizon"),
        "action_history_steps": _positive_int(model.action_history_steps, "action_history_steps"),
        "ifp_steps": _nonnegative_int(model.ifp_steps, "ifp_steps"),
        "max_context_steps": _positive_int(model.max_context_steps, "max_context_steps"),
        "action_decoder": model.action_decoder.value,
    }
    constraint = _action_range_value(model)
    if constraint in _BOUNDED_CONSTRAINTS:
        architecture[_ACTION_RANGE_CONSTRAINT_KEY] = constraint
    return architecture


def save_compact_wam_checkpoint(
    model: CompactWAM,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a CompactWAM checkpoint with strict architecture metadata."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = _validate_metadata(metadata or {})
    architecture = compact_wam_architecture(model)
    schema_version = (
        _BOUNDED_SCHEMA_VERSION
        if architecture.get(_ACTION_RANGE_CONSTRAINT_KEY) in _BOUNDED_CONSTRAINTS
        else _UNBOUNDED_SCHEMA_VERSION
    )
    state_dict = model.state_dict()
    _validate_metadata_constraint(metadata, architecture.get(_ACTION_RANGE_CONSTRAINT_KEY, _UNBOUNDED_CONSTRAINT))
    if schema_version == _BOUNDED_SCHEMA_VERSION:
        _validate_bound_state(state_dict)
    elif _ACTION_LOWER_KEY in state_dict or _ACTION_UPPER_KEY in state_dict:
        raise CheckpointError("unbounded checkpoints must not contain action bounds")

    payload = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": schema_version,
        "model_class": "CompactWAM",
        "architecture": architecture,
        "metadata": metadata,
        "state_dict": state_dict,
    }

    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def load_compact_wam_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device | Mapping[str, str] | None = None,
    device: str | torch.device | None = None,
) -> CompactWAM:
    """Load a CompactWAM checkpoint after validating schema, architecture, and tensors."""

    return load_compact_wam_bundle(path, map_location=map_location, device=device).model


def load_compact_wam_bundle(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device | Mapping[str, str] | None = None,
    device: str | torch.device | None = None,
) -> CompactWAMCheckpoint:
    """Load a model together with deployment-auditable scalar metadata."""

    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise CheckpointError(f"failed to read checkpoint: {exc}") from exc

    return load_compact_wam_bytes(data, map_location=map_location, device=device)


def load_compact_wam_bytes(
    data: bytes,
    *,
    map_location: str | torch.device | Mapping[str, str] | None = None,
    device: str | torch.device | None = None,
) -> CompactWAMCheckpoint:
    """Load a model from exact checkpoint bytes."""

    if map_location is not None and device is not None:
        raise CheckpointError("pass either map_location or device, not both")

    location = _torch_map_location(device if device is not None else map_location)
    try:
        payload = torch.load(BytesIO(data), map_location=location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"failed to load checkpoint: {exc}") from exc

    architecture = _validate_payload(payload)
    payload_architecture = payload["architecture"]
    schema_version = payload["schema_version"]
    mode = ActionDecoder(payload["architecture"].get("action_decoder", ActionDecoder.LEGACY_MEAN.value))
    kwargs: dict[str, Any] = {**architecture, "action_decoder": mode}
    if schema_version == _BOUNDED_SCHEMA_VERSION:
        constraint = str(payload_architecture[_ACTION_RANGE_CONSTRAINT_KEY])
        lower, upper = _validate_bound_state(payload["state_dict"])
        kwargs["action_range_constraint"] = ActionRangeConstraint(constraint)
        kwargs["action_lower"] = lower
        kwargs["action_upper"] = upper
    try:
        model = CompactWAM(**kwargs)
    except ModelContractError as exc:
        raise CheckpointError(f"checkpoint architecture is invalid: {exc}") from exc
    _load_strict_state_dict(model, payload["state_dict"])
    if device is not None:
        model.to(device)
    return CompactWAMCheckpoint(
        model=model,
        metadata=MappingProxyType(_validate_metadata(payload.get("metadata", {}))),
        architecture=MappingProxyType(dict(payload["architecture"])),
    )


def _validate_payload(payload: object) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload must be a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("checkpoint format is not so101_wam.compact_wam")
    schema_version = payload.get("schema_version")
    valid_schemas = {_LEGACY_SCHEMA_VERSION, _UNBOUNDED_SCHEMA_VERSION, _BOUNDED_SCHEMA_VERSION}
    if type(schema_version) is not int or schema_version not in valid_schemas:
        raise CheckpointError(f"unsupported checkpoint schema_version={schema_version!r}")
    if payload.get("model_class") != "CompactWAM":
        raise CheckpointError("checkpoint model_class is not CompactWAM")

    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise CheckpointError("checkpoint architecture must be a mapping")
    architecture_keys = set(_ARCHITECTURE_KEYS)
    if schema_version in {_UNBOUNDED_SCHEMA_VERSION, _BOUNDED_SCHEMA_VERSION}:
        architecture_keys.add("action_decoder")
    if schema_version == _BOUNDED_SCHEMA_VERSION:
        architecture_keys.add(_ACTION_RANGE_CONSTRAINT_KEY)
    if set(architecture) != architecture_keys:
        raise CheckpointError("checkpoint architecture keys do not match CompactWAM schema")
    if schema_version in {_UNBOUNDED_SCHEMA_VERSION, _BOUNDED_SCHEMA_VERSION}:
        try:
            ActionDecoder(architecture["action_decoder"])
        except (TypeError, ValueError) as exc:
            raise CheckpointError("invalid action_decoder in checkpoint architecture") from exc
    if schema_version == _BOUNDED_SCHEMA_VERSION:
        range_constraint = architecture[_ACTION_RANGE_CONSTRAINT_KEY]
        if not isinstance(range_constraint, str) or range_constraint not in _BOUNDED_CONSTRAINTS:
            raise CheckpointError("schema v4 checkpoints require a bounded action range")
    metadata = _validate_metadata(payload.get("metadata", {}))
    _validate_metadata_constraint(metadata, architecture.get(_ACTION_RANGE_CONSTRAINT_KEY, _UNBOUNDED_CONSTRAINT))

    validated = {
        "latent_dim": _positive_int(architecture["latent_dim"], "latent_dim"),
        "transformer_layers": _positive_int(architecture["transformer_layers"], "transformer_layers"),
        "transformer_heads": _positive_int(architecture["transformer_heads"], "transformer_heads"),
        "future_steps": _positive_int(architecture["future_steps"], "future_steps"),
        "action_horizon": _positive_int(architecture["action_horizon"], "action_horizon"),
        "action_history_steps": _positive_int(architecture["action_history_steps"], "action_history_steps"),
        "ifp_steps": _nonnegative_int(architecture["ifp_steps"], "ifp_steps"),
        "max_context_steps": _positive_int(architecture["max_context_steps"], "max_context_steps"),
    }
    if "state_dict" not in payload:
        raise CheckpointError("checkpoint is missing state_dict")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise CheckpointError("checkpoint state_dict must be a mapping")
    if schema_version in {_LEGACY_SCHEMA_VERSION, _UNBOUNDED_SCHEMA_VERSION} and (
        _ACTION_LOWER_KEY in state_dict or _ACTION_UPPER_KEY in state_dict
    ):
        raise CheckpointError("legacy checkpoint contains bounded action state")
    if schema_version == _BOUNDED_SCHEMA_VERSION:
        _validate_bound_state(state_dict)
    return validated


def _load_strict_state_dict(model: CompactWAM, state_dict: object) -> None:
    if not isinstance(state_dict, Mapping):
        raise CheckpointError("checkpoint state_dict must be a mapping")

    expected = model.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise CheckpointError(f"checkpoint state_dict keys mismatch: missing={missing}, unexpected={unexpected}")

    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise CheckpointError(f"checkpoint state_dict[{key!r}] must be a tensor")
        if tuple(value.shape) != tuple(expected[key].shape):
            raise CheckpointError(
                f"checkpoint state_dict[{key!r}] shape mismatch: "
                f"got {tuple(value.shape)}, expected {tuple(expected[key].shape)}"
            )
        if value.dtype != expected[key].dtype:
            raise CheckpointError(
                f"checkpoint state_dict[{key!r}] dtype mismatch: "
                f"got {value.dtype}, expected {expected[key].dtype}"
            )
        if not bool(torch.isfinite(value).all()):
            raise CheckpointError(f"checkpoint state_dict[{key!r}] contains NaN or infinity")

    try:
        model.set_axis_normalization(state_dict["axis_mean"], state_dict["axis_scale"])
    except ModelContractError as exc:
        raise CheckpointError(f"checkpoint state_dict normalization is invalid: {exc}") from exc

    model.load_state_dict(state_dict, strict=True)


def _validate_metadata(value: object) -> dict[str, str | int | float | bool | None]:
    if not isinstance(value, Mapping):
        raise CheckpointError("checkpoint metadata must be a mapping")
    result: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise CheckpointError("checkpoint metadata keys must be non-empty strings")
        if item is not None and (not isinstance(item, (str, int, float, bool)) or isinstance(item, complex)):
            raise CheckpointError(f"checkpoint metadata[{key!r}] must be a scalar JSON value")
        if isinstance(item, float) and not isfinite(item):
            raise CheckpointError(f"checkpoint metadata[{key!r}] must be finite")
        result[key] = item
    return result


def _validate_metadata_constraint(
    metadata: Mapping[str, str | int | float | bool | None],
    constraint: object,
) -> None:
    recorded = metadata.get(_ACTION_RANGE_CONSTRAINT_KEY)
    if recorded is not None and recorded != constraint:
        raise CheckpointError("checkpoint metadata action_range_constraint contradicts architecture")


def _validate_bound_state(state_dict: object) -> tuple[Tensor, Tensor]:
    if not isinstance(state_dict, Mapping):
        raise CheckpointError("checkpoint state_dict must be a mapping")
    if _ACTION_LOWER_KEY not in state_dict or _ACTION_UPPER_KEY not in state_dict:
        raise CheckpointError("bounded checkpoint is missing action bounds")
    lower = _validate_bound_tensor(state_dict[_ACTION_LOWER_KEY], _ACTION_LOWER_KEY)
    upper = _validate_bound_tensor(state_dict[_ACTION_UPPER_KEY], _ACTION_UPPER_KEY)
    if not bool((upper > lower).all()):
        raise CheckpointError("bounded checkpoint action_upper must be greater than action_lower")
    return lower, upper


def _validate_bound_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise CheckpointError(f"checkpoint state_dict[{name!r}] must be a tensor")
    if value.shape != (ACTION_DIM,):
        raise CheckpointError(f"checkpoint state_dict[{name!r}] shape mismatch")
    if value.dtype != torch.float64:
        raise CheckpointError(f"checkpoint state_dict[{name!r}] dtype mismatch")
    if not bool(torch.isfinite(value).all()):
        raise CheckpointError(f"checkpoint state_dict[{name!r}] contains NaN or infinity")
    return value.detach().clone()


def _action_range_value(model: CompactWAM) -> str:
    constraint = model.action_range_constraint
    if not isinstance(constraint, ActionRangeConstraint):
        raise CheckpointError("invalid action_range_constraint in model")
    return constraint.value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CheckpointError(f"{name} must be a positive int")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CheckpointError(f"{name} must be a non-negative int")
    return value


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CompactWAMCheckpoint",
    "compact_wam_architecture",
    "load_compact_wam_bundle",
    "load_compact_wam_bytes",
    "load_compact_wam_checkpoint",
    "save_compact_wam_checkpoint",
]
