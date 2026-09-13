"""Numerical decoder diagnostics for offline CompactWAM audits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isfinite
import hashlib
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .constants import ACTION_DIM
from .deployment import canonical_json_sha256
from .model import ActionDecoder, CompactWAM
from .training_data import TrainingWindowSpec, materialize_training_window
from .vision import COMPACT_IMAGE_MAX_SIDE

DIAGNOSTIC_SEED = 0
DECODER_AUDIT_SCHEMA = "so101_wam.decoder_training.v1"
MEAN_RTOL = 1e-5
MEAN_ATOL = 1e-6
_ROW_KEYS = {"key", "task", "mse", "mae_native", "reverse", "permute"}
_DIAG_KEYS = {"output_mse", "error_delta"}
_DIAGNOSTICS = ("reverse", "permute")


class FutureSource(StrEnum):
    ORACLE = "stored_future_images"
    PREDICTED = "predicted_future_latents"


def named_tensor_hashes(model: nn.Module) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def reduce_decoder_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("decoder rows must be non-empty")

    seen: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if set(row) != _ROW_KEYS:
            raise ValueError("decoder row keys do not match schema")

        key = _identity(row["key"], name="key")
        if key in seen:
            raise ValueError(f"duplicate decoder row key: {key}")
        seen.add(key)

        task = _identity(row["task"], name="task")
        mse = _nonnegative(row["mse"], name="mse")
        mae = _axis_values(row["mae_native"])
        diagnostics = {
            name: _diagnostic(row[name], name=name) for name in _DIAGNOSTICS
        }
        grouped[task].append(
            {
                "key": key,
                "mse": mse,
                "mae_native": mae,
                **diagnostics,
            }
        )

    per_task: dict[str, Any] = {}
    for task in sorted(grouped):
        task_rows = grouped[task]
        diag_summary = {
            name: {
                field: _mean([item[name][field] for item in task_rows])
                for field in _DIAG_KEYS
            }
            for name in _DIAGNOSTICS
        }
        per_task[task] = {
            "windows": len(task_rows),
            "mse": _mean([item["mse"] for item in task_rows]),
            "mae_native": [
                _mean([item["mae_native"][axis] for item in task_rows])
                for axis in range(ACTION_DIM)
            ],
            **diag_summary,
        }

    task_items = list(per_task.values())
    return {
        "window_count": len(rows),
        "task_count": len(per_task),
        "task_macro_mse": _mean([item["mse"] for item in task_items]),
        "native_mae_per_axis": [
            _mean([item["mae_native"][axis] for item in task_items])
            for axis in range(ACTION_DIM)
        ],
        "per_task": per_task,
        "future_order": {
            name: {
                field: _mean(
                    [
                        item[name][field]
                        for item in task_items
                    ]
                )
                for field in _DIAG_KEYS
            }
            for name in _DIAGNOSTICS
        },
    }


def decoder_window_key(spec: TrainingWindowSpec) -> str:
    return canonical_json_sha256(
        {
            "prompt": spec.pair.prompt.fingerprint,
            "target": spec.pair.target.fingerprint,
            "anchor": spec.anchor_policy_position,
            "future_steps": spec.future_steps,
            "action_horizon": spec.action_horizon,
            "action_history_steps": spec.action_history_steps,
            "servo_hz": spec.servo_hz,
            "ifp_steps": spec.ifp_steps,
            "ifp_stride": spec.ifp_stride,
        }
    )


def order_schedule(windows: Sequence[TrainingWindowSpec]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(DIAGNOSTIC_SEED)
    schedule: list[dict[str, Any]] = []
    for spec in windows:
        if spec.future_steps < 2:
            raise ValueError("future_steps must be at least 2")

        identity = np.arange(spec.future_steps)
        permutation = rng.permutation(spec.future_steps)
        if bool(np.array_equal(permutation, identity)):
            permutation = np.roll(identity, 1)
        schedule.append(
            {
                "key": decoder_window_key(spec),
                "permutation": permutation.tolist(),
            }
        )
    return schedule


@torch.no_grad()
def evaluate_decoder(
    model: CompactWAM,
    windows: Sequence[TrainingWindowSpec],
    *,
    source: FutureSource,
) -> dict[str, Any]:
    if not isinstance(source, FutureSource):
        raise ValueError("source must be a FutureSource")

    schedule = order_schedule(windows)
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    module_modes = {module: module.training for module in model.modules()}

    with torch.random.fork_rng(devices=devices):
        model.eval()
        try:
            rows = [
                _evaluate_window(model, spec, item, source=source, device=device)
                for spec, item in zip(windows, schedule, strict=True)
            ]
        finally:
            for module, training in module_modes.items():
                module.training = training

    return {
        "source": source.value,
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "permutation_sha256": canonical_json_sha256(schedule),
        "summary": reduce_decoder_rows(rows),
        "rows": rows,
    }


def _evaluate_window(
    model: CompactWAM,
    spec: TrainingWindowSpec,
    scheduled: Mapping[str, Any],
    *,
    source: FutureSource,
    device: torch.device,
) -> dict[str, Any]:
    batch = materialize_training_window(
        spec,
        image_max_side=COMPACT_IMAGE_MAX_SIDE,
    ).to(device)
    if source is FutureSource.ORACLE:
        latents = model.encode_future_images(batch.target_future_images)
    else:
        outputs = model(**dict(batch.model_kwargs()), compute_ifp=False)
        latents = outputs["future_latents"]
        if latents is None:
            raise ValueError("model did not return future latents")

    current = model.normalize_axes(batch.live_proprio[:, -1, :])
    history = model.normalize_axes(batch.live_actions[:, -model.action_history_steps :, :])
    target = model.normalize_axes(batch.target_actions)
    prediction = model.action_head(latents, current, history)
    mse = _metric((prediction - target).square().mean(), name="mse")
    native = model.denormalize_axes(prediction)
    axis_mae = _axis_values(
        (native - batch.target_actions).abs().mean(dim=(0, 1)).cpu().tolist()
    )

    row = {
        "key": _identity(scheduled["key"], name="key"),
        "task": f"{spec.pair.target.data.task_index}:{spec.pair.target.data.task}",
        "mse": mse,
        "mae_native": axis_mae,
    }
    base_mse = mse
    row["reverse"] = _order_diag(
        model,
        latents.flip(1),
        current,
        history,
        target,
        prediction,
        base_mse,
    )
    row["permute"] = _order_diag(
        model,
        latents[:, scheduled["permutation"]],
        current,
        history,
        target,
        prediction,
        base_mse,
    )
    return row


def _order_diag(
    model: CompactWAM,
    latents: Tensor,
    current: Tensor,
    history: Tensor,
    target: Tensor,
    prediction: Tensor,
    base_mse: float,
) -> dict[str, float]:
    altered = model.action_head(latents, current, history)
    if model.action_decoder is not ActionDecoder.ORDERED_CONCAT:
        torch.testing.assert_close(
            prediction,
            altered,
            rtol=MEAN_RTOL,
            atol=MEAN_ATOL,
        )
    output_mse = _metric((altered - prediction).square().mean(), name="output_mse")
    altered_mse = _metric((altered - target).square().mean(), name="altered_mse")
    return {
        "output_mse": output_mse,
        "error_delta": _number(altered_mse - base_mse, name="error_delta"),
    }


def _number(value: Any, *, name: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative(value: Any, *, name: str) -> float:
    result = _number(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _axis_values(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != ACTION_DIM:
        raise ValueError(f"mae_native must contain {ACTION_DIM} values")
    return [_nonnegative(item, name="mae_native") for item in value]


def _diagnostic(value: Any, *, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _DIAG_KEYS:
        raise ValueError(f"{name} diagnostic keys do not match schema")
    return {
        "output_mse": _nonnegative(value["output_mse"], name=f"{name}.output_mse"),
        "error_delta": _number(value["error_delta"], name=f"{name}.error_delta"),
    }


def _metric(value: Tensor, *, name: str) -> float:
    return _number(float(value.detach().cpu()), name=name)


def _mean(values: Sequence[float]) -> float:
    return float(mean(values))


__all__ = [
    "DECODER_AUDIT_SCHEMA",
    "DIAGNOSTIC_SEED",
    "FutureSource",
    "MEAN_ATOL",
    "MEAN_RTOL",
    "decoder_window_key",
    "evaluate_decoder",
    "named_tensor_hashes",
    "order_schedule",
    "reduce_decoder_rows",
]
