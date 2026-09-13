"""Strict pre-result plans for local multi-seed IFP diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
import re
from typing import Any

from .training import PAPER_IFP_STEPS, PAPER_IFP_STRIDE


IFP_PREREGISTRATION_SCHEMA = 1
IFP_PREREGISTRATION_CLAIM_SCOPE = "preregistered_local_ifp_diagnostic_only"
IFP_VARIANT_SET = (0, 2, PAPER_IFP_STEPS)
IFP_PRIMARY_ENDPOINT = "closed_loop.success_rate"
IFP_SECONDARY_ENDPOINT = "closed_loop.terminal_error_mean"

_PLAN_FIELDS = {
    "schema_version",
    "hypothesis_id",
    "claim_scope",
    "seeds",
    "train_split_sha256",
    "validation_split_sha256",
    "variant_set",
    "optimizer_protocol",
    "closed_loop",
    "endpoints",
    "margins",
}
_OPTIMIZER_PROTOCOL_FIELDS = {
    "policy_hz",
    "servo_hz",
    "latent_dim",
    "transformer_layers",
    "transformer_heads",
    "future_steps",
    "action_horizon",
    "action_history_steps",
    "ifp_stride",
    "ifp_architecture",
    "ifp_window_steps",
    "max_context_steps",
    "stage1_steps",
    "stage2_steps",
    "sampling_strategy",
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "future_latent_weight",
    "action_weight",
    "ifp_weight",
    "device",
    "optimizer",
}
_POSITIVE_FLOAT_FIELDS = {
    "policy_hz",
    "servo_hz",
    "learning_rate",
    "max_grad_norm",
    "future_latent_weight",
    "action_weight",
    "ifp_weight",
}
_NONNEGATIVE_FLOAT_FIELDS = {"weight_decay"}
_POSITIVE_INTEGER_FIELDS = {
    "latent_dim",
    "transformer_layers",
    "transformer_heads",
    "future_steps",
    "action_horizon",
    "action_history_steps",
    "ifp_stride",
    "ifp_window_steps",
    "max_context_steps",
    "stage2_steps",
}
_NONNEGATIVE_INTEGER_FIELDS = {"stage1_steps"}
_CLOSED_LOOP_FIELDS = {"scope", "minimum_trials_per_seed_variant"}
_ENDPOINT_FIELDS = {"primary", "secondary"}
_MARGIN_FIELDS = {
    "primary_min_delta_k4_vs_k0",
    "secondary_max_delta_k4_vs_k0",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class IFPPreregistrationError(ValueError):
    """Raised when an IFP study plan or endpoint evaluation is invalid."""


@dataclass(frozen=True, slots=True)
class IFPPreregistration:
    hypothesis_id: str
    seeds: tuple[int, ...]
    train_split_sha256: str
    validation_split_sha256: str
    variant_set: tuple[int, ...]
    optimizer_protocol: dict[str, object]
    closed_loop_scope: str
    minimum_trials_per_seed_variant: int
    primary_min_delta_k4_vs_k0: float
    secondary_max_delta_k4_vs_k0: float


def _json_object(source: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, value in pairs:
            if name in payload:
                raise IFPPreregistrationError(
                    f"IFP preregistration has duplicate field: {name}"
                )
            payload[name] = value
        return payload

    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IFPPreregistrationError(
            "IFP preregistration must use UTF-8"
        ) from error
    try:
        payload = json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise IFPPreregistrationError(
            "IFP preregistration must be valid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise IFPPreregistrationError(
            "IFP preregistration must be a JSON object"
        )
    return payload


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    unexpected = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unexpected:
        raise IFPPreregistrationError(
            f"{name} has unexpected fields: {unexpected}"
        )
    if missing:
        raise IFPPreregistrationError(f"{name} is missing fields: {missing}")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IFPPreregistrationError(f"{name} must be an object")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IFPPreregistrationError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if _SAFE_ID.fullmatch(result) is None:
        raise IFPPreregistrationError(f"{name} contains unsupported characters")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IFPPreregistrationError(f"{name} must be lowercase SHA-256")
    return value


def _integer(value: object, *, name: str, minimum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise IFPPreregistrationError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise IFPPreregistrationError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise IFPPreregistrationError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise IFPPreregistrationError(f"{name} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise IFPPreregistrationError(f"{name} must be at most {maximum:g}")
    return result


def _seed_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) < 3:
        raise IFPPreregistrationError("seeds must contain at least 3 values")
    seeds = tuple(_integer(item, name="seeds[]", minimum=0) for item in value)
    if len(set(seeds)) != len(seeds) or tuple(sorted(seeds)) != seeds:
        raise IFPPreregistrationError("seeds must be unique and sorted")
    return seeds


def normalize_optimizer_protocol(value: object) -> dict[str, object]:
    """Validate and normalize one result-free optimizer protocol."""

    raw = _mapping(value, name="optimizer_protocol")
    _exact_fields(
        raw,
        _OPTIMIZER_PROTOCOL_FIELDS,
        name="optimizer_protocol",
    )
    protocol = dict(raw)
    for field in _POSITIVE_FLOAT_FIELDS:
        protocol[field] = _number(
            protocol.get(field),
            name=f"optimizer_protocol.{field}",
            minimum=0.0,
        )
        if protocol[field] == 0.0:
            raise IFPPreregistrationError(
                f"optimizer_protocol.{field} must be positive"
            )
    for field in _NONNEGATIVE_FLOAT_FIELDS:
        protocol[field] = _number(
            protocol.get(field),
            name=f"optimizer_protocol.{field}",
            minimum=0.0,
        )
    for field in _POSITIVE_INTEGER_FIELDS:
        protocol[field] = _integer(
            protocol.get(field),
            name=f"optimizer_protocol.{field}",
            minimum=1,
        )
    for field in _NONNEGATIVE_INTEGER_FIELDS:
        protocol[field] = _integer(
            protocol.get(field),
            name=f"optimizer_protocol.{field}",
            minimum=0,
        )

    architecture = _string(
        protocol.get("ifp_architecture"),
        name="optimizer_protocol.ifp_architecture",
    )
    if architecture != "fused_modules":
        raise IFPPreregistrationError(
            "optimizer_protocol must use removable K=4 fused IFP"
        )
    protocol["ifp_architecture"] = architecture
    if (
        protocol["ifp_stride"] != PAPER_IFP_STRIDE
        or protocol["ifp_window_steps"] != PAPER_IFP_STEPS
    ):
        raise IFPPreregistrationError(
            "optimizer_protocol must use removable K=4 fused IFP"
        )
    sampling = _string(
        protocol.get("sampling_strategy"),
        name="optimizer_protocol.sampling_strategy",
    )
    if sampling not in {"task_balanced", "window_shuffle"}:
        raise IFPPreregistrationError(
            "optimizer_protocol.sampling_strategy is unsupported"
        )
    protocol["sampling_strategy"] = sampling
    protocol["device"] = _string(
        protocol.get("device"),
        name="optimizer_protocol.device",
    )
    optimizer = _string(
        protocol.get("optimizer"),
        name="optimizer_protocol.optimizer",
    )
    if optimizer != "AdamW":
        raise IFPPreregistrationError("optimizer_protocol.optimizer must be AdamW")
    protocol["optimizer"] = optimizer

    latent_dim = protocol["latent_dim"]
    transformer_heads = protocol["transformer_heads"]
    assert isinstance(latent_dim, int)
    assert isinstance(transformer_heads, int)
    if latent_dim % transformer_heads:
        raise IFPPreregistrationError(
            "optimizer_protocol.latent_dim must be divisible by transformer_heads"
        )
    max_context_steps = protocol["max_context_steps"]
    assert isinstance(max_context_steps, int)
    if max_context_steps < 300:
        raise IFPPreregistrationError(
            "optimizer_protocol.max_context_steps must be at least 300"
        )
    return protocol


def load_ifp_preregistration_bytes(source: bytes) -> IFPPreregistration:
    """Load a strict IFP plan containing no result-dependent hashes or values."""

    payload = _json_object(source)
    _exact_fields(payload, _PLAN_FIELDS, name="IFP preregistration")
    schema = payload.get("schema_version")
    if (
        not isinstance(schema, int)
        or isinstance(schema, bool)
        or schema != IFP_PREREGISTRATION_SCHEMA
    ):
        raise IFPPreregistrationError(
            "IFP preregistration requires schema_version=1"
        )
    if payload.get("claim_scope") != IFP_PREREGISTRATION_CLAIM_SCOPE:
        raise IFPPreregistrationError("claim_scope mismatch")

    variants = payload.get("variant_set")
    if variants != list(IFP_VARIANT_SET):
        raise IFPPreregistrationError("variant_set must be [0, 2, 4]")

    closed_loop = _mapping(payload.get("closed_loop"), name="closed_loop")
    _exact_fields(closed_loop, _CLOSED_LOOP_FIELDS, name="closed_loop")
    endpoints = _mapping(payload.get("endpoints"), name="endpoints")
    _exact_fields(endpoints, _ENDPOINT_FIELDS, name="endpoints")
    if endpoints.get("primary") != IFP_PRIMARY_ENDPOINT:
        raise IFPPreregistrationError(
            f"primary endpoint must be {IFP_PRIMARY_ENDPOINT}"
        )
    if endpoints.get("secondary") != IFP_SECONDARY_ENDPOINT:
        raise IFPPreregistrationError(
            f"secondary endpoint must be {IFP_SECONDARY_ENDPOINT}"
        )
    margins = _mapping(payload.get("margins"), name="margins")
    _exact_fields(margins, _MARGIN_FIELDS, name="margins")

    return IFPPreregistration(
        hypothesis_id=_identifier(
            payload.get("hypothesis_id"),
            name="hypothesis_id",
        ),
        seeds=_seed_tuple(payload.get("seeds")),
        train_split_sha256=_sha256(
            payload.get("train_split_sha256"),
            name="train_split_sha256",
        ),
        validation_split_sha256=_sha256(
            payload.get("validation_split_sha256"),
            name="validation_split_sha256",
        ),
        variant_set=IFP_VARIANT_SET,
        optimizer_protocol=normalize_optimizer_protocol(
            payload.get("optimizer_protocol")
        ),
        closed_loop_scope=_string(
            closed_loop.get("scope"),
            name="closed_loop.scope",
        ),
        minimum_trials_per_seed_variant=_integer(
            closed_loop.get("minimum_trials_per_seed_variant"),
            name="closed_loop.minimum_trials_per_seed_variant",
            minimum=1,
        ),
        primary_min_delta_k4_vs_k0=_number(
            margins.get("primary_min_delta_k4_vs_k0"),
            name="margins.primary_min_delta_k4_vs_k0",
            minimum=-1.0,
            maximum=1.0,
        ),
        secondary_max_delta_k4_vs_k0=_number(
            margins.get("secondary_max_delta_k4_vs_k0"),
            name="margins.secondary_max_delta_k4_vs_k0",
        ),
    )


def _variants_by_steps(
    variants: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for variant in variants:
        ifp_steps = variant.get("ifp_steps")
        if not isinstance(ifp_steps, int) or isinstance(ifp_steps, bool):
            raise IFPPreregistrationError("IFP endpoint variant is invalid")
        if ifp_steps in result:
            raise IFPPreregistrationError("IFP endpoint variant is duplicated")
        result[ifp_steps] = variant
    if tuple(sorted(result)) != IFP_VARIANT_SET:
        raise IFPPreregistrationError("IFP endpoint variant_set is invalid")
    return result


def _closed_loop_metric(
    variant: Mapping[str, object],
    field: str,
    *,
    maximum: float | None = None,
) -> float:
    closed_loop = _mapping(variant.get("closed_loop"), name="closed_loop result")
    return _number(
        closed_loop.get(field),
        name=f"closed_loop.{field}",
        minimum=0.0,
        maximum=maximum,
    )


def _terminal_error(variant: Mapping[str, object]) -> float | None:
    closed_loop = _mapping(variant.get("closed_loop"), name="closed_loop result")
    if "terminal_error_mean" not in closed_loop:
        raise IFPPreregistrationError("closed_loop.terminal_error_mean is required")
    value = closed_loop["terminal_error_mean"]
    if value is None:
        return None
    return _number(value, name="closed_loop.terminal_error_mean", minimum=0.0)


def evaluate_ifp_endpoints(
    plan: IFPPreregistration,
    variants: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate fixed K=4 versus K=0 endpoints without selecting a winner."""

    by_steps = _variants_by_steps(variants)
    baseline = by_steps[0]
    candidate = by_steps[PAPER_IFP_STEPS]
    primary_delta = _closed_loop_metric(
        candidate,
        "success_rate",
        maximum=1.0,
    ) - _closed_loop_metric(baseline, "success_rate", maximum=1.0)
    baseline_error = _terminal_error(baseline)
    candidate_error = _terminal_error(candidate)
    # Explicit null is an unavailable observation, never an automatic pass.
    secondary_delta = (
        candidate_error - baseline_error
        if baseline_error is not None and candidate_error is not None else None
    )
    primary_passed = primary_delta >= plan.primary_min_delta_k4_vs_k0
    secondary_passed = (
        secondary_delta is not None
        and secondary_delta <= plan.secondary_max_delta_k4_vs_k0
    )

    return {
        "passed": primary_passed and secondary_passed,
        "primary_endpoint": IFP_PRIMARY_ENDPOINT,
        "primary_delta_k4_vs_k0": primary_delta,
        "primary_required_minimum": plan.primary_min_delta_k4_vs_k0,
        "primary_passed": primary_passed,
        "secondary_endpoint": IFP_SECONDARY_ENDPOINT,
        "secondary_endpoint_available": secondary_delta is not None,
        "secondary_delta_k4_vs_k0": secondary_delta,
        "secondary_required_maximum": plan.secondary_max_delta_k4_vs_k0,
        "secondary_passed": secondary_passed,
        "statistical_significance_evaluated": False,
    }


__all__ = [
    "IFP_PRIMARY_ENDPOINT",
    "IFP_PREREGISTRATION_CLAIM_SCOPE",
    "IFP_PREREGISTRATION_SCHEMA",
    "IFP_SECONDARY_ENDPOINT",
    "IFP_VARIANT_SET",
    "IFPPreregistration",
    "IFPPreregistrationError",
    "evaluate_ifp_endpoints",
    "load_ifp_preregistration_bytes",
    "normalize_optimizer_protocol",
]
