"""Shared result contract for IFP runners and report-only consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import PurePosixPath
import re
from typing import Any


IFP_ABLATION_SCHEMA = "so101_wam.ifp_ablation.v4"
IFP_PROTOCOL_LIMITATION = (
    "compiled-model identity does not attest GPU drivers or cross-build determinism"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class IFPAblationError(RuntimeError):
    """Raised when a comparable immutable ablation cannot be published."""


def _number(value: object, *, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value < 0
    ):
        raise IFPAblationError(f"closed-loop {name} must be finite and non-negative")


def _sha256(value: object, *, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IFPAblationError(f"closed-loop {name} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    """Keep execution outcomes even when no terminal metric was observed."""

    scope: str
    trial_count: int
    scored_trial_count: int
    execution_failure_count: int
    success_count: int
    success_rate: float
    terminal_error_mean: float | None
    terminal_error_basis: str
    failure_counts: tuple[tuple[str, int], ...]
    protocol_sha256: str
    artifact: str | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("scope", "terminal_error_basis"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise IFPAblationError(f"closed-loop {name} must be non-empty")
        _sha256(self.protocol_sha256, name="protocol_sha256")
        for name in (
            "trial_count", "scored_trial_count", "execution_failure_count", "success_count"
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise IFPAblationError(f"closed-loop {name} is invalid")
        if (
            self.trial_count < 1
            or self.scored_trial_count + self.execution_failure_count != self.trial_count
            or self.success_count > self.scored_trial_count
        ):
            raise IFPAblationError("closed-loop counts are inconsistent")
        _number(self.success_rate, name="success_rate")
        if self.success_rate != self.success_count / self.trial_count:
            raise IFPAblationError("closed-loop success_rate is inconsistent")

        # Missing terminal observations are never turned into zero error.
        if self.scored_trial_count:
            _number(self.terminal_error_mean, name="terminal_error_mean")
        elif self.terminal_error_mean is not None:
            raise IFPAblationError("unscored closed-loop terminal_error_mean must be null")
        if not isinstance(self.failure_counts, tuple):
            raise IFPAblationError("closed-loop failure_counts must be immutable")
        total = 0
        prior_reason = ""
        for item in self.failure_counts:
            if not isinstance(item, tuple) or len(item) != 2:
                raise IFPAblationError("closed-loop failure_counts are invalid")
            reason, count = item
            if (
                not isinstance(reason, str)
                or not reason.strip()
                or reason <= prior_reason
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
            ):
                raise IFPAblationError("closed-loop failure_counts are invalid")
            prior_reason = reason
            total += count
        if total != self.trial_count - self.success_count:
            raise IFPAblationError("closed-loop failure_counts are inconsistent")
        if (self.artifact is None) != (self.artifact_sha256 is None):
            raise IFPAblationError("closed-loop artifact and hash must appear together")
        if self.artifact is None:
            return
        if (
            not isinstance(self.artifact, str)
            or not self.artifact
            or "\\" in self.artifact
            or PurePosixPath(self.artifact).name != self.artifact
            or self.artifact in {".", ".."}
        ):
            raise IFPAblationError("closed-loop artifact metadata is invalid")
        _sha256(self.artifact_sha256, name="artifact metadata")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_counts"] = dict(self.failure_counts)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ClosedLoopResult:
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise IFPAblationError(
                "closed-loop fields differ: "
                f"missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        failures = payload["failure_counts"]
        if not isinstance(failures, Mapping) or any(
            not isinstance(reason, str) for reason in failures
        ):
            raise IFPAblationError("closed-loop failure_counts must be an object")
        normalized = dict(payload)
        normalized["failure_counts"] = tuple(sorted(failures.items()))
        return cls(**normalized)
