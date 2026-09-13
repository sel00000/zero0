"""Strict identity payload for a compiled MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


MUJOCO_IDENTITY_SCHEMA_VERSION = "so101_wam.mujoco_compiled_model.v1"
_PAYLOAD_FIELDS = {
    "schema_version",
    "engine_version",
    "compiled_model_sha256",
    "compiled_model_bytes",
}
_LOWER_HEX = frozenset("0123456789abcdef")
_SHA256_HEX_LENGTH = 64


class MujocoIdentityError(ValueError):
    """Raised when a MuJoCo model identity payload is invalid."""


@dataclass(frozen=True, slots=True)
class MujocoModelIdentity:
    engine_version: str
    compiled_model_sha256: str
    compiled_model_bytes: int

    def __post_init__(self) -> None:
        _nonempty_str(self.engine_version, name="engine_version")
        _sha256(self.compiled_model_sha256, name="compiled_model_sha256")
        _positive_int(self.compiled_model_bytes, name="compiled_model_bytes")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": self.engine_version,
            "compiled_model_sha256": self.compiled_model_sha256,
            "compiled_model_bytes": self.compiled_model_bytes,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> MujocoModelIdentity:
        if not isinstance(value, Mapping) or set(value) != _PAYLOAD_FIELDS:
            raise MujocoIdentityError("MuJoCo model identity fields are invalid")
        if value.get("schema_version") != MUJOCO_IDENTITY_SCHEMA_VERSION:
            raise MujocoIdentityError("MuJoCo model identity schema is invalid")

        return cls(
            engine_version=_nonempty_str(
                value.get("engine_version"),
                name="engine_version",
            ),
            compiled_model_sha256=_sha256(
                value.get("compiled_model_sha256"),
                name="compiled_model_sha256",
            ),
            compiled_model_bytes=_positive_int(
                value.get("compiled_model_bytes"),
                name="compiled_model_bytes",
            ),
        )


def _nonempty_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MujocoIdentityError(f"MuJoCo model identity {name} is invalid")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise MujocoIdentityError(f"MuJoCo model identity {name} is invalid")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MujocoIdentityError(f"MuJoCo model identity {name} is invalid")
    return value
