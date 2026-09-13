"""Bridge checksum-bound paired artifacts into action-free task specs."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from .paired_data import HumanRobotPairRecord
from .task_specs import (
    HumanVideoFrame,
    HumanVideoPrompt,
    TaskSpecError,
    TaskSpecProvenance,
)


HUMAN_TASK_SPEC_KEYS = frozenset({"timestamp", "rgb"})


class PairedTaskSpecError(ValueError):
    """Raised when extracted human frames cannot form a safe task spec."""


def human_prompt_from_pair(pair: HumanRobotPairRecord) -> HumanVideoPrompt:
    """Load an RGB/timestamp-only prompt after rechecking its checksum."""

    if not isinstance(pair, HumanRobotPairRecord):
        raise PairedTaskSpecError("pair must be a HumanRobotPairRecord")
    artifact = pair.human_task_spec
    if artifact is None:
        raise PairedTaskSpecError("pair does not include a human task-spec artifact")

    try:
        with artifact.path.open("rb") as stream:
            digest = sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise PairedTaskSpecError(
                    "human task-spec checksum changed after import"
                )

            stream.seek(0)
            with np.load(stream, allow_pickle=False) as payload:
                keys = set(payload.files)
                if keys != HUMAN_TASK_SPEC_KEYS:
                    raise PairedTaskSpecError(
                        "human task-spec must contain exactly timestamp and rgb arrays"
                    )
                timestamps = np.array(payload["timestamp"], copy=True)
                rgb = np.array(payload["rgb"], copy=True)
    except PairedTaskSpecError:
        raise
    except (EOFError, OSError, ValueError) as error:
        raise PairedTaskSpecError(
            f"failed to load human task-spec artifact: {error}"
        ) from error

    _validate_arrays(timestamps, rgb)
    try:
        return HumanVideoPrompt(
            frames=tuple(
                HumanVideoFrame(timestamp_s=float(timestamp), rgb=frame)
                for timestamp, frame in zip(timestamps, rgb, strict=True)
            ),
            provenance=TaskSpecProvenance(
                source_id=f"pair:{pair.fingerprint}:human-task-spec",
                source_sha256=artifact.sha256,
            ),
            text_metadata=pair.task,
        )
    except TaskSpecError as error:
        raise PairedTaskSpecError(f"invalid human task spec: {error}") from error


def _validate_arrays(timestamps: np.ndarray, rgb: np.ndarray) -> None:
    is_real = np.issubdtype(timestamps.dtype, np.integer) or np.issubdtype(
        timestamps.dtype,
        np.floating,
    )
    if timestamps.ndim != 1 or not is_real:
        raise PairedTaskSpecError("timestamp must be one numeric array")
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise PairedTaskSpecError("rgb must have uint8 shape [T, H, W, 3]")
    if len(timestamps) != len(rgb):
        raise PairedTaskSpecError("timestamp and rgb lengths must match")
    if not np.isfinite(timestamps).all():
        raise PairedTaskSpecError("timestamp values must be finite")
__all__ = [
    "HUMAN_TASK_SPEC_KEYS",
    "PairedTaskSpecError",
    "human_prompt_from_pair",
]
