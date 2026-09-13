"""Clocked rollout lifecycle with optional bounded execution evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import InitVar, dataclass, field
from hashlib import sha256
import json
from math import ceil, isclose, isfinite
from time import monotonic, sleep
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .constants import ACTION_DIM, JOINT_KEYS, PRIMARY_CAMERA_COUNT
from .runtime import PolicyStep, RuntimeState, SO101WAMRuntime, ServoStep


EXECUTABILITY_TRACE_SCHEMA = 3
DEFAULT_ACTION_TRACE_LIMIT = 64
MAX_ACTION_TRACE_LIMIT = 256
ACTION_TRACE_HASH = "sha256_float32_shape_v1"
DEFAULT_CONTACT_TRACE_LIMIT = 256
MAX_CONTACT_TRACE_LIMIT = 1024
CONTACT_PROGRESSION_CONTRACT = "mujoco_contact_progression_v1"
CONTACT_INDEX_HASH = "sha256_int64_pair_stream_v1"
JOINT_LIMIT_MARGIN_CONTRACT = "nearest_bound_margin_over_span_v1"
DEFAULT_FUTURE_LATENT_TRACE_LIMIT = 256
MAX_FUTURE_LATENT_TRACE_LIMIT = 1024
FUTURE_LATENT_ALIGNMENT_CONTRACT = (
    "next_policy_observation_compact_live_encoder_mse_v1"
)
FUTURE_LATENT_STREAM_HASH = "sha256_policy_index_timestamp_float32_shape_v1"
FUTURE_LATENT_INDEX_HASH = "sha256_int64_source_target_offset_stream_v1"
JOINT_LIMIT_MARGIN_FIELDS = (
    "decoded_policy_target_min",
    "servo_safe_target_min",
    "measured_joint_min",
    "executed_action_min",
)


class RolloutError(RuntimeError):
    """Raised when a managed rollout cannot continue safely."""


class SafetyRejectedError(RolloutError):
    """Raised with the safety reasons that stopped a servo row."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__(f"safety rejected servo row: {reasons}")


class ManagedRobotAdapter(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect(self, *, calibrate: bool = True) -> None: ...

    def disconnect(self) -> None: ...


class RolloutObserver(Protocol):
    def on_policy_step(self, *, policy_index: int, step: PolicyStep) -> None: ...

    def on_servo_step(
        self,
        *,
        policy_index: int,
        servo_index: int,
        step: ServoStep,
    ) -> None: ...


class _Digest(Protocol):
    def update(self, source: bytes) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    prompt_fingerprint: str
    policy_steps: int
    servo_steps: int
    sent_actions: int
    shadow_steps: int
    elapsed_s: float
    final_state: str


def _array_source(value: object) -> bytes:
    array = np.ascontiguousarray(value, dtype="<f4")
    shape = np.asarray(array.shape, dtype="<i8")
    return shape.tobytes() + array.tobytes()


def _index_source(policy_index: int, servo_index: int) -> bytes:
    for name, value in (
        ("policy_index", policy_index),
        ("servo_index", servo_index),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RolloutError(f"executability {name} is invalid")
    return np.asarray((policy_index, servo_index), dtype="<i8").tobytes()


def _future_latent_source(
    policy_index: int,
    observation_timestamp_s: float,
    value: object,
) -> bytes:
    if (
        not isinstance(policy_index, int)
        or isinstance(policy_index, bool)
        or policy_index < 0
    ):
        raise RolloutError("executability future latent policy_index is invalid")
    if (
        isinstance(observation_timestamp_s, bool)
        or not isinstance(observation_timestamp_s, (int, float))
        or not isfinite(float(observation_timestamp_s))
        or float(observation_timestamp_s) < 0.0
    ):
        raise RolloutError("executability future latent timestamp is invalid")
    prefix = np.asarray((policy_index,), dtype="<i8").tobytes()
    timestamp = np.asarray((observation_timestamp_s,), dtype="<f8").tobytes()
    return prefix + timestamp + _array_source(value)


def _future_latent_index_source(
    source_policy_index: int,
    target_policy_index: int,
    offset: int,
) -> bytes:
    if (
        source_policy_index < 0
        or target_policy_index <= source_policy_index
        or offset != target_policy_index - source_policy_index
    ):
        raise RolloutError("executability future latent alignment index is invalid")
    return np.asarray(
        (source_policy_index, target_policy_index, offset),
        dtype="<i8",
    ).tobytes()


def _latent_array(value: object, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(array).all():
        raise RolloutError(f"executability {name} must be finite")
    array.flags.writeable = False
    return array


def _json_source(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _contact_record(
    value: object,
    *,
    include_forbidden: bool,
) -> dict[str, Any]:
    fields = {
        "geom1",
        "geom2",
        "body1",
        "body2",
        "category1",
        "category2",
        "distance",
    }
    if include_forbidden:
        fields.add("forbidden")
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RolloutError("executability contact fields are invalid")
    for name in fields - {"distance", "forbidden"}:
        item = value.get(name)
        if not isinstance(item, str) or not item:
            raise RolloutError("executability contact identity is invalid")
    distance = value.get("distance")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not isfinite(float(distance))
    ):
        raise RolloutError("executability contact distance is invalid")
    if include_forbidden and not isinstance(value.get("forbidden"), bool):
        raise RolloutError("executability forbidden contact marker is invalid")

    result = {name: value[name] for name in sorted(fields)}
    result["distance"] = float(distance)
    return result


def _contact_sort_key(value: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        value["category1"],
        value["category2"],
        value["body1"],
        value["body2"],
        value["geom1"],
        value["geom2"],
        value["distance"],
        value.get("forbidden", False),
    )


def _joint_range(
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    lower_array = np.array(lower, dtype=np.float64, copy=True)
    upper_array = np.array(upper, dtype=np.float64, copy=True)
    if lower_array.shape != (ACTION_DIM,) or upper_array.shape != (ACTION_DIM,):
        raise RolloutError(
            f"executability {name} bounds must have shape ({ACTION_DIM},)"
        )
    if not np.isfinite(lower_array).all() or not np.isfinite(upper_array).all():
        raise RolloutError(f"executability {name} bounds must be finite")
    if not np.all(lower_array < upper_array):
        raise RolloutError(
            f"executability {name} lower bounds must be below upper bounds"
        )
    lower_array.flags.writeable = False
    upper_array.flags.writeable = False
    return lower_array, upper_array


@dataclass(slots=True)
class ExecutabilityTrace:
    """Bounded action evidence with complete streaming hashes and safety counts."""

    rollout_id: str
    joint_lower: InitVar[Sequence[float] | None] = None
    joint_upper: InitVar[Sequence[float] | None] = None
    action_limit: int = DEFAULT_ACTION_TRACE_LIMIT
    contact_limit: int = DEFAULT_CONTACT_TRACE_LIMIT
    future_latent_limit: int = DEFAULT_FUTURE_LATENT_TRACE_LIMIT
    observation_joint_lower: InitVar[Sequence[float] | None] = None
    observation_joint_upper: InitVar[Sequence[float] | None] = None
    _decoded_actions: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _policy_digest: _Digest = field(default_factory=sha256, init=False, repr=False)
    _servo_digest: _Digest = field(default_factory=sha256, init=False, repr=False)
    _executed_digest: _Digest = field(default_factory=sha256, init=False, repr=False)
    _contact_digest: _Digest = field(default_factory=sha256, init=False, repr=False)
    _servo_index_digest: _Digest = field(default_factory=sha256, init=False, repr=False)
    _contact_index_digest: _Digest = field(
        default_factory=sha256,
        init=False,
        repr=False,
    )
    _safety_reasons: Counter[str] = field(
        default_factory=Counter,
        init=False,
        repr=False,
    )
    _policy_steps: int = field(default=0, init=False, repr=False)
    _servo_steps: int = field(default=0, init=False, repr=False)
    _sent_actions: int = field(default=0, init=False, repr=False)
    _shadow_steps: int = field(default=0, init=False, repr=False)
    _safety_accepted: int = field(default=0, init=False, repr=False)
    _safety_rejected: int = field(default=0, init=False, repr=False)
    _safety_clipped: int = field(default=0, init=False, repr=False)
    _joint_lower: np.ndarray | None = field(default=None, init=False, repr=False)
    _joint_upper: np.ndarray | None = field(default=None, init=False, repr=False)
    _observation_joint_lower: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _observation_joint_upper: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _policy_margin: float | None = field(default=None, init=False, repr=False)
    _servo_margin: float | None = field(default=None, init=False, repr=False)
    _measured_margin: float | None = field(default=None, init=False, repr=False)
    _executed_margin: float | None = field(default=None, init=False, repr=False)
    _contact_progression: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _contact_samples: int = field(default=0, init=False, repr=False)
    _contact_observations: int = field(default=0, init=False, repr=False)
    _forbidden_contact_observations: int = field(default=0, init=False, repr=False)
    _object_contact_observations: int = field(default=0, init=False, repr=False)
    _minimum_contact_distance_m: float | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _collision_failure_phase: str | None = field(default=None, init=False, repr=False)
    _collision_failure_contacts: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _future_latent_predictions: dict[int, tuple[float, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _future_latent_progression: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _future_latent_prediction_digest: _Digest = field(
        default_factory=sha256,
        init=False,
        repr=False,
    )
    _future_latent_observation_digest: _Digest = field(
        default_factory=sha256,
        init=False,
        repr=False,
    )
    _future_latent_index_digest: _Digest = field(
        default_factory=sha256,
        init=False,
        repr=False,
    )
    _future_latent_prediction_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _future_latent_observation_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _future_latent_aligned_pair_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _future_latent_element_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _future_latent_squared_error_sum: float = field(
        default=0.0,
        init=False,
        repr=False,
    )
    _future_latent_future_steps: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _future_latent_dim: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _future_latent_offset_pairs: Counter[int] = field(
        default_factory=Counter,
        init=False,
        repr=False,
    )
    _future_latent_offset_elements: Counter[int] = field(
        default_factory=Counter,
        init=False,
        repr=False,
    )
    _future_latent_offset_squared_error: dict[int, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(
        self,
        joint_lower: Sequence[float] | None,
        joint_upper: Sequence[float] | None,
        observation_joint_lower: Sequence[float] | None,
        observation_joint_upper: Sequence[float] | None,
    ) -> None:
        if (
            not isinstance(self.rollout_id, str)
            or not self.rollout_id
            or self.rollout_id != self.rollout_id.strip()
        ):
            raise RolloutError("executability rollout_id must be a non-empty string")
        if (
            not isinstance(self.action_limit, int)
            or isinstance(self.action_limit, bool)
            or not 1 <= self.action_limit <= MAX_ACTION_TRACE_LIMIT
        ):
            raise RolloutError(
                "executability action_limit must be between 1 and "
                f"{MAX_ACTION_TRACE_LIMIT}"
            )
        if (
            not isinstance(self.contact_limit, int)
            or isinstance(self.contact_limit, bool)
            or not 1 <= self.contact_limit <= MAX_CONTACT_TRACE_LIMIT
        ):
            raise RolloutError(
                "executability contact_limit must be between 1 and "
                f"{MAX_CONTACT_TRACE_LIMIT}"
            )
        if (
            not isinstance(self.future_latent_limit, int)
            or isinstance(self.future_latent_limit, bool)
            or not 1 <= self.future_latent_limit <= MAX_FUTURE_LATENT_TRACE_LIMIT
        ):
            raise RolloutError(
                "executability future_latent_limit must be between 1 and "
                f"{MAX_FUTURE_LATENT_TRACE_LIMIT}"
            )
        if (joint_lower is None) is not (joint_upper is None):
            raise RolloutError(
                "executability joint lower and upper bounds must be provided together"
            )
        if (observation_joint_lower is None) is not (observation_joint_upper is None):
            raise RolloutError(
                "executability observation joint bounds must be provided together"
            )
        if joint_lower is None:
            if observation_joint_lower is not None:
                raise RolloutError(
                    "executability observation joint bounds require joint bounds"
                )
            return

        assert joint_upper is not None
        lower, upper = _joint_range(joint_lower, joint_upper, name="joint")
        obs_lower = lower
        obs_upper = upper
        if observation_joint_lower is not None and observation_joint_upper is not None:
            # Match effective safety bounds without changing legacy command margins.
            obs_lower, obs_upper = _joint_range(
                np.asarray(observation_joint_lower, dtype=np.float32),
                np.asarray(observation_joint_upper, dtype=np.float32),
                name="observation joint",
            )
        self._joint_lower = lower
        self._joint_upper = upper
        self._observation_joint_lower = obs_lower
        self._observation_joint_upper = obs_upper

    def _margin(
        self,
        value: object,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        name: str,
    ) -> float:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim not in {1, 2} or array.shape[-1] != ACTION_DIM:
            raise RolloutError(
                f"executability {name} must end with shape ({ACTION_DIM},)"
            )
        if not np.isfinite(array).all():
            raise RolloutError(f"executability {name} must be finite")
        span = upper - lower
        margin = np.minimum(array - lower, upper - array)
        return float(np.min(margin / span))

    @staticmethod
    def _minimum(current: float | None, candidate: float) -> float:
        return candidate if current is None else min(current, candidate)

    def on_policy_step(self, *, policy_index: int, step: PolicyStep) -> None:
        source = _array_source(step.action.target_joint_position)
        self._policy_digest.update(source)
        self._policy_steps += 1
        if self._joint_lower is not None:
            assert self._joint_upper is not None
            margin = self._margin(
                step.action.target_joint_position,
                lower=self._joint_lower,
                upper=self._joint_upper,
                name="decoded policy target",
            )
            self._policy_margin = self._minimum(self._policy_margin, margin)
        if len(self._decoded_actions) >= self.action_limit:
            return

        self._decoded_actions.append(
            {
                "policy_index": policy_index,
                "created_at_s": step.action.created_at_s,
                "action_shape": list(step.action.target_joint_position.shape),
                "action_sha256": sha256(source).hexdigest(),
            }
        )

    def on_future_latent_step(
        self,
        *,
        policy_index: int,
        observation_timestamp_s: float,
        future_latents: object,
        observed_latent: object,
    ) -> None:
        """Align one prediction with later policy-rate live observations."""

        if policy_index != self._future_latent_prediction_count:
            raise RolloutError(
                "executability future latent policy index sequence is invalid"
            )
        if policy_index >= self._policy_steps:
            raise RolloutError(
                "executability future latent prediction requires its policy step"
            )
        timestamp = float(observation_timestamp_s)
        prediction = _latent_array(
            future_latents,
            name="future latent prediction",
        )
        observation = _latent_array(
            observed_latent,
            name="future latent observation",
        )
        if (
            prediction.ndim != 3
            or prediction.shape[0] < 1
            or prediction.shape[1] != PRIMARY_CAMERA_COUNT
            or prediction.shape[2] < 1
        ):
            raise RolloutError(
                "executability future latent prediction shape is invalid"
            )
        if observation.shape != prediction.shape[1:]:
            raise RolloutError(
                "executability future latent observation shape is invalid"
            )

        future_steps = int(prediction.shape[0])
        latent_dim = int(prediction.shape[2])
        if (
            self._future_latent_future_steps is not None
            or self._future_latent_dim is not None
        ):
            if (
                self._future_latent_future_steps != future_steps
                or self._future_latent_dim != latent_dim
            ):
                raise RolloutError(
                    "executability future latent dimensions changed during rollout"
                )

        prediction_source = _future_latent_source(
            policy_index,
            timestamp,
            prediction,
        )
        observation_source = _future_latent_source(
            policy_index,
            timestamp,
            observation,
        )

        completed_sources: list[int] = []
        aligned_records: list[dict[str, Any]] = []
        for source_index in sorted(self._future_latent_predictions):
            source_timestamp, source_prediction = self._future_latent_predictions[
                source_index
            ]
            offset = policy_index - source_index
            if offset < 1 or offset > future_steps:
                raise RolloutError(
                    "executability future latent pending alignment is invalid"
                )
            if timestamp <= source_timestamp:
                raise RolloutError(
                    "executability future latent timestamp sequence is invalid"
                )
            predicted = source_prediction[offset - 1]
            delta = predicted.astype(np.float64) - observation.astype(np.float64)
            squared_error = float(np.square(delta).sum())
            element_count = int(delta.size)
            mse = squared_error / element_count
            index_source = _future_latent_index_source(
                source_index,
                policy_index,
                offset,
            )
            progression = None
            if (
                len(self._future_latent_progression) + len(aligned_records)
                < self.future_latent_limit
            ):
                progression = {
                    "source_policy_index": source_index,
                    "target_policy_index": policy_index,
                    "offset": offset,
                    "source_observation_timestamp_s": source_timestamp,
                    "target_observation_timestamp_s": timestamp,
                    "elapsed_s": timestamp - source_timestamp,
                    "element_count": element_count,
                    "squared_error_sum": squared_error,
                    "mse": mse,
                    "predicted_sha256": sha256(_array_source(predicted)).hexdigest(),
                    "observed_sha256": sha256(_array_source(observation)).hexdigest(),
                }
            aligned_records.append(
                {
                    "index_source": index_source,
                    "offset": offset,
                    "element_count": element_count,
                    "squared_error_sum": squared_error,
                    "progression": progression,
                }
            )
            if offset == future_steps:
                completed_sources.append(source_index)

        if self._future_latent_future_steps is None:
            self._future_latent_future_steps = future_steps
            self._future_latent_dim = latent_dim
        self._future_latent_prediction_digest.update(
            len(prediction_source).to_bytes(8, "big") + prediction_source
        )
        self._future_latent_observation_digest.update(
            len(observation_source).to_bytes(8, "big") + observation_source
        )
        for record in aligned_records:
            offset = record["offset"]
            element_count = record["element_count"]
            squared_error = record["squared_error_sum"]
            self._future_latent_index_digest.update(record["index_source"])
            self._future_latent_aligned_pair_count += 1
            self._future_latent_element_count += element_count
            self._future_latent_squared_error_sum += squared_error
            self._future_latent_offset_pairs[offset] += 1
            self._future_latent_offset_elements[offset] += element_count
            self._future_latent_offset_squared_error[offset] = (
                self._future_latent_offset_squared_error.get(offset, 0.0)
                + squared_error
            )
            if record["progression"] is not None:
                self._future_latent_progression.append(record["progression"])

        for source_index in completed_sources:
            del self._future_latent_predictions[source_index]
        self._future_latent_predictions[policy_index] = (timestamp, prediction)
        self._future_latent_prediction_count += 1
        self._future_latent_observation_count += 1

    def on_servo_step(
        self,
        *,
        policy_index: int,
        servo_index: int,
        step: ServoStep,
    ) -> None:
        self._servo_index_digest.update(_index_source(policy_index, servo_index))
        self._servo_digest.update(_array_source(step.action.target_joint_position))
        self._executed_digest.update(_array_source(step.executed_action))
        self._servo_steps += 1
        self._sent_actions += int(step.sent)
        self._shadow_steps += int(step.shadow)
        self._safety_accepted += int(step.safety.accepted)
        self._safety_rejected += int(not step.safety.accepted)
        self._safety_clipped += int(step.safety.clipped)
        self._safety_reasons.update(step.safety.reasons)

        if self._joint_lower is None:
            return
        assert self._joint_upper is not None
        assert self._observation_joint_lower is not None
        assert self._observation_joint_upper is not None
        servo_margin = self._margin(
            step.action.target_joint_position,
            lower=self._joint_lower,
            upper=self._joint_upper,
            name="servo safe target",
        )
        measured_margin = self._margin(
            step.observation.joint_position,
            lower=self._observation_joint_lower,
            upper=self._observation_joint_upper,
            name="measured joint position",
        )
        executed_margin = self._margin(
            step.executed_action,
            lower=self._joint_lower,
            upper=self._joint_upper,
            name="executed action",
        )
        self._servo_margin = self._minimum(self._servo_margin, servo_margin)
        self._measured_margin = self._minimum(self._measured_margin, measured_margin)
        self._executed_margin = self._minimum(self._executed_margin, executed_margin)

    def on_contact_sample(
        self,
        *,
        policy_index: int,
        servo_index: int,
        phase: str,
        contacts: Sequence[Mapping[str, object]],
    ) -> None:
        index_source = _index_source(policy_index, servo_index)
        if not isinstance(phase, str) or not phase:
            raise RolloutError("executability contact phase is invalid")
        if isinstance(contacts, (str, bytes)) or not isinstance(contacts, Sequence):
            raise RolloutError("executability contacts must be a sequence")

        normalized = sorted(
            (
                _contact_record(contact, include_forbidden=True)
                for contact in contacts
            ),
            key=_contact_sort_key,
        )
        forbidden = sum(int(contact["forbidden"]) for contact in normalized)
        objects = sum(
            int("object" in {contact["category1"], contact["category2"]})
            for contact in normalized
        )
        distances = [float(contact["distance"]) for contact in normalized]
        minimum_distance = min(distances) if distances else None
        category_pairs = sorted(
            {
                tuple(sorted((contact["category1"], contact["category2"])))
                for contact in normalized
            }
        )
        sample = {
            "policy_index": policy_index,
            "servo_index": servo_index,
            "phase": phase,
            "contact_count": len(normalized),
            "forbidden_contact_count": forbidden,
            "object_contact_count": objects,
            "minimum_distance_m": minimum_distance,
            "category_pairs": [list(pair) for pair in category_pairs],
            "contacts_sha256": sha256(_json_source(normalized)).hexdigest(),
        }
        source = _json_source({**sample, "contacts": normalized})
        self._contact_digest.update(len(source).to_bytes(8, "big") + source)
        self._contact_index_digest.update(index_source)
        self._contact_samples += 1
        self._contact_observations += len(normalized)
        self._forbidden_contact_observations += forbidden
        self._object_contact_observations += objects
        if minimum_distance is not None:
            self._minimum_contact_distance_m = self._minimum(
                self._minimum_contact_distance_m,
                minimum_distance,
            )
        if len(self._contact_progression) < self.contact_limit:
            self._contact_progression.append(sample)

    def on_collision_failure(
        self,
        *,
        phase: str,
        contacts: Sequence[Mapping[str, object]],
    ) -> None:
        if self._collision_failure_phase is not None:
            raise RolloutError("executability collision failure is already recorded")
        if not isinstance(phase, str) or not phase:
            raise RolloutError("executability collision failure phase is invalid")
        normalized = [
            _contact_record(contact, include_forbidden=False)
            for contact in contacts
        ]
        if not normalized:
            raise RolloutError("executability collision failure requires contacts")
        self._collision_failure_phase = phase
        self._collision_failure_contacts = normalized
        self.on_contact_sample(
            policy_index=max(0, self._policy_steps - 1),
            servo_index=self._servo_steps,
            phase=f"collision:{phase}",
            contacts=tuple({**contact, "forbidden": True} for contact in normalized),
        )

    def snapshot(self) -> dict[str, Any]:
        future_steps = self._future_latent_future_steps
        future_latent_mse = (
            None
            if not self._future_latent_element_count
            else self._future_latent_squared_error_sum
            / self._future_latent_element_count
        )
        future_latent_by_offset = {
            str(offset): {
                "aligned_pair_count": self._future_latent_offset_pairs[offset],
                "element_count": self._future_latent_offset_elements[offset],
                "squared_error_sum": self._future_latent_offset_squared_error.get(
                    offset,
                    0.0,
                ),
                "mse_mean": (
                    None
                    if not self._future_latent_offset_elements[offset]
                    else self._future_latent_offset_squared_error.get(offset, 0.0)
                    / self._future_latent_offset_elements[offset]
                ),
            }
            for offset in range(1, (future_steps or 0) + 1)
        }
        return {
            "schema_version": EXECUTABILITY_TRACE_SCHEMA,
            "rollout_id": self.rollout_id,
            "action_hash": ACTION_TRACE_HASH,
            "action_limit": self.action_limit,
            "decoded_action_count": self._policy_steps,
            "decoded_actions_truncated": self._policy_steps > self.action_limit,
            "decoded_actions": [dict(item) for item in self._decoded_actions],
            "policy_steps": self._policy_steps,
            "servo_steps": self._servo_steps,
            "sent_actions": self._sent_actions,
            "shadow_steps": self._shadow_steps,
            "safety_accepted_steps": self._safety_accepted,
            "safety_rejected_steps": self._safety_rejected,
            "safety_clipped_steps": self._safety_clipped,
            "safety_reason_counts": dict(sorted(self._safety_reasons.items())),
            "policy_action_stream_sha256": self._policy_digest.hexdigest(),
            "servo_target_stream_sha256": self._servo_digest.hexdigest(),
            "executed_action_stream_sha256": self._executed_digest.hexdigest(),
            "joint_limit_margin_contract": JOINT_LIMIT_MARGIN_CONTRACT,
            "joint_limit_margins": {
                "decoded_policy_target_min": self._policy_margin,
                "servo_safe_target_min": self._servo_margin,
                "measured_joint_min": self._measured_margin,
                "executed_action_min": self._executed_margin,
            },
            "contact_progression_contract": CONTACT_PROGRESSION_CONTRACT,
            "contact_progression_limit": self.contact_limit,
            "contact_progression_count": self._contact_samples,
            "contact_progression_truncated": (
                self._contact_samples > self.contact_limit
            ),
            "contact_progression": [
                {
                    **item,
                    "category_pairs": [
                        list(pair) for pair in item["category_pairs"]
                    ],
                }
                for item in self._contact_progression
            ],
            "contact_index_hash": CONTACT_INDEX_HASH,
            "servo_index_stream_sha256": self._servo_index_digest.hexdigest(),
            "contact_index_stream_sha256": self._contact_index_digest.hexdigest(),
            "contact_stream_sha256": self._contact_digest.hexdigest(),
            "contact_observation_count": self._contact_observations,
            "forbidden_contact_observation_count": (
                self._forbidden_contact_observations
            ),
            "object_contact_observation_count": self._object_contact_observations,
            "minimum_contact_distance_m": self._minimum_contact_distance_m,
            "collision_failure_phase": self._collision_failure_phase,
            "collision_failure_contacts": [
                dict(contact) for contact in self._collision_failure_contacts
            ],
            "future_latent_alignment_contract": FUTURE_LATENT_ALIGNMENT_CONTRACT,
            "future_latent_stream_hash": FUTURE_LATENT_STREAM_HASH,
            "future_latent_index_hash": FUTURE_LATENT_INDEX_HASH,
            "future_latent_progression_limit": self.future_latent_limit,
            "future_latent_prediction_count": (
                self._future_latent_prediction_count
            ),
            "future_latent_observation_count": (
                self._future_latent_observation_count
            ),
            "future_latent_aligned_pair_count": (
                self._future_latent_aligned_pair_count
            ),
            "future_latent_censored_pair_count": (
                0
                if future_steps is None
                else self._future_latent_prediction_count * future_steps
                - self._future_latent_aligned_pair_count
            ),
            "future_latent_future_steps": future_steps,
            "future_latent_latent_dim": self._future_latent_dim,
            "future_latent_element_count": self._future_latent_element_count,
            "future_latent_squared_error_sum": (
                self._future_latent_squared_error_sum
            ),
            "future_latent_mse_mean": future_latent_mse,
            "future_latent_mse_by_offset": future_latent_by_offset,
            "future_latent_progression_count": (
                self._future_latent_aligned_pair_count
            ),
            "future_latent_progression_truncated": (
                self._future_latent_aligned_pair_count > self.future_latent_limit
            ),
            "future_latent_progression": [
                dict(item) for item in self._future_latent_progression
            ],
            "future_latent_prediction_stream_sha256": (
                self._future_latent_prediction_digest.hexdigest()
            ),
            "future_latent_observation_stream_sha256": (
                self._future_latent_observation_digest.hexdigest()
            ),
            "future_latent_index_stream_sha256": (
                self._future_latent_index_digest.hexdigest()
            ),
            "runtime_future_latent_proxy_available": (
                self._future_latent_aligned_pair_count > 0
            ),
            "runtime_future_latent_success_claimed": False,
        }


def _count(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RolloutError(f"executability {name} must be a non-negative integer")
    return value


def _sha256_text(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RolloutError(f"executability {name} must be a SHA-256 digest")
    return value


def _nonnegative_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RolloutError(f"executability {name} must be finite and non-negative")
    return float(value)


def _optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    result = _count(value, name=name)
    if result < 1:
        raise RolloutError(f"executability {name} must be positive")
    return result


def _same_float(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def validate_executability_trace(
    value: object,
    *,
    rollout_id: str,
) -> dict[str, Any]:
    """Validate trace shape before publishing it as semantic evidence."""

    fields = {
        "schema_version",
        "rollout_id",
        "action_hash",
        "action_limit",
        "decoded_action_count",
        "decoded_actions_truncated",
        "decoded_actions",
        "policy_steps",
        "servo_steps",
        "sent_actions",
        "shadow_steps",
        "safety_accepted_steps",
        "safety_rejected_steps",
        "safety_clipped_steps",
        "safety_reason_counts",
        "policy_action_stream_sha256",
        "servo_target_stream_sha256",
        "executed_action_stream_sha256",
        "joint_limit_margin_contract",
        "joint_limit_margins",
        "contact_progression_contract",
        "contact_progression_limit",
        "contact_progression_count",
        "contact_progression_truncated",
        "contact_progression",
        "contact_index_hash",
        "servo_index_stream_sha256",
        "contact_index_stream_sha256",
        "contact_stream_sha256",
        "contact_observation_count",
        "forbidden_contact_observation_count",
        "object_contact_observation_count",
        "minimum_contact_distance_m",
        "collision_failure_phase",
        "collision_failure_contacts",
        "future_latent_alignment_contract",
        "future_latent_stream_hash",
        "future_latent_index_hash",
        "future_latent_progression_limit",
        "future_latent_prediction_count",
        "future_latent_observation_count",
        "future_latent_aligned_pair_count",
        "future_latent_censored_pair_count",
        "future_latent_future_steps",
        "future_latent_latent_dim",
        "future_latent_element_count",
        "future_latent_squared_error_sum",
        "future_latent_mse_mean",
        "future_latent_mse_by_offset",
        "future_latent_progression_count",
        "future_latent_progression_truncated",
        "future_latent_progression",
        "future_latent_prediction_stream_sha256",
        "future_latent_observation_stream_sha256",
        "future_latent_index_stream_sha256",
        "runtime_future_latent_proxy_available",
        "runtime_future_latent_success_claimed",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RolloutError("executability trace fields are invalid")
    if value.get("schema_version") != EXECUTABILITY_TRACE_SCHEMA:
        raise RolloutError("executability trace schema is invalid")
    if value.get("rollout_id") != rollout_id:
        raise RolloutError("executability trace rollout_id mismatch")
    if value.get("action_hash") != ACTION_TRACE_HASH:
        raise RolloutError("executability action hash contract is invalid")

    action_limit = _count(value.get("action_limit"), name="action_limit")
    if not 1 <= action_limit <= MAX_ACTION_TRACE_LIMIT:
        raise RolloutError("executability action_limit is outside its bound")
    policy_steps = _count(value.get("policy_steps"), name="policy_steps")
    servo_steps = _count(value.get("servo_steps"), name="servo_steps")
    decoded_count = _count(
        value.get("decoded_action_count"),
        name="decoded_action_count",
    )
    sent_actions = _count(value.get("sent_actions"), name="sent_actions")
    shadow_steps = _count(value.get("shadow_steps"), name="shadow_steps")
    safety_accepted = _count(
        value.get("safety_accepted_steps"),
        name="safety_accepted_steps",
    )
    safety_rejected = _count(
        value.get("safety_rejected_steps"),
        name="safety_rejected_steps",
    )
    safety_clipped = _count(
        value.get("safety_clipped_steps"),
        name="safety_clipped_steps",
    )
    if decoded_count != policy_steps:
        raise RolloutError("executability decoded action count mismatch")
    if safety_accepted + safety_rejected != servo_steps:
        raise RolloutError("executability safety step counts mismatch")
    if sent_actions > safety_accepted or shadow_steps > servo_steps:
        raise RolloutError("executability execution counts are invalid")
    if safety_clipped > safety_accepted:
        raise RolloutError("executability clipped count is invalid")

    truncated = value.get("decoded_actions_truncated")
    if not isinstance(truncated, bool) or truncated is not (
        policy_steps > action_limit
    ):
        raise RolloutError("executability decoded action truncation marker is invalid")
    actions = value.get("decoded_actions")
    if not isinstance(actions, list) or len(actions) != min(policy_steps, action_limit):
        raise RolloutError("executability decoded action inventory is invalid")
    for index, action in enumerate(actions):
        action_fields = {
            "policy_index",
            "created_at_s",
            "action_shape",
            "action_sha256",
        }
        if not isinstance(action, Mapping) or set(action) != action_fields:
            raise RolloutError("executability decoded action fields are invalid")
        policy_index = action.get("policy_index")
        if (
            not isinstance(policy_index, int)
            or isinstance(policy_index, bool)
            or policy_index != index
        ):
            raise RolloutError("executability policy index sequence is invalid")
        created_at_s = action.get("created_at_s")
        if (
            isinstance(created_at_s, bool)
            or not isinstance(created_at_s, (int, float))
            or not isfinite(float(created_at_s))
            or float(created_at_s) < 0.0
        ):
            raise RolloutError("executability action timestamp is invalid")
        shape = action.get("action_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not isinstance(shape[0], int)
            or isinstance(shape[0], bool)
            or shape[0] < 1
            or shape[1] != ACTION_DIM
        ):
            raise RolloutError("executability action shape is invalid")
        _sha256_text(action.get("action_sha256"), name="action_sha256")

    reasons = value.get("safety_reason_counts")
    if not isinstance(reasons, Mapping):
        raise RolloutError("executability safety reason counts are invalid")
    for reason, count in reasons.items():
        if not isinstance(reason, str) or not reason:
            raise RolloutError("executability safety reason is invalid")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise RolloutError("executability safety reason count is invalid")
    for field_name in (
        "policy_action_stream_sha256",
        "servo_target_stream_sha256",
        "executed_action_stream_sha256",
    ):
        _sha256_text(value.get(field_name), name=field_name)

    if value.get("joint_limit_margin_contract") != JOINT_LIMIT_MARGIN_CONTRACT:
        raise RolloutError("executability joint margin contract is invalid")
    margins = value.get("joint_limit_margins")
    margin_fields = set(JOINT_LIMIT_MARGIN_FIELDS)
    if not isinstance(margins, Mapping) or set(margins) != margin_fields:
        raise RolloutError("executability joint margin fields are invalid")
    for name, margin in margins.items():
        if margin is None:
            continue
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not isfinite(float(margin))
        ):
            raise RolloutError(f"executability {name} is invalid")
    servo_margin_fields = margin_fields - {"decoded_policy_target_min"}
    margins_configured = any(margin is not None for margin in margins.values())
    if margins_configured:
        if policy_steps and margins.get("decoded_policy_target_min") is None:
            raise RolloutError("executability decoded policy margin is missing")
        if servo_steps and any(
            margins.get(name) is None for name in servo_margin_fields
        ):
            raise RolloutError("executability servo joint margins are missing")

    if value.get("contact_progression_contract") != CONTACT_PROGRESSION_CONTRACT:
        raise RolloutError("executability contact progression contract is invalid")
    if value.get("contact_index_hash") != CONTACT_INDEX_HASH:
        raise RolloutError("executability contact index hash contract is invalid")
    contact_limit = _count(
        value.get("contact_progression_limit"),
        name="contact_progression_limit",
    )
    if not 1 <= contact_limit <= MAX_CONTACT_TRACE_LIMIT:
        raise RolloutError("executability contact progression limit is outside its bound")
    contact_samples = _count(
        value.get("contact_progression_count"),
        name="contact_progression_count",
    )
    contact_truncated = value.get("contact_progression_truncated")
    if not isinstance(contact_truncated, bool) or contact_truncated is not (
        contact_samples > contact_limit
    ):
        raise RolloutError(
            "executability contact progression truncation marker is invalid"
        )
    progression = value.get("contact_progression")
    if not isinstance(progression, list) or len(progression) != min(
        contact_samples,
        contact_limit,
    ):
        raise RolloutError("executability contact progression inventory is invalid")

    sample_fields = {
        "policy_index",
        "servo_index",
        "phase",
        "contact_count",
        "forbidden_contact_count",
        "object_contact_count",
        "minimum_distance_m",
        "category_pairs",
        "contacts_sha256",
    }
    for sample in progression:
        if not isinstance(sample, Mapping) or set(sample) != sample_fields:
            raise RolloutError("executability contact sample fields are invalid")
        for name in ("policy_index", "servo_index"):
            _count(sample.get(name), name=f"contact {name}")
        phase = sample.get("phase")
        if not isinstance(phase, str) or not phase:
            raise RolloutError("executability contact sample phase is invalid")
        sample_count = _count(sample.get("contact_count"), name="contact_count")
        sample_forbidden = _count(
            sample.get("forbidden_contact_count"),
            name="forbidden_contact_count",
        )
        sample_objects = _count(
            sample.get("object_contact_count"),
            name="object_contact_count",
        )
        if sample_forbidden > sample_count or sample_objects > sample_count:
            raise RolloutError("executability contact sample counts are invalid")
        sample_minimum = sample.get("minimum_distance_m")
        if sample_minimum is None:
            if sample_count:
                raise RolloutError("executability contact sample minimum is missing")
        elif (
            isinstance(sample_minimum, bool)
            or not isinstance(sample_minimum, (int, float))
            or not isfinite(float(sample_minimum))
            or not sample_count
        ):
            raise RolloutError("executability contact sample minimum is invalid")
        pairs = sample.get("category_pairs")
        if not isinstance(pairs, list):
            raise RolloutError("executability contact category pairs are invalid")
        normalized_pairs: list[list[str]] = []
        for pair in pairs:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(item, str) or not item for item in pair)
                or pair != sorted(pair)
            ):
                raise RolloutError("executability contact category pair is invalid")
            normalized_pairs.append(list(pair))
        if normalized_pairs != sorted(normalized_pairs) or len(normalized_pairs) != len(
            {tuple(pair) for pair in normalized_pairs}
        ):
            raise RolloutError(
                "executability contact category pairs must be sorted and unique"
            )
        if len(normalized_pairs) > sample_count:
            raise RolloutError("executability contact category pair count is invalid")
        _sha256_text(sample.get("contacts_sha256"), name="contacts_sha256")

    for field_name in (
        "servo_index_stream_sha256",
        "contact_index_stream_sha256",
        "contact_stream_sha256",
    ):
        _sha256_text(value.get(field_name), name=field_name)
    contact_observations = _count(
        value.get("contact_observation_count"),
        name="contact_observation_count",
    )
    forbidden_observations = _count(
        value.get("forbidden_contact_observation_count"),
        name="forbidden_contact_observation_count",
    )
    object_observations = _count(
        value.get("object_contact_observation_count"),
        name="object_contact_observation_count",
    )
    if (
        forbidden_observations > contact_observations
        or object_observations > contact_observations
        or (not contact_samples and contact_observations)
    ):
        raise RolloutError("executability contact observation counts are invalid")
    minimum_distance = value.get("minimum_contact_distance_m")
    if minimum_distance is None:
        if contact_observations:
            raise RolloutError("executability minimum contact distance is missing")
    elif (
        isinstance(minimum_distance, bool)
        or not isinstance(minimum_distance, (int, float))
        or not isfinite(float(minimum_distance))
        or not contact_observations
    ):
        raise RolloutError("executability minimum contact distance is invalid")
    if not contact_truncated:
        if sum(int(sample["contact_count"]) for sample in progression) != contact_observations:
            raise RolloutError("executability contact observation total mismatch")
        if (
            sum(int(sample["forbidden_contact_count"]) for sample in progression)
            != forbidden_observations
        ):
            raise RolloutError("executability forbidden contact total mismatch")
        if (
            sum(int(sample["object_contact_count"]) for sample in progression)
            != object_observations
        ):
            raise RolloutError("executability object contact total mismatch")
        sample_minima = [
            float(sample["minimum_distance_m"])
            for sample in progression
            if sample["minimum_distance_m"] is not None
        ]
        expected_minimum = min(sample_minima) if sample_minima else None
        if expected_minimum != minimum_distance:
            raise RolloutError("executability minimum contact distance mismatch")

    collision_phase = value.get("collision_failure_phase")
    collision_contacts = value.get("collision_failure_contacts")
    if not isinstance(collision_contacts, list):
        raise RolloutError("executability collision failure contacts are invalid")
    normalized_collision_contacts = [
        _contact_record(contact, include_forbidden=False)
        for contact in collision_contacts
    ]
    if collision_phase is None:
        if normalized_collision_contacts:
            raise RolloutError("executability collision contacts require a phase")
    elif (
        not isinstance(collision_phase, str)
        or not collision_phase
        or not normalized_collision_contacts
        or not forbidden_observations
    ):
        raise RolloutError("executability collision failure evidence is invalid")

    if (
        value.get("future_latent_alignment_contract")
        != FUTURE_LATENT_ALIGNMENT_CONTRACT
    ):
        raise RolloutError("executability future latent alignment contract is invalid")
    if value.get("future_latent_stream_hash") != FUTURE_LATENT_STREAM_HASH:
        raise RolloutError("executability future latent stream hash is invalid")
    if value.get("future_latent_index_hash") != FUTURE_LATENT_INDEX_HASH:
        raise RolloutError("executability future latent index hash is invalid")
    future_limit = _count(
        value.get("future_latent_progression_limit"),
        name="future_latent_progression_limit",
    )
    if not 1 <= future_limit <= MAX_FUTURE_LATENT_TRACE_LIMIT:
        raise RolloutError(
            "executability future latent progression limit is outside its bound"
        )
    future_predictions = _count(
        value.get("future_latent_prediction_count"),
        name="future_latent_prediction_count",
    )
    future_observations = _count(
        value.get("future_latent_observation_count"),
        name="future_latent_observation_count",
    )
    future_pairs = _count(
        value.get("future_latent_aligned_pair_count"),
        name="future_latent_aligned_pair_count",
    )
    future_censored = _count(
        value.get("future_latent_censored_pair_count"),
        name="future_latent_censored_pair_count",
    )
    future_steps = _optional_positive_int(
        value.get("future_latent_future_steps"),
        name="future_latent_future_steps",
    )
    latent_dim = _optional_positive_int(
        value.get("future_latent_latent_dim"),
        name="future_latent_latent_dim",
    )
    if future_predictions != future_observations or future_predictions > policy_steps:
        raise RolloutError("executability future latent step counts are invalid")
    if bool(future_predictions) is not (
        future_steps is not None and latent_dim is not None
    ):
        raise RolloutError("executability future latent dimensions are invalid")

    expected_pairs = 0
    if future_steps is not None:
        expected_pairs = sum(
            max(0, future_predictions - offset)
            for offset in range(1, future_steps + 1)
        )
    if future_pairs != expected_pairs:
        raise RolloutError("executability future latent aligned pair count mismatch")
    expected_censored = (
        0
        if future_steps is None
        else future_predictions * future_steps - future_pairs
    )
    if future_censored != expected_censored:
        raise RolloutError("executability future latent censored pair count mismatch")

    future_elements = _count(
        value.get("future_latent_element_count"),
        name="future_latent_element_count",
    )
    expected_elements = (
        0
        if latent_dim is None
        else future_pairs * PRIMARY_CAMERA_COUNT * latent_dim
    )
    if future_elements != expected_elements:
        raise RolloutError("executability future latent element count mismatch")
    future_squared_error = _nonnegative_float(
        value.get("future_latent_squared_error_sum"),
        name="future_latent_squared_error_sum",
    )
    future_mse_value = value.get("future_latent_mse_mean")
    if not future_elements:
        if future_mse_value is not None or future_squared_error != 0.0:
            raise RolloutError("executability future latent empty MSE is invalid")
    else:
        future_mse = _nonnegative_float(
            future_mse_value,
            name="future_latent_mse_mean",
        )
        if not _same_float(
            future_mse,
            future_squared_error / future_elements,
        ):
            raise RolloutError("executability future latent MSE mismatch")

    by_offset = value.get("future_latent_mse_by_offset")
    expected_offset_keys = (
        set() if future_steps is None else {str(index) for index in range(1, future_steps + 1)}
    )
    latent_width = 0 if latent_dim is None else latent_dim
    if not isinstance(by_offset, Mapping) or set(by_offset) != expected_offset_keys:
        raise RolloutError("executability future latent offset fields are invalid")
    offset_pair_total = 0
    offset_element_total = 0
    offset_squared_error_total = 0.0
    offset_fields = {
        "aligned_pair_count",
        "element_count",
        "squared_error_sum",
        "mse_mean",
    }
    for offset_text in sorted(expected_offset_keys, key=int):
        offset = int(offset_text)
        item = by_offset[offset_text]
        if not isinstance(item, Mapping) or set(item) != offset_fields:
            raise RolloutError("executability future latent offset item is invalid")
        pair_count = _count(
            item.get("aligned_pair_count"),
            name="future latent offset aligned_pair_count",
        )
        expected_pair_count = max(0, future_predictions - offset)
        if pair_count != expected_pair_count:
            raise RolloutError("executability future latent offset pair count mismatch")
        element_count = _count(
            item.get("element_count"),
            name="future latent offset element_count",
        )
        expected_element_count = pair_count * PRIMARY_CAMERA_COUNT * latent_width
        if element_count != expected_element_count:
            raise RolloutError(
                "executability future latent offset element count mismatch"
            )
        squared_error = _nonnegative_float(
            item.get("squared_error_sum"),
            name="future latent offset squared_error_sum",
        )
        mse_value = item.get("mse_mean")
        if not element_count:
            if mse_value is not None or squared_error != 0.0:
                raise RolloutError(
                    "executability future latent empty offset MSE is invalid"
                )
        else:
            mse = _nonnegative_float(
                mse_value,
                name="future latent offset mse_mean",
            )
            if not _same_float(mse, squared_error / element_count):
                raise RolloutError("executability future latent offset MSE mismatch")
        offset_pair_total += pair_count
        offset_element_total += element_count
        offset_squared_error_total += squared_error
    if (
        offset_pair_total != future_pairs
        or offset_element_total != future_elements
        or not _same_float(offset_squared_error_total, future_squared_error)
    ):
        raise RolloutError("executability future latent offset totals mismatch")

    progression_count = _count(
        value.get("future_latent_progression_count"),
        name="future_latent_progression_count",
    )
    if progression_count != future_pairs:
        raise RolloutError("executability future latent progression count mismatch")
    progression_truncated = value.get("future_latent_progression_truncated")
    if not isinstance(progression_truncated, bool) or progression_truncated is not (
        future_pairs > future_limit
    ):
        raise RolloutError(
            "executability future latent progression truncation marker is invalid"
        )
    future_progression = value.get("future_latent_progression")
    if not isinstance(future_progression, list) or len(future_progression) != min(
        future_pairs,
        future_limit,
    ):
        raise RolloutError(
            "executability future latent progression inventory is invalid"
        )
    expected_indices = []
    if future_steps is not None:
        for target_index in range(future_predictions):
            for source_index in range(
                max(0, target_index - future_steps),
                target_index,
            ):
                expected_indices.append(
                    (source_index, target_index, target_index - source_index)
                )
    expected_index_digest = sha256()
    for source_index, target_index, offset in expected_indices:
        expected_index_digest.update(
            _future_latent_index_source(source_index, target_index, offset)
        )
    future_record_fields = {
        "source_policy_index",
        "target_policy_index",
        "offset",
        "source_observation_timestamp_s",
        "target_observation_timestamp_s",
        "elapsed_s",
        "element_count",
        "squared_error_sum",
        "mse",
        "predicted_sha256",
        "observed_sha256",
    }
    progression_squared_error = 0.0
    for position, item in enumerate(future_progression):
        if not isinstance(item, Mapping) or set(item) != future_record_fields:
            raise RolloutError("executability future latent record fields are invalid")
        source_index = _count(
            item.get("source_policy_index"),
            name="future latent source_policy_index",
        )
        target_index = _count(
            item.get("target_policy_index"),
            name="future latent target_policy_index",
        )
        offset = _count(item.get("offset"), name="future latent offset")
        if (source_index, target_index, offset) != expected_indices[position]:
            raise RolloutError("executability future latent record alignment is invalid")
        source_timestamp = _nonnegative_float(
            item.get("source_observation_timestamp_s"),
            name="future latent source timestamp",
        )
        target_timestamp = _nonnegative_float(
            item.get("target_observation_timestamp_s"),
            name="future latent target timestamp",
        )
        elapsed = _nonnegative_float(
            item.get("elapsed_s"),
            name="future latent elapsed_s",
        )
        if target_timestamp <= source_timestamp or not _same_float(
            elapsed,
            target_timestamp - source_timestamp,
        ):
            raise RolloutError("executability future latent record timing is invalid")
        item_elements = _count(
            item.get("element_count"),
            name="future latent record element_count",
        )
        if item_elements != PRIMARY_CAMERA_COUNT * latent_width:
            raise RolloutError("executability future latent record size is invalid")
        item_squared_error = _nonnegative_float(
            item.get("squared_error_sum"),
            name="future latent record squared_error_sum",
        )
        item_mse = _nonnegative_float(
            item.get("mse"),
            name="future latent record mse",
        )
        if not _same_float(item_mse, item_squared_error / item_elements):
            raise RolloutError("executability future latent record MSE mismatch")
        _sha256_text(item.get("predicted_sha256"), name="predicted_sha256")
        _sha256_text(item.get("observed_sha256"), name="observed_sha256")
        progression_squared_error += item_squared_error

    for field_name in (
        "future_latent_prediction_stream_sha256",
        "future_latent_observation_stream_sha256",
        "future_latent_index_stream_sha256",
    ):
        _sha256_text(value.get(field_name), name=field_name)
    if expected_index_digest.hexdigest() != value.get(
        "future_latent_index_stream_sha256"
    ):
        raise RolloutError(
            "executability future latent progression index hash mismatch"
        )
    if not progression_truncated:
        if not _same_float(progression_squared_error, future_squared_error):
            raise RolloutError(
                "executability future latent progression error total mismatch"
            )

    available = value.get("runtime_future_latent_proxy_available")
    if not isinstance(available, bool) or available is not bool(future_pairs):
        raise RolloutError("executability future latent availability claim is invalid")
    if value.get("runtime_future_latent_success_claimed") is not False:
        raise RolloutError("executability future latent success claim is invalid")

    normalized = dict(value)
    normalized["decoded_actions"] = [dict(action) for action in actions]
    normalized["safety_reason_counts"] = dict(reasons)
    normalized["joint_limit_margins"] = dict(margins)
    normalized["contact_progression"] = [
        {
            **sample,
            "category_pairs": [list(pair) for pair in sample["category_pairs"]],
        }
        for sample in progression
    ]
    normalized["collision_failure_contacts"] = normalized_collision_contacts
    normalized["future_latent_mse_by_offset"] = {
        str(offset): dict(by_offset[str(offset)])
        for offset in range(1, (future_steps or 0) + 1)
    }
    normalized["future_latent_progression"] = [
        dict(item) for item in future_progression
    ]
    return normalized


def validate_scored_trace(
    value: Mapping[str, Any],
    *,
    policy_steps: int,
) -> None:
    """Require complete execution evidence before accepting a scored trial."""

    expected_policy_steps = _count(policy_steps, name="expected_policy_steps")
    if expected_policy_steps < 1:
        raise RolloutError("scored executability policy_steps must be positive")
    if _count(value.get("policy_steps"), name="policy_steps") != expected_policy_steps:
        raise RolloutError("scored executability policy step count mismatch")
    if (
        _count(value.get("decoded_action_count"), name="decoded_action_count")
        != expected_policy_steps
    ):
        raise RolloutError("scored executability decoded action count mismatch")

    servo_steps = _count(value.get("servo_steps"), name="servo_steps")
    if servo_steps < 1:
        raise RolloutError("scored executability trace has no servo steps")
    if _count(value.get("safety_rejected_steps"), name="safety_rejected_steps"):
        raise RolloutError("scored executability trace contains a safety rejection")

    sent_actions = _count(value.get("sent_actions"), name="sent_actions")
    shadow_steps = _count(value.get("shadow_steps"), name="shadow_steps")
    if sent_actions + shadow_steps != servo_steps:
        raise RolloutError("scored executability execution coverage mismatch")


def validate_mujoco_scored_trace(
    value: Mapping[str, Any],
    *,
    policy_steps: int,
) -> None:
    """Require complete joint and contact evidence for a scored MuJoCo trial."""

    validate_scored_trace(value, policy_steps=policy_steps)
    margins = value.get("joint_limit_margins")
    if (
        not isinstance(margins, Mapping)
        or set(margins) != set(JOINT_LIMIT_MARGIN_FIELDS)
        or any(
            margin is None
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not isfinite(float(margin))
            or float(margin) < 0.0
            for margin in margins.values()
        )
    ):
        raise RolloutError("scored MuJoCo trace has invalid joint-limit margin")
    servo_steps = _count(value.get("servo_steps"), name="servo_steps")
    if (
        _count(
            value.get("contact_progression_count"),
            name="contact_progression_count",
        )
        != servo_steps
    ):
        raise RolloutError("scored MuJoCo trace contact progression is incomplete")
    if value.get("servo_index_stream_sha256") != value.get(
        "contact_index_stream_sha256"
    ):
        raise RolloutError("scored MuJoCo trace contact index stream mismatch")
    if _count(
        value.get("forbidden_contact_observation_count"),
        name="forbidden_contact_observation_count",
    ):
        raise RolloutError("scored MuJoCo trace contains a forbidden contact")
    if value.get("collision_failure_phase") is not None or value.get(
        "collision_failure_contacts"
    ):
        raise RolloutError("scored MuJoCo trace contains collision failure evidence")
    future_observations = _count(
        value.get("future_latent_observation_count"),
        name="future_latent_observation_count",
    )
    if future_observations and future_observations != policy_steps:
        raise RolloutError(
            "scored MuJoCo trace future latent observation coverage mismatch"
        )


def _history_steps(runtime: SO101WAMRuntime) -> int:
    value = getattr(runtime.policy, "required_history_steps", 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RolloutError("policy.required_history_steps must be a positive integer")
    return value


def _verify_actuation_home(runtime: SO101WAMRuntime, measured: np.ndarray) -> None:
    """Fail closed before policy output unless the measured pose is near the recorded home."""

    if (
        not runtime.config.runtime.actuation_enabled
        or runtime.config.runtime.backend != "lerobot"
    ):
        return
    home = runtime.config.lerobot.home_joint_position
    tolerance = runtime.config.lerobot.home_joint_tolerance
    if home is None or tolerance is None:
        raise RolloutError("real actuation requires home position and tolerance")

    error = np.abs(np.asarray(measured, dtype=np.float64) - np.asarray(home, dtype=np.float64))
    limits = np.asarray(tolerance, dtype=np.float64)
    violating = np.flatnonzero(error > limits)
    if violating.size:
        ratios = error[violating] / limits[violating]
        index = int(violating[int(np.argmax(ratios))])
        raise RolloutError(
            "measured pose is outside home tolerance at "
            f"{JOINT_KEYS[index]}: error={float(error[index]):.6g}, "
            f"tolerance={float(limits[index]):.6g}"
        )


def _sleep_until(
    deadline_s: float,
    *,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> float:
    now_s = float(clock())
    if not isfinite(now_s):
        raise RolloutError("rollout clock returned a non-finite timestamp")
    if now_s < deadline_s:
        sleeper(deadline_s - now_s)
        now_s = float(clock())
    if not isfinite(now_s) or now_s + 1e-9 < deadline_s:
        raise RolloutError("rollout clock did not advance to the requested deadline")
    return now_s


def run_managed_rollout(
    runtime: SO101WAMRuntime,
    robot: ManagedRobotAdapter,
    *,
    policy_steps: int,
    calibrate_on_connect: bool = False,
    clock: Callable[[], float] = monotonic,
    sleeper: Callable[[float], None] = sleep,
    terminal_observer: Callable[[float], None] | None = None,
    rollout_observer: RolloutObserver | None = None,
) -> RolloutSummary:
    """Connect, prime history, run a receding-horizon rollout, then disconnect.

    Real output remains controlled exclusively by ``runtime.config`` and the
    adapter's matching actuation flag. This function never enables it.
    """

    if policy_steps < 1:
        raise RolloutError("policy_steps must be positive")
    if runtime.state is not RuntimeState.PROMPT_CACHED:
        raise RolloutError(f"managed rollout requires prompt_cached state, got {runtime.state.value}")
    if robot.is_connected:
        raise RolloutError("managed rollout requires ownership of a disconnected robot")

    policy_dt_s = 1.0 / runtime.config.runtime.policy_hz
    servo_dt_s = 1.0 / runtime.config.runtime.servo_hz
    servo_ticks_per_policy = ceil(runtime.config.runtime.servo_hz / runtime.config.runtime.policy_hz)
    runtime.clock = clock

    servo_steps = 0
    sent_actions = 0
    shadow_steps = 0
    connected = False
    try:
        robot.connect(calibrate=calibrate_on_connect)
        connected = True
        start_s = float(clock())
        if not isfinite(start_s):
            raise RolloutError("rollout clock returned a non-finite start timestamp")

        history_steps = _history_steps(runtime)
        for index in range(history_steps):
            now_s = _sleep_until(start_s + index * policy_dt_s, clock=clock, sleeper=sleeper)
            snapshot = runtime.prime_live(now_s=now_s)
            _verify_actuation_home(runtime, snapshot.live_frames[-1].joint_position)

        rollout_start_s = start_s + history_steps * policy_dt_s
        for policy_index in range(policy_steps):
            policy_deadline_s = rollout_start_s + policy_index * policy_dt_s
            policy_now_s = _sleep_until(policy_deadline_s, clock=clock, sleeper=sleeper)
            policy_step = runtime.policy_step(now_s=policy_now_s)
            if rollout_observer is not None:
                rollout_observer.on_policy_step(
                    policy_index=policy_index,
                    step=policy_step,
                )

            for servo_index in range(min(runtime.executor.pending, servo_ticks_per_policy)):
                servo_deadline_s = policy_deadline_s + servo_index * servo_dt_s
                servo_now_s = _sleep_until(servo_deadline_s, clock=clock, sleeper=sleeper)
                step = runtime.servo_step(now_s=servo_now_s)
                if rollout_observer is not None:
                    rollout_observer.on_servo_step(
                        policy_index=policy_index,
                        servo_index=servo_index,
                        step=step,
                    )
                servo_steps += 1
                sent_actions += int(step.sent)
                shadow_steps += int(step.shadow)
                if not step.safety.accepted:
                    raise SafetyRejectedError(step.safety.reasons)

        end_s = float(clock())
        if terminal_observer is not None:
            terminal_observer(end_s)
        runtime.pause_rollout()
        return RolloutSummary(
            prompt_fingerprint=runtime.prompt_fingerprint,
            policy_steps=policy_steps,
            servo_steps=servo_steps,
            sent_actions=sent_actions,
            shadow_steps=shadow_steps,
            elapsed_s=max(0.0, end_s - start_s),
            final_state=runtime.state.value,
        )
    except Exception as error:
        if runtime.state is not RuntimeState.HALT:
            runtime.halt(f"managed_rollout:{type(error).__name__}")
        raise
    finally:
        if connected or robot.is_connected:
            robot.disconnect()


__all__ = [
    "ACTION_TRACE_HASH",
    "CONTACT_INDEX_HASH",
    "CONTACT_PROGRESSION_CONTRACT",
    "DEFAULT_ACTION_TRACE_LIMIT",
    "DEFAULT_CONTACT_TRACE_LIMIT",
    "DEFAULT_FUTURE_LATENT_TRACE_LIMIT",
    "EXECUTABILITY_TRACE_SCHEMA",
    "ExecutabilityTrace",
    "FUTURE_LATENT_ALIGNMENT_CONTRACT",
    "FUTURE_LATENT_INDEX_HASH",
    "FUTURE_LATENT_STREAM_HASH",
    "JOINT_LIMIT_MARGIN_CONTRACT",
    "JOINT_LIMIT_MARGIN_FIELDS",
    "MAX_ACTION_TRACE_LIMIT",
    "MAX_CONTACT_TRACE_LIMIT",
    "MAX_FUTURE_LATENT_TRACE_LIMIT",
    "ManagedRobotAdapter",
    "RolloutObserver",
    "RolloutError",
    "RolloutSummary",
    "SafetyRejectedError",
    "run_managed_rollout",
    "validate_executability_trace",
    "validate_mujoco_scored_trace",
    "validate_scored_trace",
]
