"""One-shot simulation-only gripper boundary evidence; never a policy fix."""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum
from functools import partial
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np

from so101_wam.adapters.mujoco import MujocoBiSOAdapter
from so101_wam.config import ProjectConfig
from so101_wam.constants import JOINT_KEYS, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from so101_wam.dataset import load_episode
from so101_wam.deployment import file_sha256, project_config_sha256
from so101_wam.reference_learning import _development_task, _require_inputs, _require_source
from so101_wam.rollout import SafetyRejectedError
from so101_wam.rollout import run_managed_rollout
from so101_wam.runtime import PolicyStep, SO101WAMRuntime, ServoStep
from so101_wam.semantic_pilot import _fixed_runtime, _runtime
from so101_wam.sim_reference import (
    SimTrialConfig,
    TrialKind,
    _VirtualClock,
    _initial_penetration,
    _policy_inputs,
)
from so101_wam.robot_free_cli import _write_json_no_overwrite


__all__ = ("PhysicalPrompt", "file_sha256", "_require_inputs")

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / "runs/reference_learning_normalized_001"
RECOVERY = REPO / "runs/reference_learning_normalized_recovery_001"
OUTPUT = REPO / "runs/gripper_boundary_diagnostic_001"
SEEDS = (7, 13)
MAX_SENDS = 1
SERVO_HZ = 50.0
POLICY_HZ = 10.0
TIMESTEP_S = 0.005
PHYSICS_STEPS = 4
GRIPPER = JOINT_KEYS.index("right_gripper.pos")
ORIGINAL_REASON = "observation_joint_limit:right_gripper.pos:below"
RECOVERY_SHA = "0fd99035c39e0388539181199bed7fdf0489279552f231f25863dcbd3179a055"
SCHEMA = "so101_wam.gripper_boundary_diagnostic.v1"
PROMPT_NAME = "reference_reference-nudge-block_seed-7_ep-000000_episode"

_T = TypeVar("_T")


class _Mismatch(ValueError):
    pass


class _BudgetEnd(RuntimeError):
    pass


class _Result(StrEnum):
    REPRODUCED = "original_rejection_reproduced"
    NOT_REPRODUCED = "not_reproduced_within_budget"
    FAILED = "diagnostic_execution_failure"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal(actual: Any, expected: Any, label: str) -> None:
    if np.array_equal(np.asarray(actual), np.asarray(expected)):
        return

    raise _Mismatch(f"{label} mismatch")


def _image_hashes(images: Any) -> dict[str, str]:
    return {
        key: hashlib.sha256(np.asarray(images[key]).tobytes(order="C")).hexdigest()
        for key in PRIMARY_CAMERA_KEYS
    }


def _historical() -> dict[str, str]:
    report_path = RECOVERY / "recovery-report.json"
    if file_sha256(report_path) != RECOVERY_SHA:
        raise _Mismatch("recovery report hash mismatch")

    report = _read(report_path)
    inventory = {
        str(RECOVERY / name): digest
        for name, digest in report["artifact_sha256"].items()
    }
    inventory[str(report_path)] = RECOVERY_SHA
    _require_inputs(inventory)

    protocol_path = RECOVERY / "recovery-protocol.json"
    protocol = _read(protocol_path)
    inventory.update(protocol["original_artifact_sha256"])
    _require_inputs(inventory)
    return inventory


def _context() -> tuple[dict[str, Any], ProjectConfig, Any]:
    original = _read(ORIGINAL / "protocol.json")
    config = ProjectConfig.from_mapping(original["config"])
    task = _development_task()
    task_payload = json.loads(json.dumps(asdict(task), allow_nan=False))

    if project_config_sha256(config) != original["config_sha256"]:
        raise _Mismatch("config hash mismatch")
    if task_payload != original["task"]:
        raise _Mismatch("task mismatch")
    if _runtime() != original["runtime"]:
        raise _Mismatch("runtime mismatch")
    if config.runtime.backend != "mujoco":
        raise _Mismatch("runtime backend mismatch")
    if not config.runtime.actuation_enabled:
        raise _Mismatch("actuation disabled")
    if not config.mujoco.forbid_collisions:
        raise _Mismatch("collisions not forbidden")
    if config.runtime.policy_hz != POLICY_HZ or config.runtime.servo_hz != SERVO_HZ:
        raise _Mismatch("frozen policy/servo rate mismatch")

    _require_source(original["source_sha256"])
    return original, config, task


def _freeze() -> None:
    if OUTPUT.exists() and (not OUTPUT.is_dir() or any(OUTPUT.iterdir())):
        raise _Mismatch("diagnostic output must be empty")

    inventory = _historical()
    original, config, _task = _context()
    _require_inputs(inventory)
    _require_source(original["source_sha256"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": SCHEMA,
        "seeds": list(SEEDS),
        "max_sends": MAX_SENDS,
        "input_sha256": inventory,
        "source_sha256": original["source_sha256"],
        "diagnostic_sha256": file_sha256(Path(__file__)),
        "config_sha256": project_config_sha256(config),
        "safety_lower": list(config.safety.joint_lower),
        "safety_upper": list(config.safety.joint_upper),
        "runtime": original["runtime"],
        "automatic_retries": False,
    }
    _write_json_no_overwrite(OUTPUT / "protocol.json", protocol)


def _verify(protocol: dict[str, Any]) -> None:
    original, config, _task = _context()
    inventory = _historical()

    checks = (
        (protocol.get("schema_version"), SCHEMA, "schema_version"),
        (protocol.get("seeds"), list(SEEDS), "seeds"),
        (protocol.get("max_sends"), MAX_SENDS, "max_sends"),
        (protocol.get("input_sha256"), inventory, "input_sha256"),
        (protocol.get("source_sha256"), original["source_sha256"], "source_sha256"),
        (protocol.get("diagnostic_sha256"), file_sha256(Path(__file__)), "diagnostic_sha256"),
        (protocol.get("config_sha256"), project_config_sha256(config), "config_sha256"),
        (protocol.get("safety_lower"), list(config.safety.joint_lower), "safety_lower"),
        (protocol.get("safety_upper"), list(config.safety.joint_upper), "safety_upper"),
        (protocol.get("runtime"), original["runtime"], "runtime"),
        (protocol.get("automatic_retries"), False, "automatic_retries"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise _Mismatch(f"{label} mismatch")

    _require_inputs(inventory)
    _require_source(original["source_sha256"])


def _guarded_load(protocol: dict[str, Any], loader: Callable[[], _T]) -> _T:
    _verify(protocol)
    loaded = loader()
    _verify(protocol)
    return loaded


def _trial_inputs(seed: int, root: Path) -> tuple[SimTrialConfig, dict[str, Any]]:
    original, config, task = _context()
    report = _read(RECOVERY / f"learned-{seed}-complete.json")
    outcome = report["outcome"]
    episode_payload = outcome["episode"]
    episode = load_episode(episode_payload["npz_path"], episode_payload["manifest_path"])

    if outcome["command_count"] != MAX_SENDS or episode.frame_count != MAX_SENDS:
        raise _Mismatch("one-shot episode mismatch")
    if outcome["config_sha256"] != original["config_sha256"]:
        raise _Mismatch("outcome config mismatch")
    if outcome["initial_previous_action"] != episode.joint_state[0].tolist():
        raise _Mismatch("initial previous action mismatch")
    if list(task.initial_position(seed)) != outcome["initial_object_position"]:
        raise _Mismatch("initial object position mismatch")

    images = {
        key: value
        for key, value in zip(PRIMARY_CAMERA_KEYS, episode.wrist_rgb[0], strict=True)
    }
    expected = {
        "joint_position": episode.joint_state[0].tolist(),
        "previous_action": outcome["initial_previous_action"],
        "command": episode.action[0].tolist(),
        "image_sha256": _image_hashes(images),
        "model_identity": outcome["mujoco_model_identity"],
        "object_body": task.object_body,
        "object_position": list(task.initial_position(seed)),
    }
    prompt_path = ORIGINAL / "control-0" / f"{PROMPT_NAME}.npz"
    trial = SimTrialConfig(
        config=config,
        task=task,
        seed=seed,
        output_dir=root,
        kind=TrialKind.LEARNED,
        duration_s=task.policy_steps / POLICY_HZ,
        episode_index=SEEDS.index(seed) + 4,
        checkpoint_path=RECOVERY / "candidate.pt",
        prompt_path=prompt_path,
        prompt_manifest_path=prompt_path.with_suffix(".json"),
        device="cpu",
    )
    return trial, expected


class _Journal:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._count = 0

    def add(self, kind: str, **payload: Any) -> None:
        event = {"kind": kind, **payload}
        path = self._root / f"event-{self._count:03d}.json"
        _write_json_no_overwrite(path, event)
        self._count += 1


class _Observer:
    def __init__(self, journal: _Journal) -> None:
        self._journal = journal
        self._proposal: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._reasons: tuple[str, ...] = ()

    def on_policy_step(self, *, policy_index: int, step: PolicyStep) -> None:
        self._proposal = np.array(
            step.action.target_joint_position,
            dtype=np.float32,
            copy=True,
        )
        self._journal.add(
            "policy",
            policy_index=policy_index,
            proposed_horizon=self._proposal.tolist(),
            created_at_s=step.action.created_at_s,
            dt_s=step.action.dt_s,
        )

    def on_servo_step(
        self,
        *,
        policy_index: int,
        servo_index: int,
        step: ServoStep,
    ) -> None:
        if self._proposal is None:
            raise _Mismatch("missing policy proposal")

        proposal = self._proposal
        self._reasons = tuple(step.safety.reasons)
        transmitted = step.executed_action.tolist() if step.sent else None
        self._journal.add(
            "servo",
            policy_index=policy_index,
            servo_index=servo_index,
            timestamp_s=step.observation.timestamp_s,
            observed=step.observation.joint_position.tolist(),
            proposed=proposal[servo_index].tolist(),
            safety_action=step.action.target_joint_position[0].tolist(),
            transmitted_action=transmitted,
            sent=step.sent,
            accepted=step.safety.accepted,
            clipped=step.safety.clipped,
            hold=step.safety.hold,
            reasons=list(self._reasons),
        )

    def reasons(self) -> tuple[str, ...]:
        return self._reasons


def _classify(
    error: BaseException | None,
    reasons: tuple[str, ...],
    sent: int,
) -> _Result:
    if sent != MAX_SENDS:
        return _Result.FAILED
    if isinstance(error, SafetyRejectedError):
        if reasons == (ORIGINAL_REASON,):
            return _Result.REPRODUCED
        return _Result.NOT_REPRODUCED
    if isinstance(error, _BudgetEnd):
        return _Result.NOT_REPRODUCED
    return _Result.FAILED


def _run_seed(seed: int) -> None:
    if seed not in SEEDS:
        raise _Mismatch("seed not in frozen seed set")
    if seed == 13 and not (OUTPUT / "seed-7" / "result.json").is_file():
        raise _Mismatch("seed 7 result required before seed 13")

    root = OUTPUT / f"seed-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    _write_json_no_overwrite(root / "attempt.json", {"seed": seed, "attempt": 1})

    journal = _Journal(root)
    observer = _Observer(journal)
    probe: _Probe | None = None
    error: BaseException | None = None
    protocol: dict[str, Any] | None = None
    protocol_sha256: str | None = None
    cleanup_error: str | None = None
    integrity_error: str | None = None
    phase = "read_protocol"

    try:
        protocol_path = OUTPUT / "protocol.json"
        protocol_sha256 = file_sha256(protocol_path)
        protocol = _read(protocol_path)

        phase = "load_trial_inputs"
        trial, expected = _guarded_load(
            protocol,
            partial(_trial_inputs, seed, root),
        )
        journal.add("equivalence_target", **expected)
        probe = _Probe(trial, expected, journal)

        phase = "initial_penetration"
        penetration, identity = _initial_penetration(probe)
        if penetration is not None:
            journal.add("initial_collision", penetration=penetration, model_identity=identity)
            raise _Mismatch("initial collision")

        phase = "load_policy_prompt"
        prompt, policy, metadata = _guarded_load(
            protocol,
            partial(_policy_inputs, trial, probe),
        )
        journal.add(
            "loaded",
            checkpoint=metadata,
            prompt_fingerprint=prompt.fingerprint,
            initial_identity=identity,
        )

        runtime = SO101WAMRuntime(
            config=trial.config,
            prompt=prompt,
            robot=probe,
            policy=policy,
        )
        clock = _VirtualClock()

        phase = "managed_rollout"
        run_managed_rollout(
            runtime,
            probe,
            policy_steps=1,
            calibrate_on_connect=False,
            clock=clock,
            sleeper=clock.sleep,
            rollout_observer=observer,
        )
        raise _Mismatch("expected stop missing")
    except BaseException as caught:
        error = caught
    finally:
        if probe is not None and probe.is_connected:
            try:
                probe.disconnect()
            except BaseException as caught:
                cleanup_error = f"{type(caught).__name__}: {caught}"
        if protocol is not None:
            try:
                _verify(protocol)
                if file_sha256(OUTPUT / "protocol.json") != protocol_sha256:
                    raise _Mismatch("protocol file changed after read")
            except BaseException as caught:
                integrity_error = f"{type(caught).__name__}: {caught}"

    counts = (
        probe.counts()
        if probe is not None
        else {"base_send_calls": 0, "sent_count": 0}
    )
    reasons = observer.reasons()
    status = _classify(error, reasons, int(counts["sent_count"]))
    if cleanup_error is not None or integrity_error is not None:
        status = _Result.FAILED

    result = {
        "schema_version": SCHEMA,
        "seed": seed,
        "status": status.value,
        **counts,
        "protocol_sha256": protocol_sha256,
        "stop_phase": phase,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
        "safety_reasons": list(reasons),
        "next_safety_decision_available": isinstance(error, SafetyRejectedError),
        "cleanup_error": cleanup_error,
        "integrity_error": integrity_error,
        "evidence_complete": False,
        "evidence_note": "final audit required",
    }
    _write_json_no_overwrite(root / "result.json", result)


class _Probe(MujocoBiSOAdapter):
    def __init__(
        self,
        trial: SimTrialConfig,
        expected: dict[str, Any],
        journal: _Journal,
    ) -> None:
        super().__init__(
            config=trial.config.mujoco,
            servo_hz=trial.config.runtime.servo_hz,
            actuation_enabled=trial.config.runtime.actuation_enabled,
            include_head_camera=trial.config.runtime.use_head_camera,
            semantic_object_body=trial.task.object_body,
            initial_object_position=trial.task.initial_position(trial.seed),
        )
        self._expected = expected
        self._journal = journal
        self._calls = 0
        self._sent = 0
        self._observations = 0
        self._last_observation: dict[str, Any] | None = None

    def _snapshot(self) -> str:
        if (
            self._bindings is None
            or self._model is None
            or self._data is None
            or self._coordinates is None
        ):
            raise _Mismatch("native state unavailable")

        joint_id = int(self._bindings.joint_ids[GRIPPER])
        actuator_id = int(self._bindings.actuator_ids[GRIPPER])
        qpos_address = int(self._bindings.qpos_addresses[GRIPPER])
        dof_address = int(self._bindings.dof_addresses[GRIPPER])
        qpos = float(self._data.qpos[qpos_address])
        low = float(self._coordinates.native_lower[GRIPPER])
        high = float(self._coordinates.native_upper[GRIPPER])
        canonical = (qpos - low) * 100.0 / (high - low)
        payload = {
            "actuator_id": actuator_id,
            "canonical_float32_calculated": float(np.float32(canonical)),
            "canonical_float64": canonical,
            "coordinate_margin_rad": qpos - low,
            "coordinate_range_rad": [low, high],
            "ctrl_range_rad": self._model.actuator_ctrlrange[actuator_id].tolist(),
            "dof_address": dof_address,
            "joint_id": joint_id,
            "joint_margin_rad": qpos - float(self._model.jnt_range[joint_id, 0]),
            "joint_name": "right_gripper",
            "joint_range_rad": self._model.jnt_range[joint_id].tolist(),
            "model_time_s": float(self._data.time),
            "physics_step_count": self.physics_step_count,
            "physics_timestep_s": float(self._model.opt.timestep),
            "qpos_address": qpos_address,
            "qpos_rad": qpos,
            "qvel_rad_s": float(self._data.qvel[dof_address]),
            "raw_ctrl_rad": float(self._data.ctrl[actuator_id]),
        }
        return json.dumps(payload, allow_nan=False, sort_keys=True)

    def connect(self, *, calibrate: bool = True) -> None:
        try:
            super().connect(calibrate=calibrate)
            native = json.loads(self._snapshot())
            model_identity = self.model_identity()
            if model_identity != self._expected["model_identity"]:
                raise _Mismatch("model_identity mismatch")
            if native["physics_timestep_s"] != TIMESTEP_S:
                raise _Mismatch("physics timestep mismatch")
            if self.physics_steps_per_servo_tick != PHYSICS_STEPS:
                raise _Mismatch("physics steps mismatch")
            object_body = self._expected["object_body"]
            position = self.object_body_position(object_body)
            if list(position) != self._expected["object_position"]:
                raise _Mismatch("object position mismatch")
            self._journal.add(
                "connected",
                native=native,
                model_identity=model_identity,
                object_position=list(position),
            )
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self, *, timestamp_s: float) -> SensorimotorFrame:
        native = json.loads(self._snapshot())
        frame = super().get_observation(timestamp_s=timestamp_s)
        record = {
            "timestamp_s": timestamp_s,
            "joint_position": frame.joint_position.tolist(),
            "image_sha256": _image_hashes(frame.images),
            "native": native,
            "canonical_float32_observed": float(frame.joint_position[GRIPPER]),
        }
        self._last_observation = record
        self._journal.add("observation", index=self._observations, **record)
        self._observations += 1
        if self._calls == 0:
            _equal(record["joint_position"], self._expected["joint_position"], "joint_position")
            _equal(record["image_sha256"], self._expected["image_sha256"], "image_sha256")
        return frame

    def send_action(self, action: ActionChunk | np.ndarray) -> np.ndarray:
        if isinstance(action, ActionChunk):
            vector = np.asarray(action.target_joint_position[0])
        else:
            vector = np.asarray(action)
            if vector.ndim == 2:
                vector = vector[0]

        self._journal.add(
            "send_request",
            requested_action=vector.tolist(),
            base_calls=self._calls,
            last_observation=self._last_observation,
        )
        if self._calls >= MAX_SENDS:
            raise _BudgetEnd("second send blocked; SafetyDecision not returned")

        _equal(vector, self._expected["command"], "first command")
        self._journal.add("before_send", native=json.loads(self._snapshot()))
        self._calls += 1
        try:
            returned = super().send_action(action)
            self._sent += 1
            self._journal.add("adapter_return", action=returned.tolist())
            _equal(returned, self._expected["command"], "adapter return")
            return returned
        finally:
            self._journal.add(
                "after_send",
                native=json.loads(self._snapshot()),
                base_calls=self._calls,
                sent_count=self._sent,
            )

    def counts(self) -> dict[str, int]:
        return {"base_send_calls": self._calls, "sent_count": self._sent}


_NATIVE_FIELDS = frozenset(
    {
        "joint_name",
        "joint_id",
        "actuator_id",
        "qpos_address",
        "dof_address",
        "qpos_rad",
        "qvel_rad_s",
        "joint_range_rad",
        "ctrl_range_rad",
        "coordinate_range_rad",
        "raw_ctrl_rad",
        "joint_margin_rad",
        "coordinate_margin_rad",
        "canonical_float64",
        "canonical_float32_calculated",
        "model_time_s",
        "physics_timestep_s",
        "physics_step_count",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "seed",
        "status",
        "base_send_calls",
        "sent_count",
        "protocol_sha256",
        "stop_phase",
        "error_type",
        "error_message",
        "safety_reasons",
        "next_safety_decision_available",
        "cleanup_error",
        "integrity_error",
        "evidence_complete",
        "evidence_note",
    }
)
_AUDIT_FAILURES = (
    KeyError,
    ValueError,
    TypeError,
    ArithmeticError,
    StopIteration,
    OSError,
    IndexError,
)


def _failed(attempted: bool, **payload: Any) -> dict[str, Any]:
    return {
        "status": _Result.FAILED.value,
        "attempted": attempted,
        "evidence_complete": False,
        **payload,
    }


def _vector(value: Any, *, name: str) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != len(JOINT_KEYS):
        raise ValueError(f"{name} must contain {len(JOINT_KEYS)} values")
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _event_kind(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"{kind} evidence count mismatch")
    return matches[0]


def _events_kind(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("kind") == kind]


def _check_native(native: Mapping[str, Any]) -> None:
    if set(native) != _NATIVE_FIELDS:
        raise ValueError("native fields mismatch")

    # JSON serialization is the report boundary; reject non-finite floats here.
    json.dumps(native, allow_nan=False, sort_keys=True)
    joint_range = [float(item) for item in native["joint_range_rad"]]
    ctrl_range = [float(item) for item in native["ctrl_range_rad"]]
    coordinate_range = [float(item) for item in native["coordinate_range_rad"]]
    if len(joint_range) != 2 or len(ctrl_range) != 2 or len(coordinate_range) != 2:
        raise ValueError("native ranges must contain two values")

    joint_low, joint_high = joint_range
    ctrl_low, ctrl_high = ctrl_range
    expected_range = [max(joint_low, ctrl_low), min(joint_high, ctrl_high)]
    if coordinate_range != expected_range:
        raise ValueError("coordinate_range_rad mismatch")

    qpos = float(native["qpos_rad"])
    expected_canonical = (qpos - expected_range[0]) * 100.0 / (
        expected_range[1] - expected_range[0]
    )
    checks = (
        ("joint_margin_rad", qpos - joint_low),
        ("coordinate_margin_rad", qpos - expected_range[0]),
        ("canonical_float64", expected_canonical),
        ("canonical_float32_calculated", float(np.float32(expected_canonical))),
        ("physics_timestep_s", TIMESTEP_S),
    )
    for key, expected in checks:
        if native[key] != expected:
            raise ValueError(f"{key} mismatch")


def _audit_seed(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    attempted = (root / "attempt.json").is_file()
    if not (root / "result.json").is_file():
        return _failed(attempted, missing_evidence=["result.json"])

    try:
        if not attempted:
            raise ValueError("attempt marker missing")

        result = _read(root / "result.json")
        if not _RESULT_FIELDS <= set(result):
            raise ValueError("result fields missing")

        events = [_read(path) for path in sorted(root.glob("event-*.json"))]
        required = {
            "equivalence_target",
            "connected",
            "loaded",
            "policy",
            "observation",
            "send_request",
            "before_send",
            "adapter_return",
            "after_send",
            "servo",
        }
        kinds = {str(event["kind"]) for event in events}
        missing = sorted(required - kinds)
        if missing:
            raise ValueError(f"missing events: {missing}")

        if result["protocol_sha256"] != protocol["file_sha256"]:
            raise ValueError("protocol hash mismatch")
        if result["cleanup_error"] is not None or result["integrity_error"] is not None:
            raise ValueError("result cleanup/integrity error")
        if result["base_send_calls"] != MAX_SENDS or result["sent_count"] != MAX_SENDS:
            raise ValueError("send counts mismatch")

        expected = _event_kind(events, "equivalence_target")
        connected_events = _events_kind(events, "connected")
        if not connected_events:
            raise ValueError("missing connected evidence")
        before_events = _events_kind(events, "before_send")
        adapter_return_events = _events_kind(events, "adapter_return")
        after_events = _events_kind(events, "after_send")
        if len(before_events) != 1:
            raise ValueError("before_send evidence count mismatch")
        if len(adapter_return_events) != 1:
            raise ValueError("adapter_return evidence count mismatch")
        if len(after_events) != 1:
            raise ValueError("after_send evidence count mismatch")

        connected = connected_events[0]
        before = _event_kind(events, "before_send")["native"]
        _check_native(before)
        after_event = after_events[0]
        after = after_event["native"]
        _check_native(after)
        adapter_return = adapter_return_events[0]
        requests = _events_kind(events, "send_request")
        observations = _events_kind(events, "observation")
        servos = _events_kind(events, "servo")
        sent_servos = [servo for servo in servos if servo["sent"] is True]

        if len(requests) < 1:
            raise ValueError("missing request evidence")
        if len(sent_servos) != 1:
            raise ValueError("sent servo count mismatch")
        if sent_servos[0]["accepted"] is not True:
            raise ValueError("sent servo accepted mismatch")

        connected_native = connected["native"]
        _check_native(connected_native)
        for connected_event in connected_events:
            _check_native(connected_event["native"])
            if connected_event["model_identity"] != expected["model_identity"]:
                raise ValueError("model identity mismatch")
            if connected_event["object_position"] != expected["object_position"]:
                raise ValueError("object position mismatch")
            if connected_event["native"] != connected_native:
                raise ValueError("connected native mismatch")
            if connected_event["native"]["physics_step_count"] != 0:
                raise ValueError("connected physics count mismatch")

        if before["physics_step_count"] != 0:
            raise ValueError("before physics count mismatch")
        if after["physics_step_count"] - before["physics_step_count"] != PHYSICS_STEPS:
            raise ValueError("physics step delta mismatch")
        if after["model_time_s"] - before["model_time_s"] != 1.0 / SERVO_HZ:
            raise ValueError("model time delta mismatch")
        if after_event["base_calls"] != MAX_SENDS or after_event["sent_count"] != MAX_SENDS:
            raise ValueError("after counts mismatch")

        first_observation = observations[0]
        if first_observation["joint_position"] != expected["joint_position"]:
            raise ValueError("initial q mismatch")
        if first_observation["image_sha256"] != expected["image_sha256"]:
            raise ValueError("initial image hash mismatch")
        if requests[0]["requested_action"] != expected["command"]:
            raise ValueError("request action mismatch")
        if adapter_return["action"] != expected["command"]:
            raise ValueError("adapter action mismatch")
        if servos[0]["transmitted_action"] != expected["command"]:
            raise ValueError("sent action mismatch")
        if servos[0]["observed"] != expected["joint_position"]:
            raise ValueError("sent row observed mismatch")

        status = result["status"]
        if status == _Result.REPRODUCED.value:
            return _audit_reproduced(protocol, result, requests, observations, servos, after)
        if status != _Result.NOT_REPRODUCED.value:
            raise ValueError("unknown diagnostic status")
        return _audit_not_reproduced(result, requests, observations, servos, after)
    except _AUDIT_FAILURES as error:
        return _failed(
            attempted,
            missing_or_invalid_evidence=f"{type(error).__name__}: {error}",
        )


def _post_send_observation(
    observations: list[dict[str, Any]],
    after: dict[str, Any],
    accepted_timestamp_s: Any,
) -> tuple[dict[str, Any], list[float]]:
    if len(observations) < 2:
        raise ValueError("post-send observation missing")

    accepted_timestamp = _timestamp(accepted_timestamp_s, name="accepted servo timestamp")
    last_observation = observations[-1]
    observed = _vector(last_observation["joint_position"], name="post-send observation")
    timestamp = _timestamp(last_observation["timestamp_s"], name="post-send observation")
    prior_timestamps = [
        _timestamp(observation["timestamp_s"], name="prior observation")
        for observation in observations[:-1]
    ]
    if timestamp <= accepted_timestamp:
        raise ValueError("post-send observation must follow accepted servo")
    if timestamp <= max(prior_timestamps):
        raise ValueError("post-send observation must follow prior observations")
    _check_native(last_observation["native"])
    if last_observation["native"] != after:
        raise ValueError("post-send observation native mismatch")
    if last_observation["canonical_float32_observed"] != observed[GRIPPER]:
        raise ValueError("observed canonical mismatch")
    if after["canonical_float32_calculated"] != observed[GRIPPER]:
        raise ValueError("calculated canonical mismatch")

    return last_observation, observed


def _timestamp(value: Any, *, name: str) -> float:
    timestamp = float(value)
    if not isfinite(timestamp):
        raise ValueError(f"{name} timestamp must be finite")
    if timestamp < 0.0:
        raise ValueError(f"{name} timestamp must be nonnegative")
    return timestamp


def _audit_reproduced(
    protocol: dict[str, Any],
    result: Mapping[str, Any],
    requests: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    servos: list[dict[str, Any]],
    after: dict[str, Any],
) -> dict[str, Any]:
    if len(requests) != 1:
        raise ValueError("reproduced request count mismatch")
    if len(servos) != 2:
        raise ValueError("reproduced servo row count mismatch")
    if result["error_type"] != "SafetyRejectedError":
        raise ValueError("reproduced error type mismatch")
    if result["next_safety_decision_available"] is not True:
        raise ValueError("reproduced safety decision availability mismatch")
    if result["safety_reasons"] != [ORIGINAL_REASON]:
        raise ValueError("original safety reason mismatch")

    accepted = servos[0]
    if accepted["sent"] is not True or accepted["accepted"] is not True:
        raise ValueError("accepted servo evidence mismatch")

    rejected = servos[-1]
    if (
        rejected["sent"] is not False
        or rejected["accepted"] is not False
        or rejected["reasons"] != [ORIGINAL_REASON]
        or rejected["transmitted_action"] is not None
    ):
        raise ValueError("rejected servo evidence mismatch")

    last_observation, observed = _post_send_observation(
        observations,
        after,
        accepted["timestamp_s"],
    )
    if last_observation["joint_position"] != rejected["observed"]:
        raise ValueError("rejected observation q mismatch")
    if last_observation["timestamp_s"] != rejected["timestamp_s"]:
        raise ValueError("rejected observation timestamp mismatch")

    safety_lower = float(protocol["safety_lower"][GRIPPER])
    if observed[GRIPPER] >= safety_lower:
        raise ValueError("reproduced gripper did not cross lower safety limit")

    return {
        "status": _Result.REPRODUCED.value,
        "attempted": True,
        "evidence_complete": True,
        "native": after,
        "observed_gripper": observed[GRIPPER],
        "safety_lower": safety_lower,
        "substep_of_crossing": "not observed",
        "physical_mechanism": "not established",
    }


def _audit_not_reproduced(
    result: Mapping[str, Any],
    requests: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    servos: list[dict[str, Any]],
    after: dict[str, Any],
) -> dict[str, Any]:
    error_type = result["error_type"]
    accepted = servos[0]
    last_observation, _observed = _post_send_observation(
        observations,
        after,
        accepted["timestamp_s"],
    )
    if error_type == "_BudgetEnd":
        if len(requests) != 2 or len(servos) != 1:
            raise ValueError("budget evidence count mismatch")
        if requests[-1]["last_observation"] != last_observation:
            raise ValueError("budget last observation mismatch")
        if result["next_safety_decision_available"] is not False:
            raise ValueError("budget safety decision availability mismatch")
        if result["safety_reasons"] != servos[0]["reasons"]:
            raise ValueError("budget safety reasons mismatch")
        if ORIGINAL_REASON in servos[0]["reasons"]:
            raise ValueError("budget used original rejection reason")
    elif error_type == "SafetyRejectedError":
        if len(requests) != 1 or len(servos) != 2:
            raise ValueError("alternate rejection servo count mismatch")
        if result["next_safety_decision_available"] is not True:
            raise ValueError("alternate safety decision availability mismatch")
        rejected = servos[-1]
        if (
            rejected["sent"] is not False
            or rejected["accepted"] is not False
            or rejected["transmitted_action"] is not None
        ):
            raise ValueError("alternate rejection evidence mismatch")
        if not result["safety_reasons"]:
            raise ValueError("alternate safety reasons missing")
        if result["safety_reasons"] == [ORIGINAL_REASON]:
            raise ValueError("alternate used original rejection reason")
        if rejected["reasons"] != result["safety_reasons"]:
            raise ValueError("alternate safety reasons mismatch")
        if last_observation["joint_position"] != rejected["observed"]:
            raise ValueError("alternate observation q mismatch")
        if last_observation["timestamp_s"] != rejected["timestamp_s"]:
            raise ValueError("alternate observation timestamp mismatch")
    else:
        raise ValueError("unknown stop")

    return {
        "status": _Result.NOT_REPRODUCED.value,
        "attempted": True,
        "evidence_complete": True,
        "safety_reasons": result["safety_reasons"],
        "next_safety_decision_available": result["next_safety_decision_available"],
    }


def _artifact_sha256() -> dict[str, str]:
    report = OUTPUT / "report.json"
    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path != report)
    return {path.relative_to(OUTPUT).as_posix(): file_sha256(path) for path in paths}


def _report() -> None:
    protocol_path = OUTPUT / "protocol.json"
    protocol_sha256 = file_sha256(protocol_path)
    protocol = _read(protocol_path)
    protocol["file_sha256"] = protocol_sha256
    integrity_error = None
    try:
        _verify(protocol)
    except _AUDIT_FAILURES + (_Mismatch,) as error:
        integrity_error = f"{type(error).__name__}: {error}"

    trials = {
        str(seed): _audit_seed(OUTPUT / f"seed-{seed}", protocol)
        for seed in SEEDS
    }
    if integrity_error is not None:
        trials = {
            str(seed): _failed(
                (OUTPUT / f"seed-{seed}" / "attempt.json").is_file(),
                integrity_error=integrity_error,
            )
            for seed in SEEDS
        }

    report = {
        "schema_version": SCHEMA,
        "protocol_sha256": protocol_sha256,
        "trials": trials,
        "integrity_error": integrity_error,
        "artifact_sha256": _artifact_sha256(),
        "report_hash_self_excluded": True,
        "policy_improved": False,
        "zero_shot_claimed": False,
        "real_world_success_claimed": False,
        "deployment_ready": False,
    }
    _write_json_no_overwrite(OUTPUT / "report.json", report)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 1 or args[0] not in {"freeze", "7", "13", "report"}:
        raise SystemExit("usage: gripper_boundary_diagnostic.py freeze|7|13|report")

    with _fixed_runtime():
        if args[0] == "freeze":
            _freeze()
        elif args[0] == "report":
            _report()
        else:
            _run_seed(int(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
