import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from so101_wam.constants import ACTION_DIM
from so101_wam.config import ProjectConfig
from so101_wam.contracts import ActionChunk
from so101_wam.adapters.mujoco import MujocoCollisionError
from so101_wam.rollout import SafetyRejectedError
from so101_wam.runtime import PolicyStep, RuntimeState, ServoStep
from so101_wam.safety import SafetyDecision


@pytest.fixture
def diag():
    path = Path(__file__).parents[1] / "scripts" / "gripper_boundary_diagnostic.py"
    spec = importlib.util.spec_from_file_location("gripper_diag_test", path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_existing_file_preserved(tmp_path, diag):
    journal = diag._Journal(tmp_path)
    journal.add("initial", value=1.0)
    event = tmp_path / "event-000.json"
    before = event.read_bytes()

    fresh = diag._Journal(tmp_path)
    with pytest.raises(Exception, match="already exists"):
        fresh.add("initial", value=2.0)

    assert event.read_bytes() == before


def test_writer_rejects_nonfinite(tmp_path, diag):
    journal = diag._Journal(tmp_path)

    with pytest.raises(ValueError):
        journal.add("native", qpos=float("nan"))

    assert list(tmp_path.glob("event-*.json")) == []


def test_changed_input_fails(tmp_path, diag):
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    expected = {str(input_file): diag.file_sha256(input_file)}
    input_file.write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(Exception, match="input changed"):
        diag._require_inputs(expected)


def test_commands_require_exact(diag):
    with pytest.raises(diag._Mismatch):
        diag._equal(
            [0.0],
            [np.nextafter(np.float32(0), np.float32(1))],
            "command",
        )


@pytest.fixture
def probe(tmp_path, monkeypatch, diag):
    calls = []
    instance = diag._Probe.__new__(diag._Probe)
    instance._journal = diag._Journal(tmp_path)
    instance._expected = {"command": [0.0] * 12}
    instance._calls = 0
    instance._sent = 0
    instance._observations = 0
    instance._last_observation = None
    monkeypatch.setattr(
        instance,
        "_snapshot",
        lambda: '{"physics_step_count":0,"physics_timestep_s":0.005}',
    )

    def send_action(self, action):
        del self
        target = np.asarray(action.target_joint_position[0], dtype=np.float32)
        calls.append(target.copy())
        return target.copy()

    monkeypatch.setattr(diag.MujocoBiSOAdapter, "send_action", send_action)
    return instance, calls


def _command(diag, value=0):
    return diag.ActionChunk(
        target_joint_position=np.full((1, 12), value, dtype=np.float32),
        dt_s=0.02,
        created_at_s=100.1,
    )


def _trial(tmp_path):
    runtime = SimpleNamespace(
        servo_hz=50.0,
        actuation_enabled=True,
        use_head_camera=False,
    )
    task = SimpleNamespace(
        object_body="task_block",
        initial_position=lambda seed: (seed, 0.0, 0.4),
    )
    config = SimpleNamespace(
        mujoco=SimpleNamespace(model="model.xml"),
        runtime=runtime,
    )
    return SimpleNamespace(
        config=config,
        task=task,
        seed=7,
        output_dir=tmp_path,
    )


def _frame(diag, value=0.0):
    return diag.SensorimotorFrame(
        timestamp_s=100.0,
        images={
            "left_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            "right_wrist": np.ones((2, 2, 3), dtype=np.uint8),
        },
        joint_position=np.full(12, value, dtype=np.float32),
    )


def test_probe_constructor_delegates_expected_adapter_args(tmp_path, monkeypatch, diag):
    kwargs = {}

    def init(self, **payload):
        del self
        kwargs.update(payload)

    monkeypatch.setattr(diag.MujocoBiSOAdapter, "__init__", init)
    trial = _trial(tmp_path)
    expected = {"command": [0.0] * 12}
    journal = diag._Journal(tmp_path)

    instance = diag._Probe(trial, expected, journal)

    assert kwargs == {
        "config": trial.config.mujoco,
        "servo_hz": 50.0,
        "actuation_enabled": True,
        "include_head_camera": False,
        "semantic_object_body": "task_block",
        "initial_object_position": (7, 0.0, 0.4),
    }
    assert instance.counts() == {"base_send_calls": 0, "sent_count": 0}


def test_probe_connect_records_native_identity_and_object(tmp_path, monkeypatch, diag):
    expected = {
        "command": [0.0] * 12,
        "model_identity": {"engine_version": "test"},
        "object_body": "task_block",
        "object_position": [0.2, 0.0, 0.4],
    }
    instance = diag._Probe.__new__(diag._Probe)
    instance._journal = diag._Journal(tmp_path)
    instance._expected = expected
    instance._calls = 0
    instance._sent = 0
    instance._observations = 0
    instance._last_observation = None
    instance._physics_steps_per_servo_tick = diag.PHYSICS_STEPS
    monkeypatch.setattr(
        instance,
        "_snapshot",
        lambda: '{"physics_step_count":0,"physics_timestep_s":0.005}',
    )
    monkeypatch.setattr(
        diag.MujocoBiSOAdapter,
        "connect",
        lambda self, calibrate=True: None,
    )
    monkeypatch.setattr(diag.MujocoBiSOAdapter, "disconnect", lambda self: None)
    monkeypatch.setattr(
        diag.MujocoBiSOAdapter,
        "model_identity",
        lambda self: {"engine_version": "test"},
    )
    monkeypatch.setattr(
        diag.MujocoBiSOAdapter,
        "object_body_position",
        lambda self, body: (0.2, 0.0, 0.4),
    )

    instance.connect()

    event = json.loads((tmp_path / "event-000.json").read_text(encoding="utf-8"))
    assert event["kind"] == "connected"
    assert event["native"] == {"physics_step_count": 0, "physics_timestep_s": 0.005}
    assert event["model_identity"] == {"engine_version": "test"}
    assert event["object_position"] == [0.2, 0.0, 0.4]


def test_probe_observation_records_first_frame_evidence(tmp_path, monkeypatch, diag):
    frame = _frame(diag)
    expected = {
        "command": [0.0] * 12,
        "joint_position": [0.0] * 12,
        "image_sha256": diag._image_hashes(frame.images),
    }
    instance = diag._Probe.__new__(diag._Probe)
    instance._journal = diag._Journal(tmp_path)
    instance._expected = expected
    instance._calls = 0
    instance._sent = 0
    instance._observations = 0
    instance._last_observation = None
    monkeypatch.setattr(
        instance,
        "_snapshot",
        lambda: '{"canonical_float32_calculated":12.5}',
    )
    monkeypatch.setattr(
        diag.MujocoBiSOAdapter,
        "get_observation",
        lambda self, timestamp_s: frame,
    )

    assert instance.get_observation(timestamp_s=101.0) is frame

    event = json.loads((tmp_path / "event-000.json").read_text(encoding="utf-8"))
    assert event["kind"] == "observation"
    assert event["timestamp_s"] == 101.0
    assert event["canonical_float32_observed"] == 0.0
    assert event["native"] == {"canonical_float32_calculated": 12.5}
    assert instance._last_observation == {
        "timestamp_s": 101.0,
        "joint_position": [0.0] * 12,
        "image_sha256": expected["image_sha256"],
        "native": {"canonical_float32_calculated": 12.5},
        "canonical_float32_observed": 0.0,
    }


def test_probe_first_observation_mismatch_keeps_evidence(tmp_path, monkeypatch, diag):
    instance = diag._Probe.__new__(diag._Probe)
    instance._journal = diag._Journal(tmp_path)
    instance._expected = {
        "command": [0.0] * 12,
        "joint_position": [0.0] * 12,
        "image_sha256": {},
    }
    instance._calls = 0
    instance._sent = 0
    instance._observations = 0
    instance._last_observation = None
    monkeypatch.setattr(instance, "_snapshot", lambda: '{"physics_step_count":0}')
    monkeypatch.setattr(
        diag.MujocoBiSOAdapter,
        "get_observation",
        lambda self, timestamp_s: _frame(diag, value=1.0),
    )

    with pytest.raises(diag._Mismatch, match="joint_position"):
        instance.get_observation(timestamp_s=101.0)

    assert (tmp_path / "event-000.json").is_file()


def test_probe_blocks_second_send(probe, diag):
    instance, calls = probe

    instance.send_action(_command(diag))
    with pytest.raises(diag._BudgetEnd, match="second send blocked"):
        instance.send_action(_command(diag))

    assert len(calls) == 1
    assert instance.counts() == {"base_send_calls": 1, "sent_count": 1}


def test_probe_rejects_first_command_mismatch(probe, diag):
    instance, calls = probe

    with pytest.raises(diag._Mismatch, match="first command"):
        instance.send_action(_command(diag, value=1))

    assert calls == []
    assert instance.counts() == {"base_send_calls": 0, "sent_count": 0}


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_probe_base_exception_consumes_budget(tmp_path, monkeypatch, diag, error_type):
    instance = diag._Probe.__new__(diag._Probe)
    instance._journal = diag._Journal(tmp_path)
    instance._expected = {"command": [0.0] * 12}
    instance._calls = 0
    instance._sent = 0
    instance._observations = 0
    instance._last_observation = None
    monkeypatch.setattr(instance, "_snapshot", lambda: '{"physics_step_count":0}')

    def send_action(self, action):
        del self, action
        raise error_type("base failure")

    monkeypatch.setattr(diag.MujocoBiSOAdapter, "send_action", send_action)

    with pytest.raises(error_type, match="base failure"):
        instance.send_action(_command(diag))
    with pytest.raises(diag._BudgetEnd, match="second send blocked"):
        instance.send_action(_command(diag))

    assert instance.counts() == {"base_send_calls": 1, "sent_count": 0}


def test_probe_rejects_adapter_return_mismatch(tmp_path, monkeypatch, diag):
    instance = diag._Probe.__new__(diag._Probe)
    instance._journal = diag._Journal(tmp_path)
    instance._expected = {"command": [0.0] * 12}
    instance._calls = 0
    instance._sent = 0
    instance._observations = 0
    instance._last_observation = None
    monkeypatch.setattr(instance, "_snapshot", lambda: '{"physics_step_count":0}')

    def send_action(self, action):
        del self, action
        return np.ones(12, dtype=np.float32)

    monkeypatch.setattr(diag.MujocoBiSOAdapter, "send_action", send_action)

    with pytest.raises(diag._Mismatch, match="adapter return"):
        instance.send_action(_command(diag))

    assert instance.counts() == {"base_send_calls": 1, "sent_count": 1}


def test_probe_native_snapshot_is_immutable_and_read_only(diag):
    instance = diag._Probe.__new__(diag._Probe)
    instance._bindings = SimpleNamespace(
        joint_ids=np.arange(12),
        actuator_ids=np.arange(12),
        qpos_addresses=np.arange(12),
        dof_addresses=np.arange(12),
    )
    instance._model = SimpleNamespace(
        jnt_range=np.tile([-.174533, 1.7453292], (12, 1)),
        actuator_ctrlrange=np.tile([-.17453, 1.74533], (12, 1)),
        opt=SimpleNamespace(timestep=.005),
    )
    instance._coordinates = SimpleNamespace(
        native_lower=instance._model.actuator_ctrlrange[:, 0].copy(),
        native_upper=instance._model.jnt_range[:, 1].copy(),
    )
    instance._data = SimpleNamespace(
        qpos=np.full(12, -.174531),
        qvel=np.zeros(12),
        ctrl=np.full(12, -.17453),
        time=.02,
    )
    instance._physics_step_count = 4
    before = (
        instance._data.qpos.tobytes(),
        instance._data.qvel.tobytes(),
        instance._data.ctrl.tobytes(),
    )

    first = json.loads(instance._snapshot())
    second = json.loads(instance._snapshot())
    after = (
        instance._data.qpos.tobytes(),
        instance._data.qvel.tobytes(),
        instance._data.ctrl.tobytes(),
    )

    assert first == second
    assert after == before
    assert first["joint_margin_rad"] > 0
    assert first["coordinate_margin_rad"] < 0


def _policy_step(diag, *, value: float = 2.0) -> PolicyStep:
    action = diag.ActionChunk(
        target_joint_position=np.full((2, ACTION_DIM), value, dtype=np.float32),
        dt_s=0.02,
        created_at_s=100.0,
    )
    return PolicyStep(snapshot=object(), batch=object(), action=action)


def _rejected_servo_step(diag) -> ServoStep:
    observed = np.full(ACTION_DIM, -0.001, dtype=np.float32)
    frame = diag.SensorimotorFrame(
        timestamp_s=100.12,
        images={
            "left_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            "right_wrist": np.ones((2, 2, 3), dtype=np.uint8),
        },
        joint_position=observed,
    )
    safety_action = diag.ActionChunk(
        target_joint_position=np.zeros((1, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=100.12,
    )
    decision = SafetyDecision(
        accepted=False,
        action=safety_action,
        reasons=(diag.ORIGINAL_REASON,),
        hold=True,
    )
    return ServoStep(
        state=RuntimeState.RECOVERY,
        observation=frame,
        action=safety_action,
        safety=decision,
        executed_action=np.zeros(ACTION_DIM, dtype=np.float32),
        sent=False,
        shadow=False,
        pending=0,
    )


def test_observer_preserves_rejected_servo_rows(tmp_path, diag):
    observer = diag._Observer(diag._Journal(tmp_path))

    observer.on_policy_step(policy_index=0, step=_policy_step(diag))
    observer.on_servo_step(
        policy_index=0,
        servo_index=0,
        step=_rejected_servo_step(diag),
    )

    policy = json.loads((tmp_path / "event-000.json").read_text(encoding="utf-8"))
    servo = json.loads((tmp_path / "event-001.json").read_text(encoding="utf-8"))

    assert policy == {
        "kind": "policy",
        "policy_index": 0,
        "proposed_horizon": [[2.0] * ACTION_DIM, [2.0] * ACTION_DIM],
        "created_at_s": 100.0,
        "dt_s": 0.02,
    }
    assert servo == {
        "kind": "servo",
        "policy_index": 0,
        "servo_index": 0,
        "timestamp_s": 100.12,
        "observed": [-0.0010000000474974513] * ACTION_DIM,
        "proposed": [2.0] * ACTION_DIM,
        "safety_action": [0.0] * ACTION_DIM,
        "transmitted_action": None,
        "sent": False,
        "accepted": False,
        "clipped": False,
        "hold": True,
        "reasons": [diag.ORIGINAL_REASON],
    }
    assert observer.reasons() == (diag.ORIGINAL_REASON,)
    assert list(tmp_path.glob("*.npz")) == []


def test_observer_requires_policy_proposal(tmp_path, diag):
    observer = diag._Observer(diag._Journal(tmp_path))

    with pytest.raises(diag._Mismatch, match="missing policy proposal"):
        observer.on_servo_step(
            policy_index=0,
            servo_index=0,
            step=_rejected_servo_step(diag),
        )


def test_observer_copies_policy_proposal(tmp_path, diag):
    observer = diag._Observer(diag._Journal(tmp_path))
    source = np.full((2, ACTION_DIM), 2.0, dtype=np.float32)
    action = diag.ActionChunk(
        target_joint_position=source,
        dt_s=0.02,
        created_at_s=100.0,
    )
    policy = PolicyStep(snapshot=object(), batch=object(), action=action)

    observer.on_policy_step(policy_index=0, step=policy)
    source[0, 0] = 99.0
    observer.on_servo_step(
        policy_index=0,
        servo_index=0,
        step=_rejected_servo_step(diag),
    )

    servo = json.loads((tmp_path / "event-001.json").read_text(encoding="utf-8"))
    assert servo["proposed"] == [2.0] * ACTION_DIM


@pytest.mark.parametrize(
    ("error_kind", "reasons_kind", "sent", "expected"),
    (
        ("budget", "original", 1, "not_reproduced"),
        ("rejected", "alternate", 1, "not_reproduced"),
        ("rejected", "original", 1, "reproduced"),
        ("mismatch", "original", 1, "failed"),
        ("budget", "original", 0, "failed"),
        ("none", "original", 1, "failed"),
    ),
)
def test_classify_diagnostic_outcome(diag, error_kind, reasons_kind, sent, expected):
    reasons = {
        "alternate": ("alternate",),
        "original": (diag.ORIGINAL_REASON,),
    }[reasons_kind]
    error = {
        "budget": diag._BudgetEnd("budget"),
        "rejected": SafetyRejectedError(reasons),
        "mismatch": diag._Mismatch("mismatch"),
        "none": None,
    }[error_kind]
    expected_result = {
        "failed": diag._Result.FAILED,
        "not_reproduced": diag._Result.NOT_REPRODUCED,
        "reproduced": diag._Result.REPRODUCED,
    }[expected]

    assert diag._classify(error, reasons, sent) is expected_result


def test_freeze_rejects_nonempty_output_before_historical(tmp_path, monkeypatch, diag):
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "keep.json"
    existing.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(diag, "OUTPUT", output)
    monkeypatch.setattr(
        diag,
        "_historical",
        lambda: pytest.fail("historical loaded before output guard"),
    )

    with pytest.raises(diag._Mismatch, match="diagnostic output must be empty"):
        diag._freeze()

    assert existing.read_text(encoding="utf-8") == "keep"


def test_freeze_writes_protocol_and_returns_none(tmp_path, monkeypatch, diag):
    output = tmp_path / "output"
    inventory = {str(tmp_path / "candidate.pt"): "a" * 64}
    source = {"src/x.py": "b" * 64}
    runtime = {"device": "cpu"}
    config = SimpleNamespace(
        safety=SimpleNamespace(joint_lower=(0.0,), joint_upper=(1.0,)),
    )
    monkeypatch.setattr(diag, "OUTPUT", output)
    monkeypatch.setattr(diag, "_historical", lambda: inventory)
    monkeypatch.setattr(
        diag,
        "_context",
        lambda: ({"source_sha256": source, "runtime": runtime}, config, object()),
    )
    monkeypatch.setattr(diag, "project_config_sha256", lambda value: "c" * 64)
    monkeypatch.setattr(diag, "_require_inputs", lambda value: None)
    monkeypatch.setattr(diag, "_require_source", lambda value: None)

    assert diag._freeze() is None

    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["automatic_retries"] is False


def test_guarded_load_verifies_exact_order(monkeypatch, diag):
    calls = []
    protocol = {"schema_version": diag.SCHEMA}

    monkeypatch.setattr(diag, "_verify", lambda value: calls.append(("verify", value)))

    def loader() -> str:
        calls.append(("load", protocol))
        return "loaded"

    assert diag._guarded_load(protocol, loader) == "loaded"
    assert calls == [("verify", protocol), ("load", protocol), ("verify", protocol)]


def test_guarded_load_raises_after_post_load_verify(monkeypatch, diag):
    calls = []
    protocol = {"schema_version": diag.SCHEMA}

    def verify(value):
        calls.append(("verify", value))
        if len(calls) == 3:
            raise diag._Mismatch("changed after load")

    monkeypatch.setattr(diag, "_verify", verify)

    with pytest.raises(diag._Mismatch, match="changed after load"):
        diag._guarded_load(protocol, lambda: calls.append(("load", protocol)))

    assert calls == [("verify", protocol), ("load", protocol), ("verify", protocol)]


def test_historical_builds_exact_inventory(tmp_path, monkeypatch, diag):
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    report = recovery / "recovery-report.json"
    protocol = recovery / "recovery-protocol.json"
    artifact = recovery / "candidate.pt"
    artifact.write_text("candidate", encoding="utf-8")
    report.write_text("report", encoding="utf-8")
    protocol.write_text("protocol", encoding="utf-8")
    expected = {
        "candidate.pt": diag.file_sha256(artifact),
        "recovery-report.json": diag.file_sha256(report),
        "recovery-protocol.json": diag.file_sha256(protocol),
    }
    report.write_text(
        json.dumps({"artifact_sha256": {"candidate.pt": expected["candidate.pt"]}}),
        encoding="utf-8",
    )
    expected["recovery-report.json"] = diag.file_sha256(report)
    monkeypatch.setattr(diag, "RECOVERY", recovery)
    monkeypatch.setattr(diag, "RECOVERY_SHA", expected["recovery-report.json"])
    protocol.write_text(
        json.dumps(
            {
                "original_artifact_sha256": {
                    str(protocol): expected["recovery-protocol.json"]
                }
            }
        ),
        encoding="utf-8",
    )
    expected["recovery-protocol.json"] = diag.file_sha256(protocol)
    protocol.write_text(
        json.dumps(
            {
                "original_artifact_sha256": {
                    str(protocol): expected["recovery-protocol.json"]
                }
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        diag,
        "_require_inputs",
        lambda inventory: calls.append(dict(inventory)),
    )

    inventory = diag._historical()

    expected_paths = {
        str(artifact): expected["candidate.pt"],
        str(report): expected["recovery-report.json"],
        str(protocol): expected["recovery-protocol.json"],
    }
    assert inventory == expected_paths
    assert calls == [
        {
            str(artifact): expected["candidate.pt"],
            str(report): expected["recovery-report.json"],
        },
        expected_paths,
    ]


def test_verify_enforces_frozen_protocol_contract(tmp_path, monkeypatch, diag):
    source = {"src/x.py": "a" * 64}
    inventory = {str(tmp_path / "candidate.pt"): "b" * 64}
    runtime = {"device": "cpu"}
    config = SimpleNamespace(
        safety=SimpleNamespace(joint_lower=(0.0,), joint_upper=(1.0,)),
    )
    task = SimpleNamespace(policy_steps=56)
    protocol = {
        "schema_version": diag.SCHEMA,
        "seeds": [7, 13],
        "max_sends": 1,
        "input_sha256": inventory,
        "source_sha256": source,
        "diagnostic_sha256": diag.file_sha256(Path(diag.__file__)),
        "config_sha256": "c" * 64,
        "safety_lower": [0.0],
        "safety_upper": [1.0],
        "runtime": runtime,
        "automatic_retries": False,
    }
    monkeypatch.setattr(
        diag,
        "_context",
        lambda: ({"source_sha256": source, "runtime": runtime}, config, task),
    )
    monkeypatch.setattr(diag, "_historical", lambda: inventory)
    monkeypatch.setattr(diag, "_runtime", lambda: runtime)
    monkeypatch.setattr(diag, "project_config_sha256", lambda value: "c" * 64)
    input_calls = []
    source_calls = []
    monkeypatch.setattr(diag, "_require_inputs", lambda value: input_calls.append(value))
    monkeypatch.setattr(diag, "_require_source", lambda value: source_calls.append(value))

    diag._verify(protocol)

    assert input_calls == [inventory]
    assert source_calls == [source]
    protocol["automatic_retries"] = True
    with pytest.raises(diag._Mismatch, match="automatic_retries"):
        diag._verify(protocol)


def test_trial_inputs_uses_explicit_recovery_paths(tmp_path, monkeypatch, diag):
    recovery = tmp_path / "recovery"
    original = tmp_path / "original"
    recovery.mkdir()
    original.mkdir()
    prompt = original / "control-0" / "reference_reference-nudge-block_seed-7_ep-000000_episode.npz"
    prompt.parent.mkdir()
    manifest = prompt.with_suffix(".json")
    prompt.write_text("prompt", encoding="utf-8")
    manifest.write_text("manifest", encoding="utf-8")
    monkeypatch.setattr(diag, "RECOVERY", recovery)
    monkeypatch.setattr(diag, "ORIGINAL", original)

    config = SimpleNamespace()
    task = SimpleNamespace(
        policy_steps=56,
        object_body="task_block",
        initial_position=lambda seed: [seed, 0.0, 0.4],
    )
    original_payload = {"config_sha256": "cfg"}
    monkeypatch.setattr(diag, "_context", lambda: (original_payload, config, task))
    monkeypatch.setattr(diag, "_read", lambda path: {
        "outcome": {
            "command_count": 1,
            "config_sha256": "cfg",
            "initial_previous_action": [float(path.name.split("-")[1])] * ACTION_DIM,
            "initial_object_position": [7, 0.0, 0.4],
            "mujoco_model_identity": {"id": path.name},
            "episode": {
                "npz_path": str(path.with_suffix(".npz")),
                "manifest_path": str(path.with_suffix(".json")),
            },
        }
    })

    def load_episode(npz_path, manifest_path):
        assert str(npz_path).endswith("learned-7-complete.npz")
        assert str(manifest_path).endswith("learned-7-complete.json")
        seed = 7.0
        return SimpleNamespace(
            frame_count=1,
            joint_state=np.full((1, ACTION_DIM), seed, dtype=np.float32),
            action=np.full((1, ACTION_DIM), 2.0, dtype=np.float32),
            wrist_rgb=np.arange(2 * 1 * 1 * 3, dtype=np.uint8).reshape(1, 2, 1, 1, 3),
        )

    monkeypatch.setattr(diag, "load_episode", load_episode)

    trial, expected = diag._trial_inputs(7, tmp_path / "out")

    assert trial.prompt_path == prompt
    assert trial.prompt_manifest_path == manifest
    assert trial.checkpoint_path == recovery / "candidate.pt"
    assert trial.episode_index == 4
    assert trial.duration_s == pytest.approx(5.6)
    assert expected["previous_action"] == [7.0] * ACTION_DIM
    assert expected["command"] == [2.0] * ACTION_DIM
    assert expected["object_position"] == [7, 0.0, 0.4]


def test_run_seed_writes_failed_cleanup_result(tmp_path, monkeypatch, diag):
    output = tmp_path / "output"
    output.mkdir()
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps({"schema_version": diag.SCHEMA}), encoding="utf-8")
    monkeypatch.setattr(diag, "OUTPUT", output)
    monkeypatch.setattr(diag, "_verify", lambda protocol: None)

    trial = SimpleNamespace(config=object(), output_dir=output / "seed-7", task=object())
    expected = {"command": [0.0] * ACTION_DIM}

    def guarded(protocol, loader):
        loaded = loader()
        if isinstance(loaded, tuple) and len(loaded) == 2:
            return trial, expected
        return loaded

    def trial_inputs(seed, root):
        del seed, root
        return trial, expected

    def policy_inputs(trial_arg, probe):
        del trial_arg, probe
        return SimpleNamespace(fingerprint="prompt"), SimpleNamespace(), {"checkpoint_sha256": "sha"}

    class FakeProbe:
        def __init__(self, trial_arg, expected_arg, journal):
            del trial_arg, expected_arg, journal
            self.is_connected = True

        def disconnect(self):
            raise RuntimeError("cleanup failed")

        def counts(self):
            return {"base_send_calls": 1, "sent_count": 1}

    monkeypatch.setattr(diag, "_trial_inputs", trial_inputs)
    monkeypatch.setattr(diag, "_policy_inputs", policy_inputs)
    monkeypatch.setattr(diag, "_guarded_load", guarded)
    monkeypatch.setattr(diag, "_Probe", FakeProbe)
    monkeypatch.setattr(diag, "_initial_penetration", lambda probe: (None, {"id": "model"}))
    monkeypatch.setattr(diag, "SO101WAMRuntime", lambda **kwargs: object())
    monkeypatch.setattr(diag, "_VirtualClock", lambda: SimpleNamespace(sleep=lambda value: None))
    monkeypatch.setattr(diag, "run_managed_rollout", lambda *args, **kwargs: (_ for _ in ()).throw(SafetyRejectedError((diag.ORIGINAL_REASON,))))

    assert diag._run_seed(7) is None

    saved = json.loads((output / "seed-7" / "result.json").read_text(encoding="utf-8"))
    assert saved["status"] == diag._Result.FAILED.value
    assert "counts" not in saved
    assert saved["base_send_calls"] == 1
    assert saved["sent_count"] == 1
    assert saved["cleanup_error"] == "RuntimeError: cleanup failed"
    assert saved["next_safety_decision_available"] is True
    assert saved["evidence_complete"] is False

    loaded = json.loads((output / "seed-7" / "event-001.json").read_text(encoding="utf-8"))
    assert loaded["kind"] == "loaded"
    assert loaded["checkpoint"] == {"checkpoint_sha256": "sha"}


def _native(diag, *, qpos=-0.17453, time=0.0, count=0):
    low = -0.17453
    high = 1.7453292
    canonical = (qpos - low) * 100.0 / (high - low)
    return {
        "joint_name": "right_gripper",
        "joint_id": 11,
        "actuator_id": 11,
        "qpos_address": 11,
        "dof_address": 11,
        "qpos_rad": qpos,
        "qvel_rad_s": 0.0,
        "joint_range_rad": [-0.174533, high],
        "ctrl_range_rad": [low, 1.74533],
        "coordinate_range_rad": [low, high],
        "raw_ctrl_rad": low,
        "joint_margin_rad": qpos - -0.174533,
        "coordinate_margin_rad": qpos - low,
        "canonical_float64": canonical,
        "canonical_float32_calculated": float(np.float32(canonical)),
        "model_time_s": time,
        "physics_timestep_s": diag.TIMESTEP_S,
        "physics_step_count": count,
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _protocol(tmp_path):
    return {
        "schema_version": "test",
        "file_sha256": "p" * 64,
        "safety_lower": [0.0] * ACTION_DIM,
    }


def _expected(diag, *, observed=None):
    q = [0.0] * ACTION_DIM
    command = [0.25] * ACTION_DIM
    left, right = diag.PRIMARY_CAMERA_KEYS
    return {
        "joint_position": q,
        "previous_action": q,
        "command": command,
        "image_sha256": {left: "l", right: "r"},
        "model_identity": {"model": "fixture"},
        "object_body": "task_block",
        "object_position": [0.0, 0.0, 0.5],
        "observed_after": observed,
    }


def _event(root, event_index, kind, **payload):
    _write(root / f"event-{event_index:03d}.json", {"kind": kind, **payload})


def _positive_seed(root, diag, protocol, *, status=None):
    before = _native(diag, qpos=-0.17453, time=0.0, count=0)
    after = _native(diag, qpos=-0.174531, time=0.02, count=4)
    expected = _expected(diag)
    next_q = [0.0] * ACTION_DIM
    next_q[diag.GRIPPER] = after["canonical_float32_calculated"]

    _write(root / "attempt.json", {"seed": 7, "attempt": 1})
    _event(root, 0, "equivalence_target", **expected)
    _event(
        root,
        1,
        "connected",
        model_identity=expected["model_identity"],
        object_position=expected["object_position"],
        native=before,
    )
    _event(root, 2, "loaded", checkpoint={}, prompt_fingerprint="prompt")
    _event(root, 3, "policy", proposed_horizon=[expected["command"]])
    _event(
        root,
        4,
        "observation",
        index=0,
        timestamp_s=0.0,
        joint_position=expected["joint_position"],
        image_sha256=expected["image_sha256"],
        native=before,
        canonical_float32_observed=0.0,
    )
    _event(
        root,
        5,
        "send_request",
        requested_action=expected["command"],
        base_calls=0,
        last_observation=None,
    )
    _event(root, 6, "before_send", native=before)
    _event(root, 7, "adapter_return", action=expected["command"])
    _event(root, 8, "after_send", native=after, base_calls=1, sent_count=1)
    _event(
        root,
        9,
        "servo",
        timestamp_s=0.0,
        observed=expected["joint_position"],
        transmitted_action=expected["command"],
        sent=True,
        accepted=True,
        reasons=[],
    )
    _event(
        root,
        10,
        "observation",
        index=1,
        timestamp_s=0.02,
        joint_position=next_q,
        image_sha256=expected["image_sha256"],
        native=after,
        canonical_float32_observed=next_q[diag.GRIPPER],
    )
    _event(
        root,
        11,
        "servo",
        timestamp_s=0.02,
        observed=next_q,
        transmitted_action=None,
        sent=False,
        accepted=False,
        reasons=[diag.ORIGINAL_REASON],
    )
    _write(
        root / "result.json",
        {
            "schema_version": diag.SCHEMA,
            "seed": 7,
            "status": status or diag._Result.REPRODUCED.value,
            "base_send_calls": 1,
            "sent_count": 1,
            "protocol_sha256": protocol["file_sha256"],
            "stop_phase": "managed_rollout",
            "error_type": "SafetyRejectedError",
            "error_message": diag.ORIGINAL_REASON,
            "safety_reasons": [diag.ORIGINAL_REASON],
            "next_safety_decision_available": True,
            "cleanup_error": None,
            "integrity_error": None,
            "evidence_complete": False,
            "evidence_note": "final audit required",
        },
    )


def _set_result(root, **updates):
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(updates)
    _write(result_path, result)


def _budget_seed(root, diag, protocol, *, accepted_reasons=None):
    _positive_seed(root, diag, protocol, status=diag._Result.NOT_REPRODUCED.value)
    reasons = list(accepted_reasons or [])
    servo = json.loads((root / "event-009.json").read_text(encoding="utf-8"))
    servo["reasons"] = reasons
    _write(root / "event-009.json", servo)
    observation = json.loads((root / "event-010.json").read_text(encoding="utf-8"))
    expected = _expected(diag)
    _event(
        root,
        11,
        "send_request",
        requested_action=expected["command"],
        base_calls=1,
        last_observation=observation,
    )
    _set_result(
        root,
        error_type="_BudgetEnd",
        error_message="second send blocked; SafetyDecision not returned",
        safety_reasons=reasons,
        next_safety_decision_available=False,
    )


def _alternate_seed(root, diag, protocol):
    _positive_seed(root, diag, protocol, status=diag._Result.NOT_REPRODUCED.value)
    servo = json.loads((root / "event-011.json").read_text(encoding="utf-8"))
    servo["reasons"] = ["alternate"]
    _write(root / "event-011.json", servo)
    _set_result(
        root,
        error_type="SafetyRejectedError",
        error_message="alternate",
        safety_reasons=["alternate"],
    )


def test_check_native_requires_exact_native_contract(diag):
    native = _native(diag)

    assert diag._check_native(native) is None
    missing = dict(native)
    missing.pop("joint_id")
    with pytest.raises(ValueError, match="native fields"):
        diag._check_native(missing)
    bad_range = dict(native)
    bad_range["coordinate_range_rad"] = native["joint_range_rad"]
    with pytest.raises(ValueError, match="coordinate_range_rad"):
        diag._check_native(bad_range)
    bad_json = dict(native)
    bad_json["qpos_rad"] = float("nan")
    with pytest.raises(ValueError):
        diag._check_native(bad_json)


def test_audit_missing_and_invalid_evidence_fail_closed(tmp_path, diag):
    protocol = _protocol(tmp_path)

    missing = diag._audit_seed(tmp_path / "missing", protocol)
    assert missing["status"] == diag._Result.FAILED.value
    assert missing["attempted"] is False
    assert missing["evidence_complete"] is False
    assert missing["missing_evidence"] == ["result.json"]

    unbound = tmp_path / "unbound"
    _write(unbound / "attempt.json", {})
    _write(unbound / "result.json", {"status": diag._Result.REPRODUCED.value})
    assert diag._audit_seed(unbound, protocol)["status"] == diag._Result.FAILED.value

    no_attempt = tmp_path / "no_attempt"
    _positive_seed(no_attempt, diag, protocol)
    (no_attempt / "attempt.json").unlink()
    audited = diag._audit_seed(no_attempt, protocol)
    assert audited["status"] == diag._Result.FAILED.value
    assert audited["evidence_complete"] is False


def test_audit_rejects_short_observed_and_duplicate_transition(tmp_path, diag):
    protocol = _protocol(tmp_path)
    short = tmp_path / "short"
    _positive_seed(short, diag, protocol)
    event = json.loads((short / "event-011.json").read_text(encoding="utf-8"))
    event["observed"] = [0.0]
    _write(short / "event-011.json", event)

    duplicate = tmp_path / "duplicate"
    _positive_seed(duplicate, diag, protocol)
    _event(duplicate, 12, "before_send", native=_native(diag))

    assert diag._audit_seed(short, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(duplicate, protocol)["status"] == diag._Result.FAILED.value


def test_audit_accepts_positive_reproduced_fixture(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _positive_seed(root, diag, protocol)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.REPRODUCED.value
    assert audited["attempted"] is True
    assert audited["evidence_complete"] is True
    assert audited["observed_gripper"] < audited["safety_lower"]
    assert audited["native"]["physics_step_count"] == diag.PHYSICS_STEPS
    assert audited["physical_mechanism"] == "not established"


def test_audit_accepts_repeated_consistent_connected_events(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _positive_seed(root, diag, protocol)
    event = json.loads((root / "event-001.json").read_text(encoding="utf-8"))
    _write(root / "event-012.json", event)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.REPRODUCED.value


def test_audit_rejects_inconsistent_connected_events(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _positive_seed(root, diag, protocol)
    event = json.loads((root / "event-001.json").read_text(encoding="utf-8"))
    event["object_position"] = [9.0, 0.0, 0.5]
    _write(root / "event-012.json", event)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


def test_audit_reproduced_requires_one_send_request(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _positive_seed(root, diag, protocol)
    expected = _expected(diag)
    _event(
        root,
        12,
        "send_request",
        requested_action=expected["command"],
        base_calls=1,
        last_observation={},
    )

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


def test_audit_reproduced_requires_one_sent_servo(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _positive_seed(root, diag, protocol)
    event = json.loads((root / "event-009.json").read_text(encoding="utf-8"))
    event["sent"] = False
    _write(root / "event-009.json", event)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


def test_audit_accepts_budget_not_reproduced_fixture(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _budget_seed(root, diag, protocol)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.NOT_REPRODUCED.value
    assert audited["evidence_complete"] is True


def test_audit_accepts_clipped_budget_reasons(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _budget_seed(root, diag, protocol, accepted_reasons=["max_delta_clipped"])

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.NOT_REPRODUCED.value
    assert audited["evidence_complete"] is True
    assert audited["safety_reasons"] == ["max_delta_clipped"]


def test_audit_budget_requires_sent_and_accepted_servo(tmp_path, diag):
    protocol = _protocol(tmp_path)
    sent_false = tmp_path / "sent_false"
    _budget_seed(sent_false, diag, protocol)
    servo = json.loads((sent_false / "event-009.json").read_text(encoding="utf-8"))
    servo["sent"] = False
    _write(sent_false / "event-009.json", servo)

    accepted_false = tmp_path / "accepted_false"
    _budget_seed(accepted_false, diag, protocol)
    servo = json.loads((accepted_false / "event-009.json").read_text(encoding="utf-8"))
    servo["accepted"] = False
    _write(accepted_false / "event-009.json", servo)

    assert diag._audit_seed(sent_false, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(accepted_false, protocol)["status"] == diag._Result.FAILED.value


def test_audit_budget_rejects_contradictory_result_fields(tmp_path, diag):
    protocol = _protocol(tmp_path)
    available = tmp_path / "available"
    _budget_seed(available, diag, protocol)
    _set_result(available, next_safety_decision_available=True)

    mismatch = tmp_path / "mismatch"
    _budget_seed(mismatch, diag, protocol, accepted_reasons=["max_delta_clipped"])
    _set_result(mismatch, safety_reasons=["other"])

    original = tmp_path / "original"
    _budget_seed(original, diag, protocol, accepted_reasons=[diag.ORIGINAL_REASON])

    assert diag._audit_seed(available, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(mismatch, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(original, protocol)["status"] == diag._Result.FAILED.value


def test_audit_accepts_alternate_rejection_fixture(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _alternate_seed(root, diag, protocol)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.NOT_REPRODUCED.value
    assert audited["evidence_complete"] is True


def test_audit_alternate_rejects_contradictory_result_fields(tmp_path, diag):
    protocol = _protocol(tmp_path)
    unavailable = tmp_path / "unavailable"
    _alternate_seed(unavailable, diag, protocol)
    _set_result(unavailable, next_safety_decision_available=False)

    mismatch = tmp_path / "mismatch"
    _alternate_seed(mismatch, diag, protocol)
    servo = json.loads((mismatch / "event-011.json").read_text(encoding="utf-8"))
    servo["reasons"] = ["other"]
    _write(mismatch / "event-011.json", servo)

    original = tmp_path / "original"
    _alternate_seed(original, diag, protocol)
    servo = json.loads((original / "event-011.json").read_text(encoding="utf-8"))
    servo["reasons"] = [diag.ORIGINAL_REASON]
    _write(original / "event-011.json", servo)
    _set_result(original, safety_reasons=[diag.ORIGINAL_REASON])

    extra_request = tmp_path / "extra_request"
    _alternate_seed(extra_request, diag, protocol)
    expected = _expected(diag)
    _event(
        extra_request,
        12,
        "send_request",
        requested_action=expected["command"],
        base_calls=1,
        last_observation={},
    )

    assert diag._audit_seed(unavailable, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(mismatch, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(original, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(extra_request, protocol)["status"] == diag._Result.FAILED.value


def test_audit_alternate_requires_sent_and_accepted_servo(tmp_path, diag):
    protocol = _protocol(tmp_path)
    sent_false = tmp_path / "sent_false"
    _alternate_seed(sent_false, diag, protocol)
    servo = json.loads((sent_false / "event-009.json").read_text(encoding="utf-8"))
    servo["sent"] = False
    _write(sent_false / "event-009.json", servo)

    accepted_false = tmp_path / "accepted_false"
    _alternate_seed(accepted_false, diag, protocol)
    servo = json.loads((accepted_false / "event-009.json").read_text(encoding="utf-8"))
    servo["accepted"] = False
    _write(accepted_false / "event-009.json", servo)

    assert diag._audit_seed(sent_false, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(accepted_false, protocol)["status"] == diag._Result.FAILED.value


def test_audit_reproduced_requires_rejection_result_fields(tmp_path, diag):
    protocol = _protocol(tmp_path)
    wrong_error = tmp_path / "wrong_error"
    _positive_seed(wrong_error, diag, protocol)
    _set_result(wrong_error, error_type="_BudgetEnd")

    unavailable = tmp_path / "unavailable"
    _positive_seed(unavailable, diag, protocol)
    _set_result(unavailable, next_safety_decision_available=False)

    assert diag._audit_seed(wrong_error, protocol)["status"] == diag._Result.FAILED.value
    assert diag._audit_seed(unavailable, protocol)["status"] == diag._Result.FAILED.value


def test_audit_result_requires_cleanup_integrity_fields(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _positive_seed(root, diag, protocol)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("cleanup_error")
    _write(result_path, result)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


def test_audit_requires_rejected_observation(tmp_path, diag):
    protocol = _protocol(tmp_path)
    root = tmp_path / "seed"
    _positive_seed(root, diag, protocol)
    (root / "event-010.json").unlink()

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value
    assert audited["evidence_complete"] is False


def test_report_hashes_artifacts_and_forces_integrity_failure(tmp_path, monkeypatch, diag):
    output = tmp_path / "out"
    monkeypatch.setattr(diag, "OUTPUT", output)
    _write(output / "protocol.json", {"schema_version": diag.SCHEMA})
    protocol_sha = diag.file_sha256(output / "protocol.json")

    def verify(protocol):
        assert protocol["file_sha256"] == protocol_sha
        raise diag._Mismatch("integrity drift")

    monkeypatch.setattr(diag, "_verify", verify)

    diag._report()

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["protocol_sha256"] == protocol_sha
    assert report["integrity_error"] == "_Mismatch: integrity drift"
    assert report["report_hash_self_excluded"] is True
    assert report["policy_improved"] is False
    assert report["trials"]["7"]["status"] == diag._Result.FAILED.value
    assert report["artifact_sha256"] == {"protocol.json": protocol_sha}


def test_existing_attempt_preserves_bytes_and_rejects_retry(tmp_path, monkeypatch, diag):
    output = tmp_path / "out"
    root = output / "seed-7"
    root.mkdir(parents=True)
    attempt = root / "attempt.json"
    attempt.write_text("original", encoding="utf-8")
    monkeypatch.setattr(diag, "OUTPUT", output)

    with pytest.raises(Exception, match="already exists"):
        diag._run_seed(7)

    assert attempt.read_text(encoding="utf-8") == "original"


def test_seed_13_requires_seed_7_result_before_attempt(tmp_path, monkeypatch, diag):
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(diag, "OUTPUT", output)

    with pytest.raises(diag._Mismatch, match="seed 7 result required"):
        diag._run_seed(13)

    assert not (output / "seed-13" / "attempt.json").exists()


def test_run_seed_records_guarded_load_memory_error(tmp_path, monkeypatch, diag):
    output = tmp_path / "out"
    output.mkdir()
    _write(output / "protocol.json", {"schema_version": diag.SCHEMA})
    monkeypatch.setattr(diag, "OUTPUT", output)
    monkeypatch.setattr(diag, "_verify", lambda protocol: None)

    def guarded(protocol, loader):
        del protocol, loader
        raise MemoryError("oom")

    monkeypatch.setattr(diag, "_guarded_load", guarded)

    diag._run_seed(7)

    result = json.loads((output / "seed-7" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == diag._Result.FAILED.value
    assert result["base_send_calls"] == 0
    assert result["sent_count"] == 0
    assert result["error_type"] == "MemoryError"


def test_collision_consumes_base_budget(tmp_path, monkeypatch, diag):
    instance = diag._Probe.__new__(diag._Probe)
    instance._journal = diag._Journal(tmp_path)
    instance._expected = {"command": [0.0] * ACTION_DIM}
    instance._calls = 0
    instance._sent = 0
    instance._observations = 0
    instance._last_observation = None
    monkeypatch.setattr(instance, "_snapshot", lambda: '{"physics_step_count":0}')

    def send_action(self, action):
        del self, action
        raise MujocoCollisionError("target preflight", ())

    monkeypatch.setattr(diag.MujocoBiSOAdapter, "send_action", send_action)

    with pytest.raises(MujocoCollisionError, match="target preflight"):
        instance.send_action(_command(diag))
    with pytest.raises(diag._BudgetEnd):
        instance.send_action(_command(diag))

    assert instance.counts() == {"base_send_calls": 1, "sent_count": 0}


class _FakePolicy:
    required_history_steps = 1

    def predict(self, snapshot, *, now_s):
        del snapshot
        return ActionChunk(
            target_joint_position=np.zeros((10, ACTION_DIM), dtype=np.float32),
            dt_s=0.02,
            created_at_s=now_s,
        )


def _runtime_trial(tmp_path):
    config = ProjectConfig.from_mapping(
        {
            "runtime": {
                "backend": "mujoco",
                "camera_hz": 10.0,
                "policy_hz": 10.0,
                "servo_hz": 50.0,
                "context_seconds": 4.0,
                "prompt_min_seconds": 3.0,
                "prompt_max_seconds": 3.0,
                "action_horizon": 10,
                "actuation_enabled": True,
            },
            "safety": {
                "joint_lower": [0.0] * ACTION_DIM,
                "joint_upper": [100.0] * ACTION_DIM,
                "max_delta_per_servo_tick": [10.0] * ACTION_DIM,
            },
            "mujoco": {"model_path": "fixture.xml"},
        }
    )
    task = SimpleNamespace(
        object_body="task_block",
        initial_position=lambda seed: (0.0, 0.0, 0.5),
    )
    return SimpleNamespace(config=config, task=task, seed=7, output_dir=tmp_path)


def _runtime_prompt(diag):
    left, right = diag.PRIMARY_CAMERA_KEYS
    images = {
        left: np.zeros((2, 2, 3), dtype=np.uint8),
        right: np.zeros((2, 2, 3), dtype=np.uint8),
    }
    frames = tuple(
        diag.SensorimotorFrame(
            timestamp_s=index / 10.0,
            images=images,
            joint_position=np.zeros(ACTION_DIM, dtype=np.float32),
            executed_action=np.zeros(ACTION_DIM, dtype=np.float32),
        )
        for index in range(31)
    )
    return diag.PhysicalPrompt(frames=frames)


@pytest.mark.parametrize("mode", ["rejected", "budget", "collision", "exception"])
def test_real_runtime_with_fake_base_adapter(tmp_path, monkeypatch, diag, mode):
    trial = _runtime_trial(tmp_path)
    left, right = diag.PRIMARY_CAMERA_KEYS
    expected = {
        "command": [0.0] * ACTION_DIM,
        "joint_position": [0.0] * ACTION_DIM,
        "image_sha256": {
            left: hashlib.sha256(np.zeros((2, 2, 3), dtype=np.uint8).tobytes()).hexdigest(),
            right: hashlib.sha256(np.zeros((2, 2, 3), dtype=np.uint8).tobytes()).hexdigest(),
        },
        "model_identity": {"model": "fixture"},
        "object_body": "task_block",
        "object_position": [0.0, 0.0, 0.5],
    }
    journal = diag._Journal(tmp_path)
    probe = diag._Probe(trial, expected, journal)
    calls = {"send": 0, "disconnect": 0}

    probe._connected = False
    probe._physics_steps_per_servo_tick = 4
    monkeypatch.setattr(probe, "_snapshot", lambda: json.dumps({"physics_timestep_s": 0.005}))
    monkeypatch.setattr(diag.MujocoBiSOAdapter, "model_identity", lambda self: {"model": "fixture"})
    monkeypatch.setattr(diag.MujocoBiSOAdapter, "object_body_position", lambda self, body: (0.0, 0.0, 0.5))
    monkeypatch.setattr(diag.MujocoBiSOAdapter, "connect", lambda self, calibrate=True: setattr(self, "_connected", True))
    monkeypatch.setattr(diag.MujocoBiSOAdapter, "disconnect", lambda self: setattr(self, "_connected", False))

    def observe(self, *, timestamp_s):
        q = np.zeros(ACTION_DIM, dtype=np.float32)
        if mode == "rejected" and self._sent:
            q[diag.GRIPPER] = -2.0
        images = {
            left: np.zeros((2, 2, 3), dtype=np.uint8),
            right: np.zeros((2, 2, 3), dtype=np.uint8),
        }
        return diag.SensorimotorFrame(
            timestamp_s=timestamp_s,
            images=images,
            joint_position=q,
        )

    def send(self, action):
        calls["send"] += 1
        if mode == "collision":
            raise MujocoCollisionError("target preflight", ())
        if mode == "exception":
            raise RuntimeError("base failed")
        return np.zeros(ACTION_DIM, dtype=np.float32)

    monkeypatch.setattr(diag.MujocoBiSOAdapter, "get_observation", observe)
    monkeypatch.setattr(diag.MujocoBiSOAdapter, "send_action", send)
    monkeypatch.setattr(
        diag.MujocoBiSOAdapter,
        "disconnect",
        lambda self: (calls.__setitem__("disconnect", calls["disconnect"] + 1), setattr(self, "_connected", False)),
    )

    runtime = diag.SO101WAMRuntime(
        config=trial.config,
        prompt=_runtime_prompt(diag),
        robot=probe,
        policy=_FakePolicy(),
    )
    clock = diag._VirtualClock()

    expected_error = {
        "rejected": SafetyRejectedError,
        "budget": diag._BudgetEnd,
        "collision": MujocoCollisionError,
        "exception": RuntimeError,
    }[mode]
    with pytest.raises(expected_error):
        diag.run_managed_rollout(
            runtime,
            probe,
            policy_steps=1,
            calibrate_on_connect=False,
            clock=clock,
            sleeper=clock.sleep,
            rollout_observer=diag._Observer(journal),
        )

    assert probe.counts()["base_send_calls"] == 1
    assert calls == {"send": 1, "disconnect": 1}
    assert probe.is_connected is False
    servos = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(tmp_path.glob("event-*.json"))
        if json.loads(path.read_text(encoding="utf-8"))["kind"] == "servo"
    ]
    if mode == "rejected":
        assert len(servos) == 2
        assert servos[-1]["sent"] is False
        assert servos[-1]["transmitted_action"] is None
    if mode == "budget":
        requests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(tmp_path.glob("event-*.json"))
            if json.loads(path.read_text(encoding="utf-8"))["kind"] == "send_request"
        ]
        assert len(servos) == 1
        assert len(requests) == 2


def test_main_dispatches_exact_commands(monkeypatch, diag):
    calls = []

    class FixedRuntime:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    monkeypatch.setattr(diag, "_fixed_runtime", lambda: FixedRuntime())
    monkeypatch.setattr(diag, "_freeze", lambda: calls.append("freeze"))
    monkeypatch.setattr(diag, "_report", lambda: calls.append("report"))
    monkeypatch.setattr(diag, "_run_seed", lambda seed: calls.append(("seed", seed)))

    assert diag.main(["freeze"]) == 0
    assert diag.main(["7"]) == 0
    assert diag.main(["13"]) == 0
    assert diag.main(["report"]) == 0
    with pytest.raises(SystemExit, match="usage"):
        diag.main(["8"])

    assert calls == [
        "enter",
        "freeze",
        "exit",
        "enter",
        ("seed", 7),
        "exit",
        "enter",
        ("seed", 13),
        "exit",
        "enter",
        "report",
        "exit",
    ]


def test_main_uses_process_argv_by_default(monkeypatch, diag):
    calls = []

    class FixedRuntime:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    monkeypatch.setattr(sys, "argv", ["gripper_boundary_diagnostic.py", "report"])
    monkeypatch.setattr(diag, "_fixed_runtime", lambda: FixedRuntime())
    monkeypatch.setattr(diag, "_report", lambda: calls.append("report"))
    monkeypatch.setattr(diag, "_freeze", lambda: pytest.fail("unexpected freeze"))
    monkeypatch.setattr(diag, "_run_seed", lambda seed: pytest.fail(f"unexpected {seed}"))

    assert diag.main() == 0
    assert calls == ["enter", "report", "exit"]
