import json
import importlib.util
import sys
from pathlib import Path

import pytest
import test_gripper_diagnostic as base


@pytest.fixture
def diag():
    path = Path(__file__).parents[1] / "scripts" / "gripper_boundary_diagnostic.py"
    spec = importlib.util.spec_from_file_location("gripper_diag_boundary_test", path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _event(root, index):
    return json.loads((root / f"event-{index:03d}.json").read_text(encoding="utf-8"))


def _replace(root, index, payload):
    base._write(root / f"event-{index:03d}.json", payload)


def test_budget_rejects_empty_observation_binding(tmp_path, diag):
    protocol = base._protocol(tmp_path)
    root = tmp_path / "seed"
    base._budget_seed(root, diag, protocol)

    request = _event(root, 11)
    request["last_observation"] = {}
    _replace(root, 11, request)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


def test_budget_rejects_stale_observation_binding(tmp_path, diag):
    protocol = base._protocol(tmp_path)
    root = tmp_path / "seed"
    base._budget_seed(root, diag, protocol)

    request = _event(root, 11)
    request["last_observation"] = _event(root, 4)
    _replace(root, 11, request)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


def test_budget_requires_post_send_observation(tmp_path, diag):
    protocol = base._protocol(tmp_path)
    root = tmp_path / "seed"
    base._budget_seed(root, diag, protocol)

    (root / "event-010.json").unlink()
    request = _event(root, 11)
    request["last_observation"] = _event(root, 4)
    _replace(root, 11, request)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


def test_alternate_rejects_contradictory_final_observation(tmp_path, diag):
    protocol = base._protocol(tmp_path)
    root = tmp_path / "seed"
    base._alternate_seed(root, diag, protocol)

    observation = _event(root, 10)
    observation["joint_position"] = [0.0] * len(observation["joint_position"])
    observation["canonical_float32_observed"] = 0.0
    _replace(root, 10, observation)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


@pytest.mark.parametrize("timestamp_s", [-0.01, 0.0])
def test_budget_bad_time(tmp_path, diag, timestamp_s):
    protocol = base._protocol(tmp_path)
    root = tmp_path / "seed"
    base._budget_seed(root, diag, protocol)

    observation = _event(root, 10)
    observation["timestamp_s"] = timestamp_s
    _replace(root, 10, observation)
    request = _event(root, 11)
    request["last_observation"] = observation
    _replace(root, 11, request)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value


@pytest.mark.parametrize("timestamp_s", [-0.01, 0.0])
def test_alt_bad_time(tmp_path, diag, timestamp_s):
    protocol = base._protocol(tmp_path)
    root = tmp_path / "seed"
    base._alternate_seed(root, diag, protocol)

    observation = _event(root, 10)
    observation["timestamp_s"] = timestamp_s
    _replace(root, 10, observation)
    servo = _event(root, 11)
    servo["timestamp_s"] = timestamp_s
    _replace(root, 11, servo)

    audited = diag._audit_seed(root, protocol)

    assert audited["status"] == diag._Result.FAILED.value
