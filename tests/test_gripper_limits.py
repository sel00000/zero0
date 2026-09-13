"""Regression checks for distinct measured and commanded gripper limits."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from so101_wam.adapters.mujoco import (
    DEFAULT_MJCF_PATH,
    MujocoAdapterError,
    MujocoBiSOAdapter,
)
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import ActionChunk, SensorimotorFrame
from so101_wam.safety import SafetySupervisor
from so101_wam.contracts import PhysicalPrompt
from so101_wam.rollout import run_managed_rollout
from so101_wam.runtime import RuntimeState, SO101WAMRuntime
from test_rollout import FakeClock, _frame


ROOT = Path(__file__).resolve().parents[1]
GRIPPERS = (5, 11)
RECORDED_QPOS = (-0.17453006585715458, -0.174530065851183)


@pytest.fixture
def model_bindings():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF_PATH))
    bindings, coordinates = MujocoBiSOAdapter._bind_model(mujoco, model)
    return mujoco, model, bindings, coordinates


@pytest.fixture
def coordinates(model_bindings):
    return model_bindings[-1]


@pytest.fixture
def adapter(model_bindings, monkeypatch):
    mujoco, model, bindings, coordinates = model_bindings
    config = ProjectConfig.load(ROOT / "configs" / "mujoco.toml")
    adapter = MujocoBiSOAdapter(config=replace(config.mujoco, forbid_collisions=False))
    # Static model state only: no connection, policy or physics steps.
    adapter._model = model
    adapter._bindings = bindings
    adapter._coordinates = coordinates
    adapter._data = mujoco.MjData(model)
    adapter._connected = True
    adapter._data.qpos[bindings.qpos_addresses] = coordinates.to_native(
        np.zeros(ACTION_DIM)
    )
    monkeypatch.setattr(
        adapter, "_render", lambda _: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    return adapter


def _decision(config, observed, target):
    frame = SensorimotorFrame(
        timestamp_s=1.0,
        images={
            key: np.zeros((8, 8, 3), dtype=np.uint8)
            for key in ("left_wrist", "right_wrist")
        },
        joint_position=observed,
    )
    action = ActionChunk(
        target_joint_position=target.reshape(1, ACTION_DIM),
        dt_s=0.02,
        created_at_s=1.0,
    )
    return SafetySupervisor(config.safety).evaluate(frame, action, now_s=1.0)


@pytest.mark.parametrize("filename", ["mujoco.toml", "mujoco_robot_free.toml"])
@pytest.mark.parametrize("index", GRIPPERS)
@pytest.mark.parametrize("qpos", RECORDED_QPOS)
def test_recorded_valid_q_is_safe(coordinates, filename, index, qpos):
    config = ProjectConfig.load(ROOT / "configs" / filename)
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    native = coordinates.to_native(target)
    native[index] = qpos
    observed = coordinates.from_native(native)

    # Preserve the measured value; do not hide it with coordinate clipping.
    assert observed[index] < 0.0
    decision = _decision(config, observed, target)
    assert decision.accepted, decision.reasons
    np.testing.assert_array_equal(decision.action.target_joint_position[0], target)


@pytest.mark.parametrize("index", GRIPPERS)
def test_negative_command_stays_unsafe(coordinates, index):
    config = ProjectConfig.load(ROOT / "configs" / "mujoco.toml")
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    native = coordinates.to_native(target)
    native[index] = RECORDED_QPOS[0]
    observed = coordinates.from_native(native)
    target[index] = observed[index]

    decision = _decision(config, observed, target)
    assert not decision.accepted
    assert any(
        reason.startswith("action[0]_joint_limit:") for reason in decision.reasons
    )


@pytest.mark.parametrize("index", range(ACTION_DIM))
@pytest.mark.parametrize("endpoint", [0, 1])
def test_raw_joint_breach_is_blocked(model_bindings, adapter, index, endpoint):
    _, model, bindings, _ = model_bindings
    limit = model.jnt_range[bindings.joint_ids[index], endpoint]
    direction = -np.inf if endpoint == 0 else np.inf
    adapter._data.qpos[bindings.qpos_addresses[index]] = np.nextafter(limit, direction)

    with pytest.raises(MujocoAdapterError, match="physical joint limits"):
        adapter.get_observation(timestamp_s=1.0)


@pytest.mark.parametrize("index", GRIPPERS)
@pytest.mark.parametrize("endpoint", [0, 1])
def test_exact_joint_limit_is_safe(model_bindings, adapter, index, endpoint):
    _, model, bindings, _ = model_bindings
    config = ProjectConfig.load(ROOT / "configs" / "mujoco.toml")
    adapter._data.qpos[bindings.qpos_addresses[index]] = model.jnt_range[
        bindings.joint_ids[index], endpoint
    ]
    frame = adapter.get_observation(timestamp_s=1.0)
    target = np.clip(
        frame.joint_position, config.safety.joint_lower, config.safety.joint_upper
    )
    assert _decision(config, frame.joint_position, target).accepted


@pytest.mark.parametrize("value", [0.0, 25.0, 50.0, 100.0])
def test_command_mapping_is_unchanged(coordinates, value):
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    target[list(GRIPPERS)] = value
    native = coordinates.to_native(target)
    expected = -0.17453 + value * (1.7453292 - (-0.17453)) / 100.0
    np.testing.assert_array_equal(native[list(GRIPPERS)], [expected, expected])
    np.testing.assert_array_equal(coordinates.from_native(native), target)


@pytest.mark.parametrize("index", GRIPPERS)
@pytest.mark.parametrize("filename", ["mujoco.toml", "mujoco_robot_free.toml"])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_config_tracks_joint_limits(model_bindings, index, filename, endpoint):
    _, model, bindings, coordinates = model_bindings
    config = ProjectConfig.load(ROOT / "configs" / filename)
    native = coordinates.to_native(np.zeros(ACTION_DIM))
    native[index] = model.jnt_range[bindings.joint_ids[index], endpoint]
    observed = coordinates.from_native(native)
    limits = (
        config.safety.observation_joint_lower,
        config.safety.observation_joint_upper,
    )
    assert limits[endpoint] is not None
    assert np.float32(limits[endpoint][index]) == observed[index]
    target = np.clip(observed, config.safety.joint_lower, config.safety.joint_upper)
    assert _decision(config, observed, target).accepted


def test_unreachable_command_is_blocked(coordinates):
    config = ProjectConfig.load(ROOT / "configs" / "mujoco.toml")
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    native = coordinates.to_native(target)
    native[GRIPPERS[-1]] = RECORDED_QPOS[0]
    observed = coordinates.from_native(native)
    deltas = np.full(ACTION_DIM, 1e-8)
    config = replace(
        config, safety=replace(config.safety, max_delta_per_servo_tick=tuple(deltas))
    )

    decision = _decision(config, observed, target)
    assert not decision.accepted
    assert "action_unreachable_from_observation" in decision.reasons


def test_legacy_observation_is_strict(coordinates):
    config = ProjectConfig.load(ROOT / "configs" / "mujoco.toml")
    config = replace(
        config,
        safety=replace(
            config.safety, observation_joint_lower=None, observation_joint_upper=None
        ),
    )
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    native = coordinates.to_native(target)
    native[GRIPPERS[-1]] = RECORDED_QPOS[0]
    observed = coordinates.from_native(native)

    decision = _decision(config, observed, target)
    assert not decision.accepted
    assert decision.reasons == ("observation_joint_limit:right_gripper.pos:below",)


@pytest.mark.parametrize("observed_value", [-0.5, 100.5])
@pytest.mark.parametrize("max_delta", [0.25, 1.0])
def test_separate_envelope_rate_limit(observed_value, max_delta):
    config = ProjectConfig.load(ROOT / "configs" / "mujoco.toml")
    lower = list(config.safety.joint_lower)
    upper = list(config.safety.joint_upper)
    lower[GRIPPERS[-1]] = -1.0
    upper[GRIPPERS[-1]] = 101.0
    config = replace(
        config,
        safety=replace(
            config.safety,
            observation_joint_lower=tuple(lower),
            observation_joint_upper=tuple(upper),
            max_delta_per_servo_tick=(max_delta,) * ACTION_DIM,
        ),
    )
    observed = np.zeros(ACTION_DIM, dtype=np.float32)
    observed[GRIPPERS[-1]] = observed_value
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    target[GRIPPERS[-1]] = 100.0 if observed_value > 100.0 else 0.0

    decision = _decision(config, observed, target)
    assert decision.accepted is (max_delta >= 0.5)
    if decision.accepted:
        output = decision.action.target_joint_position[0]
        assert np.all(output >= config.safety.joint_lower)
        assert np.all(output <= config.safety.joint_upper)
        assert np.all(np.abs(output - observed) <= max_delta)
    else:
        assert decision.reasons == ("action_unreachable_from_observation",)


@pytest.mark.parametrize("capture_index", [1, 3])
def test_native_breach_halts_rollout(
    model_bindings, adapter, monkeypatch, capture_index
):
    _, model, bindings, _ = model_bindings
    config = ProjectConfig.load(ROOT / "configs" / "mujoco.toml")
    adapter._connected = False
    adapter.actuation_enabled = True
    monkeypatch.setattr(
        adapter,
        "connect",
        Mock(side_effect=lambda **_: setattr(adapter, "_connected", True)),
    )
    disconnect = Mock(wraps=adapter.disconnect)
    send = Mock(side_effect=AssertionError("unexpected send"))
    monkeypatch.setattr(adapter, "disconnect", disconnect)
    monkeypatch.setattr(adapter, "send_action", send)
    get_observation = adapter.get_observation
    captures = 0

    def capture(*, timestamp_s):
        nonlocal captures
        captures += 1
        if captures == capture_index:
            adapter._data.qpos[bindings.qpos_addresses[-1]] = np.nextafter(
                model.jnt_range[bindings.joint_ids[-1], 0], -np.inf
            )
        return get_observation(timestamp_s=timestamp_s)

    monkeypatch.setattr(adapter, "get_observation", capture)
    clock = FakeClock()
    policy = SimpleNamespace(
        required_history_steps=1,
        predict=lambda _, now_s: ActionChunk(
            target_joint_position=np.zeros((config.runtime.action_horizon, ACTION_DIM)),
            dt_s=0.02,
            created_at_s=now_s,
        ),
    )
    runtime = SO101WAMRuntime(
        config=config,
        robot=adapter,
        policy=policy,
        prompt=PhysicalPrompt((_frame(0.0, 0.0), _frame(3.0, 0.0))),
    )

    with pytest.raises(MujocoAdapterError, match="physical joint limits"):
        run_managed_rollout(
            runtime, adapter, policy_steps=1, clock=clock, sleeper=clock.sleep
        )

    send.assert_not_called()
    disconnect.assert_called_once()
    assert not adapter.is_connected
    assert runtime.state is RuntimeState.HALT
    assert runtime.safety.fault_latch.latched
