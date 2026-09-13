from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import so101_wam.mujoco_semantic_benchmark as semantic_module
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import ActionChunk, SensorimotorFrame
from so101_wam.deployment import canonical_json_sha256
from so101_wam.mujoco_semantic_benchmark import (
    SemanticTask,
    run_checkpoint_task_trial,
    run_task_spec_checkpoint_task_trial,
)
from so101_wam.mujoco_identity import MujocoModelIdentity
from so101_wam.rollout import (
    ExecutabilityTrace,
    RolloutError,
    validate_mujoco_scored_trace,
)
from so101_wam.safety import SafetySupervisor
from so101_wam.task_specs import LanguagePrompt, TaskSpecProvenance


OBSERVATION_GRIPPER_LOWER = -0.000156261459173829
OBJECT_PROFILE = {
    "schema_version": 1,
    "body": {
        "mass": 0.08,
        "inertia": [0.000066, 0.000066, 0.000066],
        "inertial_pos": [0.0, 0.0, 0.0],
        "inertial_quat": [1.0, 0.0, 0.0, 0.0],
    },
    "geom": {
        "type": "box",
        "size": [0.025, 0.025, 0.025],
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "friction": [1.0, 0.005, 0.0001],
        "margin": 0.0,
        "gap": 0.0,
        "solref": [0.02, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    },
}
OBJECT_PROFILE_SHA256 = canonical_json_sha256(OBJECT_PROFILE)
MODEL_IDENTITY = MujocoModelIdentity(
    engine_version="3.12.0",
    compiled_model_sha256="f" * 64,
    compiled_model_bytes=128,
).to_payload()


def _task() -> SemanticTask:
    return SemanticTask(
        task_id="heldout_block_lift",
        label="held-out block lift",
        policy_steps=1,
        seeds=(7,),
        object_body="task_block",
        initial_object_positions=((7, (0.32, -0.02, 0.475)),),
        target_object_position=(0.34, 0.0, 0.58),
        position_tolerance_m=0.03,
    )


def _action(values: np.ndarray, *, created_at_s: float = 0.0) -> ActionChunk:
    return ActionChunk(
        target_joint_position=values.reshape(1, ACTION_DIM),
        dt_s=0.02,
        created_at_s=created_at_s,
    )


def _frame(joints: np.ndarray, *, timestamp_s: float = 0.0) -> SensorimotorFrame:
    images = {
        key: np.zeros((2, 2, 3), dtype=np.uint8) for key in PRIMARY_CAMERA_KEYS
    }
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images=images,
        joint_position=joints,
        executed_action=np.zeros(ACTION_DIM, dtype=np.float32),
    )


def _record_scored_trace(
    trace: ExecutabilityTrace,
    frame: SensorimotorFrame,
    decision_action: ActionChunk,
) -> dict[str, object]:
    policy_action = _action(np.zeros(ACTION_DIM, dtype=np.float32))
    trace.on_policy_step(
        policy_index=0,
        step=SimpleNamespace(action=policy_action),
    )
    trace.on_servo_step(
        policy_index=0,
        servo_index=0,
        step=SimpleNamespace(
            action=decision_action,
            observation=frame,
            executed_action=decision_action.target_joint_position[0],
            sent=True,
            shadow=False,
            safety=SimpleNamespace(accepted=True, clipped=False, reasons=()),
        ),
    )
    trace.on_contact_sample(
        policy_index=0,
        servo_index=0,
        phase="servo",
        contacts=(),
    )
    return trace.snapshot()


def test_trace_accepts_valid_negative_observation_margin() -> None:
    config = ProjectConfig.load(Path("configs/mujoco.toml"))
    observed = np.zeros(ACTION_DIM, dtype=np.float32)
    observed[5] = np.float32(OBSERVATION_GRIPPER_LOWER)
    frame = _frame(observed)
    command = _action(np.zeros(ACTION_DIM, dtype=np.float32))
    decision = SafetySupervisor(config.safety).evaluate(frame, command, now_s=0.0)
    assert decision.accepted is True

    trace = ExecutabilityTrace(
        "negative-observation:seed:7",
        joint_lower=config.safety.joint_lower,
        joint_upper=config.safety.joint_upper,
        observation_joint_lower=config.safety.observation_joint_lower,
        observation_joint_upper=config.safety.observation_joint_upper,
    )
    payload = _record_scored_trace(trace, frame, decision.action)

    margins = payload["joint_limit_margins"]
    assert isinstance(margins, dict)
    assert margins["measured_joint_min"] == pytest.approx(0.0)
    assert margins["servo_safe_target_min"] >= 0.0
    assert margins["executed_action_min"] >= 0.0
    validate_mujoco_scored_trace(payload, policy_steps=1)


def test_negative_action_margin_still_fails_scored_validation() -> None:
    trace = ExecutabilityTrace(
        "negative-action:seed:7",
        joint_lower=(0.0,) * ACTION_DIM,
        joint_upper=(100.0,) * ACTION_DIM,
        observation_joint_lower=(-1.0,) * ACTION_DIM,
        observation_joint_upper=(100.0,) * ACTION_DIM,
    )
    frame = _frame(np.zeros(ACTION_DIM, dtype=np.float32))
    unsafe_action = _action(np.full(ACTION_DIM, -0.5, dtype=np.float32))
    payload = _record_scored_trace(trace, frame, unsafe_action)

    margins = payload["joint_limit_margins"]
    assert isinstance(margins, dict)
    assert margins["servo_safe_target_min"] < 0.0
    assert margins["executed_action_min"] < 0.0
    with pytest.raises(RolloutError, match="joint-limit margin"):
        validate_mujoco_scored_trace(payload, policy_steps=1)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ((-1.0,) * ACTION_DIM, None),
        (None, (1.0,) * ACTION_DIM),
        ((-1.0,) * (ACTION_DIM - 1), (1.0,) * ACTION_DIM),
        ((-1.0,) * ACTION_DIM, (1.0,) * (ACTION_DIM - 1)),
        ((-1.0,) * (ACTION_DIM - 1) + (float("nan"),), (1.0,) * ACTION_DIM),
        ((0.0,) * ACTION_DIM, (0.0,) * ACTION_DIM),
    ],
)
def test_trace_rejects_malformed_observation_bounds(
    lower: tuple[float, ...] | None,
    upper: tuple[float, ...] | None,
) -> None:
    with pytest.raises(RolloutError, match="observation"):
        ExecutabilityTrace(
            "malformed-observation:seed:7",
            joint_lower=(0.0,) * ACTION_DIM,
            joint_upper=(1.0,) * ACTION_DIM,
            observation_joint_lower=lower,
            observation_joint_upper=upper,
        )


def test_trace_observation_bounds_default_to_command_bounds() -> None:
    trace = ExecutabilityTrace(
        "legacy-observation:seed:7",
        joint_lower=(0.0,) * ACTION_DIM,
        joint_upper=(1.0,) * ACTION_DIM,
    )
    frame = _frame(np.full(ACTION_DIM, -0.1, dtype=np.float32))
    action = _action(np.zeros(ACTION_DIM, dtype=np.float32))
    payload = _record_scored_trace(trace, frame, action)

    margins = payload["joint_limit_margins"]
    assert isinstance(margins, dict)
    assert margins["measured_joint_min"] < 0.0


def test_trace_preserves_command_margin_precision() -> None:
    lower = (-109.9998,) * ACTION_DIM
    upper = (109.9998,) * ACTION_DIM
    target = np.full(ACTION_DIM, 0.1234567, dtype=np.float32)
    expected_policy = _expected_margin(
        np.zeros(ACTION_DIM, dtype=np.float32),
        lower=lower,
        upper=upper,
    )
    expected_target = _expected_margin(target, lower=lower, upper=upper)

    legacy = ExecutabilityTrace(
        "legacy-command-precision:seed:7",
        joint_lower=lower,
        joint_upper=upper,
    )
    explicit = ExecutabilityTrace(
        "explicit-command-precision:seed:7",
        joint_lower=lower,
        joint_upper=upper,
        observation_joint_lower=(-110.0,) * ACTION_DIM,
        observation_joint_upper=(110.0,) * ACTION_DIM,
    )

    for trace in (legacy, explicit):
        payload = _record_scored_trace(trace, _frame(target), _action(target))
        margins = payload["joint_limit_margins"]
        assert isinstance(margins, dict)
        assert margins["decoded_policy_target_min"] == expected_policy
        assert margins["servo_safe_target_min"] == expected_target
        assert margins["executed_action_min"] == expected_target


def test_checkpoint_trial_passes_observation_bounds_to_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(Path("configs/mujoco.toml"))
    task = _task()
    seen_margin: float | None = None

    def fake_session(
        session_config: ProjectConfig,
        *,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        manifest_path: str | Path | None,
        policy_steps: int,
        device: str,
        semantic_object_body: str,
        initial_object_position: tuple[float, float, float],
        rollout_observer: ExecutabilityTrace,
    ) -> dict[str, object]:
        nonlocal seen_margin
        del session_config, checkpoint_path, prompt_path, manifest_path, device
        del policy_steps, initial_object_position
        frame = _frame(np.array(config.safety.observation_joint_lower, dtype=np.float32))
        action = _action(np.zeros(ACTION_DIM, dtype=np.float32))
        payload = _record_scored_trace(rollout_observer, frame, action)
        margins = payload["joint_limit_margins"]
        assert isinstance(margins, dict)
        seen_margin = float(margins["measured_joint_min"])
        return _report(semantic_object_body)

    monkeypatch.setattr(semantic_module, "run_mujoco_checkpoint_session", fake_session)

    run_checkpoint_task_trial(
        config,
        task,
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        prompt_path=tmp_path / "prompt.npz",
    )

    assert seen_margin == pytest.approx(0.0)


def test_task_spec_checkpoint_trial_passes_observation_bounds_to_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(Path("configs/mujoco.toml"))
    task = _task()
    task_spec = LanguagePrompt(
        text="lift the block",
        provenance=TaskSpecProvenance("test"),
    )
    seen_margin: float | None = None

    def fake_session(
        session_config: ProjectConfig,
        *,
        checkpoint_path: str | Path,
        task_spec: LanguagePrompt,
        policy_steps: int,
        device: str,
        semantic_object_body: str,
        initial_object_position: tuple[float, float, float],
        rollout_observer: ExecutabilityTrace,
    ) -> dict[str, object]:
        nonlocal seen_margin
        fingerprint = task_spec.fingerprint
        del session_config, checkpoint_path, policy_steps, device
        del initial_object_position
        frame = _frame(np.array(config.safety.observation_joint_lower, dtype=np.float32))
        action = _action(np.zeros(ACTION_DIM, dtype=np.float32))
        payload = _record_scored_trace(rollout_observer, frame, action)
        margins = payload["joint_limit_margins"]
        assert isinstance(margins, dict)
        seen_margin = float(margins["measured_joint_min"])
        report = _report(semantic_object_body)
        return {
            **report,
            "runtime_placeholder_prompt_used_by_model": False,
            "task_spec_fingerprint": fingerprint,
        }

    monkeypatch.setattr(
        semantic_module,
        "run_mujoco_task_spec_checkpoint_session",
        fake_session,
    )

    run_task_spec_checkpoint_task_trial(
        config,
        task,
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        task_spec=task_spec,
    )

    assert seen_margin == pytest.approx(0.0)


def _report(object_body: str) -> dict[str, object]:
    return {
        "rollout": {"policy_steps": 1, "final_state": "rollout_ready"},
        "object_body": object_body,
        "object_physical_profile": OBJECT_PROFILE,
        "object_physical_profile_sha256": OBJECT_PROFILE_SHA256,
        "mujoco_model_identity": MODEL_IDENTITY,
        "terminal_joint_position": [0.0] * ACTION_DIM,
        "terminal_object_position": [0.34, 0.0, 0.58],
    }


def _expected_margin(
    value: np.ndarray,
    *,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
) -> float:
    array = np.asarray(value, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    span = upper_array - lower_array
    margin = np.minimum(array - lower_array, upper_array - array)
    return float(np.min(margin / span))
