from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import so101_wam.sim_reference as sim_reference
from so101_wam.config import ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.constants import ACTION_DIM
from so101_wam.context import ContextSnapshot
from so101_wam.contracts import ActionChunk, SensorimotorFrame
from so101_wam.model import ActionRangeConstraint, CompactWAM
from so101_wam.policy import CompactWAMPolicy, PolicyError
from so101_wam.safety import SafetySupervisor
from so101_wam.sim_reference import SimTrialConfig, TrialKind, run_sim_trial


LOWER = tuple(float(index) for index in range(ACTION_DIM))
UPPER = tuple(float(index + 100) for index in range(ACTION_DIM))


def _bounded_model(mode: str = "affine_tanh") -> CompactWAM:
    assert mode in {item.value for item in ActionRangeConstraint}
    model = CompactWAM(
        latent_dim=16,
        transformer_heads=4,
        future_steps=2,
        action_horizon=1,
        action_history_steps=1,
    )
    model.action_range_constraint = ActionRangeConstraint(mode)
    model.register_buffer("action_lower", torch.tensor(LOWER, dtype=torch.float64))
    model.register_buffer("action_upper", torch.tensor(UPPER, dtype=torch.float64))
    return model


def _frame(timestamp_s: float, value: float = 1.0) -> SensorimotorFrame:
    image = np.full((8, 8, 3), int(value), dtype=np.uint8)
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={"left_wrist": image, "right_wrist": image},
        joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
        executed_action=np.full(ACTION_DIM, value, dtype=np.float32),
    )


def _task() -> sim_reference.SemanticTask:
    return sim_reference.SemanticTask(
        task_id="task-a",
        label="task a",
        policy_steps=1,
        seeds=(7,),
        object_body="task_block",
        initial_object_positions=((7, (0.1, 0.0, 0.4)),),
        target_object_position=(0.2, 0.0, 0.4),
        position_tolerance_m=0.03,
    )


def _config(*, lower: tuple[float, ...] = LOWER) -> ProjectConfig:
    return ProjectConfig(
        runtime=RuntimeConfig(
            backend="mujoco",
            camera_hz=50.0,
            policy_hz=1.0,
            servo_hz=50.0,
            action_horizon=1,
            actuation_enabled=True,
        ),
        safety=SafetyConfig(
            joint_lower=lower,
            joint_upper=UPPER,
            max_delta_per_servo_tick=(10.0,) * ACTION_DIM,
        ),
    )


class _Adapter:
    backend = "mujoco"
    actuation_enabled = True
    physics_steps_per_servo_tick = 4
    physics_step_count = 0

    def __init__(self, **kwargs: object) -> None:
        del kwargs

    def connect(self, *, calibrate: bool = True) -> None:
        del calibrate

    def disconnect(self) -> None:
        pass

    def contacts(self) -> tuple[object, ...]:
        return ()

    def model_identity(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "engine_version": "test",
            "compiled_model_sha256": "a" * 64,
            "compiled_model_bytes": 1,
        }

    def object_body_position(self, body_name: str) -> tuple[float, float, float]:
        del body_name
        return (0.2, 0.0, 0.4)

    def object_physical_profile(self, body_name: str) -> dict[str, object]:
        del body_name
        return {"schema_version": 1, "body": {}, "geom": {}}

    def object_physical_profile_sha256(self, body_name: str) -> str:
        del body_name
        return "b" * 64

    def get_observation(self, *, timestamp_s: float) -> SensorimotorFrame:
        return _frame(timestamp_s)

    def send_action(self, action: ActionChunk) -> np.ndarray:
        return np.asarray(action.target_joint_position[0], dtype=np.float32)


def _write_prompt(tmp_path: Path) -> Path:
    buffer = sim_reference.EpisodeBuffer(
        fps=1.0,
        task="prompt-task",
        episode_index=0,
    )
    buffer.append(_frame(0.0))
    buffer.append(_frame(1.0))
    buffer.append(_frame(2.0))
    buffer.append(_frame(3.0))
    return buffer.save(tmp_path / "prompt")[0]


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_bounded_needs_profile(mode: str) -> None:
    model = _bounded_model(mode)
    with pytest.raises(PolicyError, match="joint range profile"):
        CompactWAMPolicy(model, servo_hz=50.0)


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_bounded_accepts_match(mode: str) -> None:
    policy = CompactWAMPolicy(
        _bounded_model(mode),
        servo_hz=50.0,
        joint_lower=LOWER,
        joint_upper=UPPER,
    )

    assert policy.required_history_steps == 1


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (LOWER[:-1], UPPER, "12"),
        ((*LOWER[:-1], LOWER[-1] + 1.0), UPPER, "mismatch"),
        (tuple(reversed(LOWER)), UPPER, "mismatch"),
        ((*LOWER[:-1], float("nan")), UPPER, "finite"),
    ],
)
def test_bounded_rejects_profile(
    mode: str,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    message: str,
) -> None:
    model = _bounded_model(mode)
    with pytest.raises(PolicyError, match=message):
        CompactWAMPolicy(
            model,
            servo_hz=50.0,
            joint_lower=lower,
            joint_upper=upper,
        )


def test_rejects_unknown_range() -> None:
    model = CompactWAM(latent_dim=16, transformer_heads=4, action_history_steps=1)
    model.action_range_constraint = "bad"  # type: ignore[assignment]

    with pytest.raises(PolicyError, match="action_range_constraint"):
        CompactWAMPolicy(model, servo_hz=50.0)


def test_rejects_inverted_bounds() -> None:
    model = _bounded_model()
    model.action_lower.copy_(torch.tensor(UPPER, dtype=torch.float64))  # type: ignore[attr-defined]
    model.action_upper.copy_(torch.tensor(LOWER, dtype=torch.float64))  # type: ignore[attr-defined]

    with pytest.raises(PolicyError, match="lower bounds"):
        CompactWAMPolicy(
            model,
            servo_hz=50.0,
            joint_lower=UPPER,
            joint_upper=LOWER,
        )


def test_legacy_profile_optional() -> None:
    model = CompactWAM(latent_dim=16, transformer_heads=4, action_history_steps=1)

    CompactWAMPolicy(model, servo_hz=50.0)
    CompactWAMPolicy(model, servo_hz=50.0, joint_lower=LOWER, joint_upper=UPPER)


def test_sim_supplies_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, tuple[float, ...] | None] = {}
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    prompt_npz = _write_prompt(tmp_path)

    class Policy:
        required_history_steps = 1

        def __init__(
            self,
            model: object,
            *,
            servo_hz: float,
            device: str = "cpu",
            joint_lower: tuple[float, ...] | None = None,
            joint_upper: tuple[float, ...] | None = None,
        ) -> None:
            del model, servo_hz, device
            seen["joint_lower"] = joint_lower
            seen["joint_upper"] = joint_upper

        def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
            del snapshot
            return ActionChunk(
                np.zeros((1, ACTION_DIM), dtype=np.float32),
                dt_s=0.02,
                created_at_s=now_s,
            )

    monkeypatch.setattr(sim_reference, "MujocoBiSOAdapter", _Adapter)
    monkeypatch.setattr(
        sim_reference,
        "load_compact_wam_bundle",
        lambda *a, **k: SimpleNamespace(
            model=_bounded_model(),
            metadata={"checkpoint_id": "candidate"},
        ),
    )
    monkeypatch.setattr(sim_reference, "CompactWAMPolicy", Policy)

    def fake_rollout(runtime: object, robot: object, **kwargs: object) -> object:
        del robot
        kwargs["terminal_observer"](100.02)
        return SimpleNamespace(
            prompt_fingerprint=runtime.prompt.fingerprint,
            policy_steps=1,
            servo_steps=0,
            sent_actions=0,
            shadow_steps=0,
            elapsed_s=0.02,
            final_state="rollout_ready",
        )

    monkeypatch.setattr(sim_reference, "run_managed_rollout", fake_rollout)

    run_sim_trial(
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=tmp_path,
            kind=TrialKind.LEARNED,
            duration_s=1.0,
            episode_index=0,
            checkpoint_path=checkpoint,
            prompt_path=prompt_npz,
        )
    )

    assert seen == {"joint_lower": LOWER, "joint_upper": UPPER}


def test_sim_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    prompt_npz = _write_prompt(tmp_path)

    def unexpected_rollout(*args: object, **kwargs: object) -> object:
        raise AssertionError("profile mismatch must stop before rollout")

    monkeypatch.setattr(sim_reference, "MujocoBiSOAdapter", _Adapter)
    monkeypatch.setattr(
        sim_reference,
        "load_compact_wam_bundle",
        lambda *a, **k: SimpleNamespace(
            model=_bounded_model(),
            metadata={"checkpoint_id": "candidate"},
        ),
    )
    monkeypatch.setattr(sim_reference, "run_managed_rollout", unexpected_rollout)

    with pytest.raises(PolicyError, match="mismatch"):
        run_sim_trial(
            SimTrialConfig(
                config=_config(lower=(*LOWER[:-1], LOWER[-1] + 1.0)),
                task=_task(),
                seed=7,
                output_dir=tmp_path,
                kind=TrialKind.LEARNED,
                duration_s=1.0,
                episode_index=0,
                checkpoint_path=checkpoint,
                prompt_path=prompt_npz,
            )
        )


def test_safety_still_applies() -> None:
    supervisor = SafetySupervisor(_config().safety)
    action = ActionChunk(
        np.full((1, ACTION_DIM), UPPER[0] + 1.0, dtype=np.float32),
        dt_s=0.02,
        created_at_s=1.0,
    )

    decision = supervisor.evaluate(_frame(1.0), action, now_s=1.0)

    assert decision.accepted is False
    assert decision.hold is True
    assert any("action[0]_joint_limit" in reason for reason in decision.reasons)
