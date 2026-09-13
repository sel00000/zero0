from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import so101_wam.sim_reference as sim_reference
from so101_wam.config import ProjectConfig, RuntimeConfig
from so101_wam.constants import ACTION_DIM
from so101_wam.context import ContextSnapshot
from so101_wam.contracts import ActionChunk, SensorimotorFrame
from so101_wam.mujoco_semantic_benchmark import SemanticTask
from so101_wam.rollout import SafetyRejectedError
from so101_wam.sim_reference import ReferenceError, SimTrialConfig, TrialKind, _WaypointPolicy, run_sim_trial
from so101_wam.training_data import load_episode_records


def _image(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _frame(timestamp_s: float, value: float, *, action: float | None = None) -> SensorimotorFrame:
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={"left_wrist": _image(int(value)), "right_wrist": _image(int(value) + 1)},
        joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
        executed_action=(
            None
            if action is None
            else np.full(ACTION_DIM, action, dtype=np.float32)
        ),
    )


def _task(*, target: tuple[float, float, float] = (0.2, 0.0, 0.4)) -> SemanticTask:
    return SemanticTask(
        task_id="task-a",
        label="task a",
        policy_steps=1,
        seeds=(7,),
        object_body="task_block",
        initial_object_positions=((7, (0.1, 0.0, 0.4)),),
        target_object_position=target,
        position_tolerance_m=0.03,
    )


def _config() -> ProjectConfig:
    return ProjectConfig(
        runtime=RuntimeConfig(
            backend="mujoco",
            camera_hz=50.0,
            policy_hz=1.0,
            servo_hz=50.0,
            action_horizon=1,
            actuation_enabled=True,
        )
    )


class _ContactAdapter:
    backend = "mujoco"
    actuation_enabled = True
    physics_steps_per_servo_tick = 4
    physics_step_count = 0

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.is_connected = False
        self.sent: list[np.ndarray] = []

    def connect(self, *, calibrate: bool = True) -> None:
        del calibrate
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

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
        assert body_name == "task_block"
        return (0.2, 0.0, 0.4)

    def object_physical_profile(self, body_name: str) -> dict[str, object]:
        assert body_name == "task_block"
        return {"schema_version": 1, "body": {}, "geom": {}}

    def object_physical_profile_sha256(self, body_name: str) -> str:
        assert body_name == "task_block"
        return "b" * 64

    def get_observation(self, *, timestamp_s: float) -> SensorimotorFrame:
        return _frame(timestamp_s, 1.0)

    def send_action(self, action: ActionChunk) -> np.ndarray:
        result = np.full(ACTION_DIM, 2.0, dtype=np.float32)
        self.sent.append(result)
        return result


def test_waypoints_match_servo_time() -> None:
    targets = np.zeros((3, 12), dtype=np.float32)
    targets[1, 0] = 10.0
    targets[2, 0] = 20.0
    policy = _WaypointPolicy((0.0, 1.0, 2.0), targets, horizon=5, servo_hz=50.0)
    snapshot = cast(ContextSnapshot, None)
    first = policy.predict(snapshot, now_s=100.0)
    np.testing.assert_allclose(first.target_joint_position[:, 0], np.arange(5) * 0.2)
    later = policy.predict(snapshot, now_s=101.0)
    np.testing.assert_allclose(later.target_joint_position[:, 0], 10 + np.arange(5) * 0.2)
    final = policy.predict(snapshot, now_s=103.0)
    np.testing.assert_array_equal(final.target_joint_position[:, 0], np.full(5, 20.0))
    assert first.created_at_s == 100.0
    assert first.dt_s == 0.02


@pytest.mark.parametrize("times", [(1.0, 2.0), (0.0, 0.0), (0.0, float("nan"))])
def test_invalid_waypoint_times(times: tuple[float, ...]) -> None:
    with pytest.raises(ReferenceError):
        _WaypointPolicy(times, np.zeros((2, 12)), horizon=5, servo_hz=50.0)


def test_waypoints_copy_input() -> None:
    targets = np.zeros((2, 12), dtype=np.float32)
    policy = _WaypointPolicy((0.0, 1.0), targets, horizon=5, servo_hz=50.0)
    targets[:] = 100.0
    result = policy.predict(cast(ContextSnapshot, None), now_s=0.0)
    np.testing.assert_array_equal(result.target_joint_position, np.zeros((5, 12)))


@pytest.mark.parametrize(
    ("bad_config", "message"),
    [
        (replace(_config(), runtime=replace(_config().runtime, backend="fake")), "runtime.backend"),
        (
            replace(_config(), runtime=replace(_config().runtime, actuation_enabled=False)),
            "actuation_enabled",
        ),
        (
            replace(_config(), mujoco=replace(_config().mujoco, forbid_collisions=False)),
            "forbid_collisions",
        ),
    ],
)
def test_trial_rejects_unsafe_config_before_write(
    tmp_path: Path,
    bad_config: ProjectConfig,
    message: str,
) -> None:

    with pytest.raises(ReferenceError, match=message):
        run_sim_trial(
            SimTrialConfig(
                config=bad_config,
                task=_task(),
                seed=7,
                output_dir=tmp_path / "out",
                kind=TrialKind.HOLD,
                duration_s=0.02,
                episode_index=0,
            )
        )

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "trial",
    [
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=Path("out"),
            kind=TrialKind.REFERENCE,
            duration_s=0.02,
            episode_index=0,
            times_s=(0.0, 0.02),
            targets=np.zeros((2, ACTION_DIM), dtype=np.float32),
            checkpoint_path=Path("candidate.pt"),
        ),
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=Path("out"),
            kind=TrialKind.REFERENCE,
            duration_s=0.02,
            episode_index=0,
            checkpoint_path=Path("candidate.pt"),
            prompt_path=Path("prompt.npz"),
        ),
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=Path("out"),
            kind=TrialKind.HOLD,
            duration_s=0.02,
            episode_index=0,
            times_s=(0.0, 0.02),
            targets=np.zeros((2, ACTION_DIM), dtype=np.float32),
        ),
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=Path("out"),
            kind=TrialKind.HOLD,
            duration_s=0.02,
            episode_index=0,
            checkpoint_path=Path("candidate.pt"),
            prompt_path=Path("prompt.npz"),
        ),
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=Path("out"),
            kind=TrialKind.LEARNED,
            duration_s=0.02,
            episode_index=0,
            checkpoint_path=Path("candidate.pt"),
            prompt_path=Path("prompt.npz"),
            times_s=(0.0, 0.02),
            targets=np.zeros((2, ACTION_DIM), dtype=np.float32),
        ),
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=Path("out"),
            kind=TrialKind.LEARNED,
            duration_s=0.02,
            episode_index=0,
            checkpoint_path=Path("candidate.pt"),
        ),
    ],
)
def test_trial_rejects_mismatched_kind_inputs_before_write(
    tmp_path: Path,
    trial: SimTrialConfig,
) -> None:
    trial = replace(trial, output_dir=tmp_path / "out")

    with pytest.raises(ReferenceError, match="kind inputs"):
        run_sim_trial(trial)

    assert not (tmp_path / "out").exists()


def test_trial_records_sent_precommand_observation_and_executed_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sim_reference, "MujocoBiSOAdapter", _ContactAdapter)

    def fake_rollout(runtime: object, robot: object, **kwargs: object) -> object:
        observer = kwargs["rollout_observer"]
        terminal = kwargs["terminal_observer"]
        step = SimpleNamespace(
            observation=_frame(100.0, 1.0),
            executed_action=np.full(ACTION_DIM, 2.0, dtype=np.float32),
            sent=True,
            shadow=False,
            safety=SimpleNamespace(accepted=True, clipped=False, reasons=()),
            action=ActionChunk(
                np.full((1, ACTION_DIM), 3.0, dtype=np.float32),
                dt_s=0.02,
                created_at_s=100.0,
            ),
            pending=0,
        )
        observer.on_servo_step(policy_index=0, servo_index=0, step=step)
        terminal(100.02)
        return SimpleNamespace(
            prompt_fingerprint=getattr(runtime, "prompt_fingerprint", "prompt"),
            policy_steps=1,
            servo_steps=1,
            sent_actions=1,
            shadow_steps=0,
            elapsed_s=0.02,
            final_state="rollout_ready",
        )

    monkeypatch.setattr(sim_reference, "run_managed_rollout", fake_rollout)

    report = run_sim_trial(
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=tmp_path,
            kind=TrialKind.HOLD,
            duration_s=0.02,
            episode_index=3,
        )
    )

    assert report["status"] == "scored"
    assert report["object_success"] is True
    assert report["command_count"] == 1
    assert report["reference_success"] is None
    assert report["initial_previous_action"] == [1.0] * ACTION_DIM
    assert report["max_command_observation_delta"] == 1.0
    assert report["episode"]["frame_count"] == 1
    with np.load(report["episode"]["npz_path"]) as payload:
        np.testing.assert_array_equal(payload["joint_state"][0], np.full(ACTION_DIM, 1.0))
        np.testing.assert_array_equal(payload["action"][0], np.full(ACTION_DIM, 2.0))
        np.testing.assert_allclose(payload["timestamp"], np.array([0.0]))
    record = load_episode_records((Path(report["episode"]["npz_path"]),))[0]
    assert record.data.metadata["success_label_available"] is True
    assert record.data.metadata["action_timing"] == "observation_then_command"
    assert record.data.metadata["initial_previous_action"] == [1.0] * ACTION_DIM


def test_trial_no_overwrite_blocks_rollout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = False
    (tmp_path / "hold_task-a_seed-7_ep-000000_episode.npz").write_bytes(b"old")

    def fake_rollout(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("rollout should not run")

    monkeypatch.setattr(sim_reference, "run_managed_rollout", fake_rollout)

    with pytest.raises(ReferenceError, match="already exists"):
        run_sim_trial(
            SimTrialConfig(
                config=_config(),
                task=_task(),
                seed=7,
                output_dir=tmp_path,
                kind=TrialKind.HOLD,
                duration_s=0.02,
                episode_index=0,
            )
        )

    assert called is False


def test_trial_writes_failure_report_for_initial_object_arm_penetration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PenetratingAdapter(_ContactAdapter):
        def contacts(self) -> tuple[object, ...]:
            return (
                SimpleNamespace(
                    geom1="left_finger",
                    geom2="task_block_geom",
                    body1="left_gripper",
                    body2="task_block",
                    category1="left_arm",
                    category2="object",
                    distance=-1e-4,
                    forbidden=False,
                ),
            )

    monkeypatch.setattr(sim_reference, "MujocoBiSOAdapter", PenetratingAdapter)

    report = run_sim_trial(
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=tmp_path,
            kind=TrialKind.REFERENCE,
            duration_s=0.02,
            episode_index=0,
            times_s=(0.0, 0.02),
            targets=np.zeros((2, ACTION_DIM), dtype=np.float32),
        )
    )

    assert report["status"] == "execution_failure"
    assert report["failure_reason"] == "initial_object_arm_penetration"
    assert report["reference_success"] is False
    assert "episode" not in report
    assert json.loads((tmp_path / "reference_task-a_seed-7_ep-000000.json").read_text()) == report


def test_trial_retains_partial_episode_on_rollout_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sim_reference, "MujocoBiSOAdapter", _ContactAdapter)

    def fake_rollout(runtime: object, robot: object, **kwargs: object) -> object:
        del runtime, robot
        step = SimpleNamespace(
            observation=_frame(100.0, 1.0),
            executed_action=np.full(ACTION_DIM, 2.0, dtype=np.float32),
            sent=True,
            shadow=False,
            safety=SimpleNamespace(accepted=True, clipped=False, reasons=()),
            action=ActionChunk(
                np.full((1, ACTION_DIM), 3.0, dtype=np.float32),
                dt_s=0.02,
                created_at_s=100.0,
            ),
            pending=0,
        )
        kwargs["rollout_observer"].on_servo_step(policy_index=0, servo_index=0, step=step)
        raise SafetyRejectedError(("stale_action",))

    monkeypatch.setattr(sim_reference, "run_managed_rollout", fake_rollout)

    report = run_sim_trial(
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=tmp_path,
            kind=TrialKind.HOLD,
            duration_s=0.02,
            episode_index=0,
        )
    )

    assert report["status"] == "execution_failure"
    assert report["episode"]["partial"] is True
    record = load_episode_records((Path(report["episode"]["npz_path"]),))[0]
    assert record.data.metadata["initial_previous_action"] == [1.0] * ACTION_DIM
    assert record.data.metadata["reference_success"] is None
    with np.load(report["episode"]["npz_path"]) as payload:
        np.testing.assert_array_equal(payload["action"][0], np.full(ACTION_DIM, 2.0))


def test_reference_rollout_failure_marks_reference_unsuccessful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sim_reference, "MujocoBiSOAdapter", _ContactAdapter)

    def fake_rollout(runtime: object, robot: object, **kwargs: object) -> object:
        del runtime, robot
        step = SimpleNamespace(
            observation=_frame(100.0, 1.0),
            executed_action=np.full(ACTION_DIM, 2.0, dtype=np.float32),
            sent=True,
            shadow=False,
            safety=SimpleNamespace(accepted=True, clipped=False, reasons=()),
            action=ActionChunk(
                np.zeros((1, ACTION_DIM), dtype=np.float32),
                dt_s=0.02,
                created_at_s=100.0,
            ),
            pending=0,
        )
        kwargs["rollout_observer"].on_servo_step(policy_index=0, servo_index=0, step=step)
        raise SafetyRejectedError(("stale_action",))

    monkeypatch.setattr(sim_reference, "run_managed_rollout", fake_rollout)

    report = run_sim_trial(
        SimTrialConfig(
            config=_config(),
            task=_task(),
            seed=7,
            output_dir=tmp_path,
            kind=TrialKind.REFERENCE,
            duration_s=0.02,
            episode_index=0,
            times_s=(0.0, 0.02),
            targets=np.zeros((2, ACTION_DIM), dtype=np.float32),
        )
    )

    assert report["reference_success"] is False
    record = load_episode_records((Path(report["episode"]["npz_path"]),))[0]
    assert record.data.metadata["reference_success"] is False


def test_learned_policy_crops_live_history_and_uses_prompt_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sim_reference, "MujocoBiSOAdapter", _ContactAdapter)
    prompt_buffer = sim_reference.EpisodeBuffer(
        fps=1.0,
        task="prompt-task",
        episode_index=0,
    )
    prompt_buffer.append(_frame(0.0, 4.0, action=4.0))
    prompt_buffer.append(_frame(1.0, 4.0, action=4.0))
    prompt_buffer.append(_frame(2.0, 5.0, action=5.0))
    prompt_buffer.append(_frame(3.0, 5.0, action=5.0))
    prompt_npz, _ = prompt_buffer.save(tmp_path / "prompt")
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    seen: dict[str, object] = {}

    class Model:
        action_history_steps = 2
        action_horizon = 1

    class Policy:
        required_history_steps = 2

        def __init__(
            self,
            model: object,
            *,
            servo_hz: float,
            device: str = "cpu",
            joint_lower: tuple[float, ...] | None = None,
            joint_upper: tuple[float, ...] | None = None,
        ) -> None:
            del model, servo_hz, device, joint_lower, joint_upper

        def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
            del now_s
            seen["live_count"] = len(snapshot.live_frames)
            seen["live_values"] = [float(frame.joint_position[0]) for frame in snapshot.live_frames]
            seen["prompt_value"] = float(snapshot.prompt_frames[0].joint_position[0])
            return ActionChunk(np.zeros((1, ACTION_DIM), dtype=np.float32), dt_s=0.02, created_at_s=100.0)

    monkeypatch.setattr(
        sim_reference,
        "load_compact_wam_bundle",
        lambda *a, **k: SimpleNamespace(model=Model(), metadata={"checkpoint_id": "candidate"}),
    )
    monkeypatch.setattr(sim_reference, "CompactWAMPolicy", Policy)

    def fake_rollout(runtime: object, robot: object, **kwargs: object) -> object:
        snapshot = SimpleNamespace(
            prompt_frames=runtime.prompt.frames,
            live_frames=(
                _frame(1.0, 10.0, action=10.0),
                _frame(1.1, 11.0, action=11.0),
                _frame(1.2, 12.0, action=12.0),
            ),
            prompt_fingerprint=runtime.prompt.fingerprint,
        )
        runtime.policy.predict(cast(ContextSnapshot, snapshot), now_s=100.0)
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
            duration_s=0.02,
            episode_index=0,
            checkpoint_path=checkpoint,
            prompt_path=prompt_npz,
        )
    )

    assert seen == {"live_count": 2, "live_values": [11.0, 12.0], "prompt_value": 4.0}
