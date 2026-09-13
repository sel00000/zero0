from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import so101_wam.mujoco_benchmark as benchmark_module
from so101_wam.adapters.mujoco import MujocoAdapterError
from so101_wam.checkpoint import save_compact_wam_checkpoint
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer, load_episode
from so101_wam.deployment import file_sha256
from so101_wam.model import ActionDecoder, CompactWAM
from so101_wam.mujoco_cli import run_mujoco_checkpoint_session
from so101_wam.mujoco_identity import MujocoModelIdentity
from so101_wam.mujoco_benchmark import (
    BENCHMARK_POLICY,
    BENCHMARK_SCOPE,
    BenchmarkTask,
    BenchmarkTrialResult,
    MujocoBenchmarkError,
    _seeded_home_joint_position,
    load_benchmark_manifest,
    main as benchmark_main,
    run_benchmark,
    run_checkpoint_task_trial,
    score_terminal_joint_success,
)


MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
requires_mujoco = pytest.mark.skipif(
    not MUJOCO_AVAILABLE, reason="optional mujoco dependency is absent"
)


def _target() -> list[float]:
    target = [0.0] * ACTION_DIM
    target[4] = 1.5
    target[5] = 1.0
    target[10] = -1.5
    target[11] = 1.0
    return target


def _manifest(tmp_path: Path, **overrides: object) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "local-heldout-v1",
        "train_task_ids": ["train_wrist_home"],
        "heldout_tasks": [
            {
                "task_id": "heldout_wrist_delta",
                "label": "held-out wrist delta",
                "policy_steps": 2,
                "seeds": [7, 13, 29],
                "target_joint_position": _target(),
                "tolerance": 0.25,
            }
        ],
    }
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prompt_episode(tmp_path: Path) -> tuple[Path, Path]:
    buffer = EpisodeBuffer(
        fps=30.0,
        task="synthetic-prompt",
        task_index=1,
        episode_index=1,
    )
    joints = np.zeros(ACTION_DIM, dtype=np.float32)
    joints[5] = 1.0
    joints[11] = 1.0
    images = {key: np.zeros((24, 32, 3), dtype=np.uint8) for key in PRIMARY_CAMERA_KEYS}
    for frame_index in range(91):
        buffer.append(
            SensorimotorFrame(
                timestamp_s=frame_index / 30.0,
                images=images,
                joint_position=joints,
                executed_action=joints,
            )
        )
    return buffer.save(tmp_path, stem="prompt")


def test_manifest_rejects_train_task_reused_as_heldout(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        train_task_ids=["heldout_wrist_delta"],
    )

    with pytest.raises(MujocoBenchmarkError, match="disjoint"):
        load_benchmark_manifest(path)


def test_manifest_requires_at_least_three_seeds(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        heldout_tasks=[
            {
                "task_id": "heldout_wrist_delta",
                "label": "held-out wrist delta",
                "policy_steps": 2,
                "seeds": [7, 13],
                "target_joint_position": _target(),
                "tolerance": 0.25,
            }
        ],
    )

    with pytest.raises(MujocoBenchmarkError, match="at least 3 seeds"):
        load_benchmark_manifest(path)


def test_manifest_rejects_artifact_unsafe_task_id(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        heldout_tasks=[
            {
                "task_id": "heldout/wrist_delta",
                "label": "held-out wrist delta",
                "policy_steps": 2,
                "seeds": [7, 13, 29],
                "target_joint_position": _target(),
                "tolerance": 0.25,
            }
        ],
    )

    with pytest.raises(MujocoBenchmarkError, match="task_id"):
        load_benchmark_manifest(path)


def test_default_manifest_is_task_disjoint() -> None:
    manifest = load_benchmark_manifest(Path("configs/mujoco_heldout_benchmark.json"))

    assert manifest.train_task_ids == ("train_wrist_home",)
    assert tuple(task.task_id for task in manifest.heldout_tasks) == (
        "heldout_wrist_delta",
    )


def test_terminal_checker_fails_when_error_exceeds_tolerance() -> None:
    task = BenchmarkTask(
        task_id="heldout_wrist_delta",
        label="held-out wrist delta",
        policy_steps=2,
        seeds=(7, 13, 29),
        target_joint_position=tuple(_target()),
        tolerance=0.25,
    )
    final = np.zeros(ACTION_DIM, dtype=np.float32)

    outcome = score_terminal_joint_success(task, final)

    assert outcome.success is False
    assert outcome.failure_reason == "terminal_tolerance"


def test_terminal_checker_rejects_non_finite_measurement() -> None:
    task = BenchmarkTask(
        task_id="heldout_wrist_delta",
        label="held-out wrist delta",
        policy_steps=2,
        seeds=(7, 13, 29),
        target_joint_position=tuple(_target()),
        tolerance=0.25,
    )
    final = np.zeros(ACTION_DIM, dtype=np.float32)
    final[0] = np.nan

    with pytest.raises(MujocoBenchmarkError, match="finite"):
        score_terminal_joint_success(task, final)


def test_seeded_home_is_deterministic_and_seed_dependent() -> None:
    config = ProjectConfig.load(Path("configs/mujoco.toml"))

    first = _seeded_home_joint_position(config, seed=7)
    repeated = _seeded_home_joint_position(config, seed=7)
    different = _seeded_home_joint_position(config, seed=13)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert first[5] > config.safety.joint_lower[5]
    assert first[11] > config.safety.joint_lower[11]


def test_benchmark_records_one_trial_artifact_per_seed(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    calls: list[tuple[str, int]] = []

    def fake_runner(
        config: ProjectConfig,
        task: BenchmarkTask,
        *,
        seed: int,
    ) -> BenchmarkTrialResult:
        del config
        calls.append((task.task_id, seed))
        success = seed != 13
        return BenchmarkTrialResult(
            success=success,
            final_error=0.1 if success else 0.3,
            failure_reason=None if success else "terminal_tolerance",
            rollout={"policy_steps": task.policy_steps, "final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=task.target_joint_position,
        )

    report = run_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        trial_runner=fake_runner,
    )

    assert calls == [
        ("heldout_wrist_delta", 7),
        ("heldout_wrist_delta", 13),
        ("heldout_wrist_delta", 29),
    ]
    assert report["gate"] == "G8-heldout-benchmark"
    assert report["evidence_level"] == "simulation"
    assert report["policy"] == BENCHMARK_POLICY
    assert report["benchmark_scope"] == BENCHMARK_SCOPE
    assert report["result"] == "fail"
    assert report["task_disjoint"] is True
    assert "checkpoint_sha256" not in report
    assert "prompt_npz_sha256" not in report
    assert report["summary"]["trial_count"] == 3
    assert report["summary"]["success_count"] == 2
    assert report["summary"]["success_rate"] == pytest.approx(2 / 3)
    assert report["summary"]["failure_counts"] == {"terminal_tolerance": 1}
    low, high = report["summary"]["success_rate_95ci"]
    assert 0.0 <= low < report["summary"]["success_rate"] < high <= 1.0
    assert report["tasks"] == [
        {
            "task_id": "heldout_wrist_delta",
            "label": "held-out wrist delta",
            "trial_count": 3,
            "success_count": 2,
            "success_rate": pytest.approx(2 / 3),
            "success_rate_95ci": pytest.approx([low, high]),
            "final_error_mean": pytest.approx(1 / 6),
            "final_error_max": 0.3,
            "failure_counts": {"terminal_tolerance": 1},
        }
    ]
    for trial in report["trials"]:
        artifact_path = artifact_dir / trial["artifact"]
        assert artifact_path.exists()
        assert trial["trial_id"] == (f"{trial['task_id']}:seed:{trial['seed']}")
        assert trial["artifact_sha256"] == file_sha256(artifact_path)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["trial_id"] == trial["trial_id"]
        assert artifact["initial_joint_position"] == [0.0] * ACTION_DIM
        assert "checkpoint_sha256" not in artifact
        assert "prompt_npz_sha256" not in artifact

    with pytest.raises(MujocoBenchmarkError, match="already exists"):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=manifest_path,
            artifact_dir=artifact_dir,
            trial_runner=fake_runner,
        )
    assert len(calls) == 3


def test_benchmark_records_checkpoint_identity_for_compact_wam_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _manifest(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    checkpoint = tmp_path / "candidate.pt"
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    save_compact_wam_checkpoint(
        CompactWAM(latent_dim=8, transformer_layers=1, transformer_heads=2),
        checkpoint,
        metadata={"checkpoint_id": "candidate-test"},
    )

    def fake_trial(
        config: ProjectConfig,
        task: BenchmarkTask,
        *,
        seed: int,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        prompt_manifest_path: str | Path | None,
        device: str,
    ) -> BenchmarkTrialResult:
        del config, seed, device
        assert checkpoint_path == checkpoint
        assert prompt_path == prompt
        assert prompt_manifest_path == prompt_manifest
        return BenchmarkTrialResult(
            success=True,
            final_error=0.0,
            failure_reason=None,
            rollout={"policy_steps": task.policy_steps, "final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=task.target_joint_position,
        )

    monkeypatch.setattr(
        "so101_wam.mujoco_benchmark.run_checkpoint_task_trial",
        fake_trial,
    )

    report = benchmark_module.run_checkpoint_benchmark(
        ProjectConfig.load(Path("configs/mujoco.toml")),
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert report["gate"] == "G8-candidate-joint-proxy"
    assert report["policy"] == "compact_wam"
    assert report["benchmark_scope"] == "candidate_joint_proxy_not_heldout_verified"
    assert report["checkpoint_sha256"] == file_sha256(checkpoint)
    assert report["checkpoint_id"] == "candidate-test"
    assert report["prompt_npz_sha256"] == file_sha256(prompt)
    assert report["prompt_manifest_sha256"] == file_sha256(prompt_manifest)
    assert (
        report["prompt_fingerprint"]
        == load_episode(prompt, prompt_manifest).fingerprint
    )
    assert report["task_disjoint_basis"] == "manifest_ids_only"
    assert report["checkpoint_task_split"] == "not_verified"
    assert report["prompt_task_match"] == "not_verified"
    artifact = json.loads(
        (artifact_dir / "heldout_wrist_delta-seed-7.json").read_text(encoding="utf-8")
    )
    assert artifact["policy"] == "compact_wam"
    assert artifact["checkpoint_sha256"] == file_sha256(checkpoint)
    assert artifact["checkpoint_id"] == "candidate-test"
    assert artifact["prompt_npz_sha256"] == file_sha256(prompt)
    assert artifact["prompt_manifest_sha256"] == file_sha256(prompt_manifest)
    assert artifact["prompt_fingerprint"] == report["prompt_fingerprint"]
    assert artifact["checkpoint_task_split"] == "not_verified"
    assert artifact["prompt_task_match"] == "not_verified"


def test_reference_benchmark_rejects_checkpoint_evidence_injection(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(TypeError, match="checkpoint_evidence"):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=artifact_dir,
            checkpoint_evidence=object(),  # type: ignore[call-arg]
        )

    assert not artifact_dir.exists()


def test_checkpoint_trial_uses_seeded_pose_and_records_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(Path("configs/mujoco.toml"))
    expected_initial = tuple(0.25 for _ in range(ACTION_DIM))
    task = BenchmarkTask(
        task_id="heldout_wrist_delta",
        label="held-out wrist delta",
        policy_steps=2,
        seeds=(7, 13, 29),
        target_joint_position=tuple(_target()),
        tolerance=0.1,
    )
    seen_home: tuple[float, ...] | None = None

    def fake_session(
        session_config: ProjectConfig,
        *,
        checkpoint_path: str | Path,
        prompt_path: str | Path,
        manifest_path: str | Path | None,
        policy_steps: int,
        device: str,
    ) -> dict[str, object]:
        nonlocal seen_home
        del checkpoint_path, prompt_path, manifest_path, device
        seen_home = tuple(session_config.mujoco.home_joint_position)
        return {
            "rollout": {"policy_steps": policy_steps, "final_state": "rollout_ready"},
            "terminal_joint_position": [0.0] * ACTION_DIM,
        }

    monkeypatch.setattr(
        "so101_wam.mujoco_benchmark._seeded_home_joint_position",
        lambda config, *, seed: expected_initial,
    )
    monkeypatch.setattr(
        "so101_wam.mujoco_benchmark.run_mujoco_checkpoint_session",
        fake_session,
    )

    result = run_checkpoint_task_trial(
        config,
        task,
        seed=7,
        checkpoint_path=tmp_path / "candidate.pt",
        prompt_path=tmp_path / "prompt.npz",
        prompt_manifest_path=tmp_path / "prompt.json",
        device="cpu",
    )

    assert seen_home == expected_initial
    assert result.success is False
    assert result.failure_reason == "terminal_tolerance"
    assert result.initial_joint_position == expected_initial


def test_checkpoint_trial_rejects_missing_terminal_position_as_infrastructure_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig.load(Path("configs/mujoco.toml"))
    task = BenchmarkTask(
        task_id="heldout_wrist_delta",
        label="held-out wrist delta",
        policy_steps=2,
        seeds=(7, 13, 29),
        target_joint_position=tuple(_target()),
        tolerance=0.1,
    )

    def fake_session(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {"rollout": {"final_state": "rollout_ready"}}

    monkeypatch.setattr(
        "so101_wam.mujoco_benchmark.run_mujoco_checkpoint_session",
        fake_session,
    )

    with pytest.raises(MujocoBenchmarkError, match="terminal joints"):
        run_checkpoint_task_trial(
            config,
            task,
            seed=7,
            checkpoint_path=tmp_path / "candidate.pt",
            prompt_path=tmp_path / "prompt.npz",
        )


@requires_mujoco
def test_checkpoint_trial_runs_real_compact_wam_session(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    prompt, prompt_manifest = _prompt_episode(tmp_path)
    model = CompactWAM(
        latent_dim=8,
        transformer_layers=1,
        transformer_heads=2,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    axis_mean = torch.zeros(ACTION_DIM)
    axis_mean[5] = 1.0
    axis_mean[11] = 1.0
    model.set_axis_normalization(axis_mean, torch.ones(ACTION_DIM))
    save_compact_wam_checkpoint(
        model,
        checkpoint,
        metadata={"checkpoint_id": "candidate-integration-test"},
    )
    task = BenchmarkTask(
        task_id="heldout_wrist_delta",
        label="held-out wrist delta",
        policy_steps=1,
        seeds=(7, 13, 29),
        target_joint_position=tuple(_target()),
        tolerance=0.25,
    )

    result = run_checkpoint_task_trial(
        ProjectConfig.load(Path("configs/mujoco_robot_free.toml")),
        task,
        seed=7,
        checkpoint_path=checkpoint,
        prompt_path=prompt,
        prompt_manifest_path=prompt_manifest,
    )

    assert isinstance(result.success, bool)
    assert len(result.initial_joint_position) == ACTION_DIM
    assert len(result.final_joint_position) == ACTION_DIM
    assert result.rollout["policy_steps"] == 1
    assert result.rollout["sent_actions"] > 0


@requires_mujoco
def test_decoder_identity(tmp_path: Path) -> None:
    config = ProjectConfig.load(Path("configs/mujoco_robot_free.toml"))
    prompt, manifest = _prompt_episode(tmp_path)
    identities = []

    for mode in ActionDecoder:
        checkpoint = tmp_path / f"{mode.value}.pt"
        model = CompactWAM(
            latent_dim=8,
            transformer_layers=1,
            transformer_heads=2,
            future_steps=3,
            action_decoder=mode,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        axis_mean = torch.zeros(ACTION_DIM)
        axis_mean[5] = 1.0
        axis_mean[11] = 1.0
        model.set_axis_normalization(axis_mean, torch.ones(ACTION_DIM))
        save_compact_wam_checkpoint(
            model,
            checkpoint,
            metadata={
                "checkpoint_id": mode.value,
                "offline_trained": True,
                "trained": False,
                "deployment_ready": False,
            },
        )

        report = run_mujoco_checkpoint_session(
            config,
            checkpoint_path=checkpoint,
            prompt_path=prompt,
            manifest_path=manifest,
            policy_steps=1,
            device="cpu",
        )
        identities.append(
            MujocoModelIdentity.from_payload(report["mujoco_model_identity"])
        )
        assert report["checkpoint_sha256"] == file_sha256(checkpoint)
        assert report["rollout"]["policy_steps"] == 1
        assert report["rollout"]["sent_actions"] > 0

    assert identities == [identities[0]] * len(identities)


def test_benchmark_rejects_target_outside_safety_before_trials(tmp_path: Path) -> None:
    unsafe = _target()
    unsafe[4] = 1000.0
    manifest_path = _manifest(
        tmp_path,
        heldout_tasks=[
            {
                "task_id": "heldout_unsafe",
                "label": "unsafe target",
                "policy_steps": 2,
                "seeds": [7, 13, 29],
                "target_joint_position": unsafe,
                "tolerance": 0.25,
            }
        ],
    )

    def unexpected_runner(
        config: ProjectConfig,
        task: BenchmarkTask,
        *,
        seed: int,
    ) -> BenchmarkTrialResult:
        del config, task, seed
        raise AssertionError("unsafe task must fail before running")

    with pytest.raises(MujocoBenchmarkError, match="outside safety limits"):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=manifest_path,
            artifact_dir=tmp_path / "artifacts",
            trial_runner=unexpected_runner,
        )


@pytest.mark.parametrize(
    ("success", "failure_reason", "final_error", "message"),
    (
        (False, None, 0.3, "failure_reason"),
        (True, "terminal_tolerance", 0.1, "failure_reason"),
        (False, "terminal_tolerance", float("inf"), "final_error"),
    ),
)
def test_benchmark_rejects_invalid_trial_evidence_before_writing(
    tmp_path: Path,
    success: bool,
    failure_reason: str | None,
    final_error: float,
    message: str,
) -> None:
    artifact_dir = tmp_path / "artifacts"

    def invalid_runner(
        config: ProjectConfig,
        task: BenchmarkTask,
        *,
        seed: int,
    ) -> BenchmarkTrialResult:
        del config, seed
        return BenchmarkTrialResult(
            success=success,
            final_error=final_error,
            failure_reason=failure_reason,
            rollout={"final_state": "rollout_ready"},
            initial_joint_position=(0.0,) * ACTION_DIM,
            final_joint_position=task.target_joint_position,
        )

    with pytest.raises(MujocoBenchmarkError, match=message):
        run_benchmark(
            ProjectConfig.load(Path("configs/mujoco.toml")),
            manifest_path=_manifest(tmp_path),
            artifact_dir=artifact_dir,
            trial_runner=invalid_runner,
        )
    assert not artifact_dir.exists()


def test_benchmark_cli_writes_report_and_keeps_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "benchmark.json"
    artifact_dir = tmp_path / "artifacts"
    manifest_path = _manifest(tmp_path)
    report = {
        "schema_version": 1,
        "gate": "G8-heldout-benchmark",
        "result": "pass",
        "evidence_level": "simulation",
        "summary": {"trial_count": 3},
    }

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return report

    monkeypatch.setattr("so101_wam.mujoco_benchmark.run_benchmark", fake_run)

    assert (
        benchmark_main(
            [
                "--config",
                "configs/mujoco.toml",
                "--manifest",
                str(manifest_path),
                "--artifact-dir",
                str(artifact_dir),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report


def test_benchmark_cli_uses_compact_wam_runner_when_checkpoint_is_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "benchmark.json"
    artifact_dir = tmp_path / "artifacts"
    manifest_path = _manifest(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    prompt = tmp_path / "prompt.npz"
    prompt_manifest = tmp_path / "prompt.json"
    save_compact_wam_checkpoint(
        CompactWAM(latent_dim=16, transformer_heads=4),
        checkpoint,
        metadata={"checkpoint_id": "candidate-test"},
    )
    prompt.write_bytes(b"prompt-bytes")
    prompt_manifest.write_text("{}", encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        assert kwargs["checkpoint_path"] == checkpoint
        assert kwargs["prompt_path"] == prompt
        assert kwargs["prompt_manifest_path"] == prompt_manifest
        return {
            "schema_version": 1,
            "gate": "G8-candidate-joint-proxy",
            "result": "pass",
            "policy": "compact_wam",
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_id": "candidate-test",
            "prompt_npz_sha256": file_sha256(prompt),
            "prompt_manifest_sha256": file_sha256(prompt_manifest),
        }

    monkeypatch.setattr(
        "so101_wam.mujoco_benchmark.run_checkpoint_benchmark",
        fake_run,
    )

    assert (
        benchmark_main(
            [
                "--config",
                "configs/mujoco.toml",
                "--manifest",
                str(manifest_path),
                "--artifact-dir",
                str(artifact_dir),
                "--report",
                str(report_path),
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                str(prompt),
                "--prompt-manifest",
                str(prompt_manifest),
            ]
        )
        == 0
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["policy"] == "compact_wam"
    assert report["checkpoint_sha256"] == file_sha256(checkpoint)
    assert report["checkpoint_id"] == "candidate-test"
    assert report["prompt_npz_sha256"] == file_sha256(prompt)
    assert report["prompt_manifest_sha256"] == file_sha256(prompt_manifest)
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize(
    "extra_args",
    (
        ("--checkpoint", "candidate.pt"),
        ("--prompt", "prompt.npz"),
        ("--prompt-manifest", "prompt.json"),
    ),
)
def test_benchmark_cli_requires_complete_checkpoint_input(
    tmp_path: Path,
    extra_args: tuple[str, str],
) -> None:
    args = [
        "--config",
        "configs/mujoco.toml",
        "--manifest",
        str(_manifest(tmp_path)),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--report",
        str(tmp_path / "report.json"),
        *extra_args,
    ]

    with pytest.raises(SystemExit) as error:
        benchmark_main(args)

    assert error.value.code == 2


def test_benchmark_cli_returns_one_for_failed_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    report = {
        "schema_version": 1,
        "gate": "G8-heldout-benchmark",
        "result": "fail",
    }

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return report

    monkeypatch.setattr("so101_wam.mujoco_benchmark.run_benchmark", fake_run)

    result = benchmark_main(
        [
            "--config",
            "configs/mujoco.toml",
            "--manifest",
            str(_manifest(tmp_path)),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--report",
            str(report_path),
        ]
    )

    assert result == 1
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report


def test_benchmark_cli_reports_adapter_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise MujocoAdapterError("adapter failed")

    monkeypatch.setattr("so101_wam.mujoco_benchmark.run_benchmark", fail_run)

    with pytest.raises(SystemExit) as error:
        benchmark_main(
            [
                "--config",
                "configs/mujoco.toml",
                "--manifest",
                str(_manifest(tmp_path)),
                "--artifact-dir",
                str(tmp_path / "artifacts"),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )

    assert error.value.code == 2
    assert "adapter failed" in capsys.readouterr().err
