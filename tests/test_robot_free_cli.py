from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from so101_wam.checkpoint import load_compact_wam_bundle
from so101_wam.dataset import load_episode
from so101_wam.deployment import file_sha256
from so101_wam.robot_free_cli import ROBOT_FREE_LIMITATIONS
from so101_wam.robot_free_verifier import (
    ROBOT_FREE_VERIFICATION_SCHEMA_VERSION,
    RobotFreeVerificationError,
    main as verifier_main,
    verify_robot_free_result,
)

ROOT = Path(__file__).resolve().parents[1]

MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
requires_mujoco = pytest.mark.skipif(
    not MUJOCO_AVAILABLE, reason="optional mujoco dependency is absent"
)


def _episode_fingerprints(directory: Path) -> tuple[str, ...]:
    return tuple(
        load_episode(path).fingerprint for path in sorted(directory.glob("*.npz"))
    )


def test_robot_free_cli_rejects_nonempty_output_dir_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import robot_free_cli

    output_dir = tmp_path / "robot_free"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def unexpected_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("robot-free E2E must not start for a nonempty output dir")

    monkeypatch.setattr(robot_free_cli, "_write_synthetic_episodes", unexpected_run)

    with pytest.raises(robot_free_cli.RobotFreePipelineError, match="must be empty"):
        robot_free_cli.run_robot_free_pipeline(output_dir)

    assert tuple(output_dir.iterdir()) == (sentinel,)


def test_robot_free_cli_prints_runner_result_as_stdout_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from so101_wam import robot_free_cli

    result = {
        "schema_version": 1,
        "mode": "robot_free",
        "evidence_level": "simulation",
        "trained": False,
        "deployment_ready": False,
    }

    def fake_run_robot_free_demo(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return result

    monkeypatch.setattr(
        robot_free_cli,
        "run_robot_free_demo",
        fake_run_robot_free_demo,
    )

    exit_code = robot_free_cli.main(
        [
            "--output-dir",
            str(tmp_path / "robot_free"),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == result


def test_robot_free_cli_rejects_invalid_policy_steps_before_writing(
    tmp_path: Path,
) -> None:
    from so101_wam import robot_free_cli

    output_dir = tmp_path / "robot_free"

    with pytest.raises(SystemExit) as captured:
        robot_free_cli.main(
            [
                "--output-dir",
                str(output_dir),
                "--policy-steps",
                "0",
            ]
        )

    assert captured.value.code == 2
    assert not output_dir.exists()


@requires_mujoco
def test_robot_free_e2e_publishes_offline_candidate_and_compact_wam_g8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from so101_wam import robot_free_cli

    output_dir = tmp_path / "robot_free"
    monkeypatch.chdir(tmp_path)

    exit_code = robot_free_cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
            "--policy-steps",
            "1",
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    candidate_path = output_dir / "candidate.pt"
    training_report_path = output_dir / "candidate.training.json"
    mujoco_report_path = output_dir / "mujoco.g8.json"
    result_path = output_dir / "robot_free_result.json"

    assert exit_code == 0
    train_dir = output_dir / "episodes" / "train"
    validation_dir = output_dir / "episodes" / "validation"
    assert len(tuple(train_dir.glob("*.npz"))) == 2
    assert len(tuple(validation_dir.glob("*.npz"))) == 2
    assert {load_episode(path).task for path in train_dir.glob("*.npz")}.isdisjoint(
        {load_episode(path).task for path in validation_dir.glob("*.npz")}
    )
    assert candidate_path.exists()
    assert training_report_path.exists()
    assert mujoco_report_path.exists()

    bundle = load_compact_wam_bundle(candidate_path)
    assert bundle.metadata["offline_trained"] is True
    assert bundle.metadata["trained"] is False
    assert bundle.metadata["deployment_ready"] is False

    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    assert training_report["trained"] is False
    assert training_report["deployment_ready"] is False
    assert training_report["protocol"]["split"] == "task_disjoint"
    assert training_report["protocol"]["real_output_authorized"] is False

    mujoco_report = json.loads(mujoco_report_path.read_text(encoding="utf-8"))
    assert mujoco_report["policy"] == "compact_wam"
    assert mujoco_report["evidence_level"] == "simulation"
    assert mujoco_report["checkpoint_sha256"] == file_sha256(candidate_path)
    assert mujoco_report["checkpoint_id"] == "robot-free-candidate-seed-17"

    assert payload["trained"] is False
    assert payload["deployment_ready"] is False
    assert payload["candidate_checkpoint_sha256"] == file_sha256(candidate_path)
    assert payload["g8_report_sha256"] == file_sha256(mujoco_report_path)
    assert payload["evidence_level"] == "simulation"
    assert payload["evidence_inputs"] == ["offline", "simulation"]
    assert payload["real_output_authorized"] is False
    assert "offline" in payload["real_output_rejection_reason"]
    assert (
        mujoco_report["rollout"]["sent_actions"]
        == mujoco_report["rollout"]["servo_steps"]
    )
    assert mujoco_report["rollout"]["shadow_steps"] == 0
    assert "result" not in payload["artifacts"]
    assert payload["artifacts"]["config"]["path"] == "effective_config.toml"
    assert (
        payload["training"]["checkpoint_path"]
        == payload["artifacts"]["checkpoint"]["path"]
    )
    assert (
        payload["training"]["report_path"]
        == payload["artifacts"]["training_report"]["path"]
    )
    assert all(
        not Path(artifact["path"]).is_absolute()
        for artifact in payload["artifacts"].values()
    )
    assert {name for name in payload["artifacts"] if "episode_manifest" in name} == {
        "train_episode_manifest_00",
        "train_episode_manifest_01",
        "validation_episode_manifest_00",
        "validation_episode_manifest_01",
    }
    assert all(
        set(artifact) == {"path", "sha256"}
        for artifact in payload["artifacts"].values()
    )
    for artifact in payload["artifacts"].values():
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = output_dir / artifact_path
        assert file_sha256(artifact_path) == artifact["sha256"]

    verification = verify_robot_free_result(result_path)
    assert verification["schema_version"] == ROBOT_FREE_VERIFICATION_SCHEMA_VERSION
    assert verification["result"] == "pass"
    assert verification["verified_artifact_count"] == len(payload["artifacts"])
    assert verification["real_output_authorized"] is False

    assert verifier_main(["--result", str(result_path)]) == 0
    assert json.loads(capsys.readouterr().out) == verification

    tampered_result = json.loads(json.dumps(payload))
    tampered_result["training"]["trained"] = True
    result_path.write_text(json.dumps(tampered_result), encoding="utf-8")
    with pytest.raises(RobotFreeVerificationError, match="training summary"):
        verify_robot_free_result(result_path)

    tampered_result = json.loads(json.dumps(payload))
    tampered_result["limitations"] = ["safe for real deployment"]
    result_path.write_text(json.dumps(tampered_result), encoding="utf-8")
    with pytest.raises(RobotFreeVerificationError, match="limitations"):
        verify_robot_free_result(result_path)

    tampered_result = json.loads(json.dumps(payload))
    tampered_result["unexpected"] = "field"
    result_path.write_text(json.dumps(tampered_result), encoding="utf-8")
    with pytest.raises(RobotFreeVerificationError, match="fields mismatch"):
        verify_robot_free_result(result_path)

    original_training_report = training_report_path.read_bytes()
    tampered_training_report = json.loads(original_training_report)
    tampered_training_report["data"]["train_tasks"][0]["task"] = "forged-task"
    training_report_path.write_text(
        json.dumps(tampered_training_report),
        encoding="utf-8",
    )
    tampered_result = json.loads(json.dumps(payload))
    tampered_result["artifacts"]["training_report"]["sha256"] = file_sha256(
        training_report_path
    )
    result_path.write_text(json.dumps(tampered_result), encoding="utf-8")
    with pytest.raises(RobotFreeVerificationError, match="core digest"):
        verify_robot_free_result(result_path)

    training_report_path.write_bytes(original_training_report)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    tampered_episode = output_dir / payload["artifacts"]["train_episode_00"]["path"]
    original_episode = tampered_episode.read_bytes()
    tampered_episode.write_bytes(original_episode + b"tampered")
    with pytest.raises(RobotFreeVerificationError, match="SHA-256 mismatch"):
        verify_robot_free_result(result_path)


def test_robot_free_verifier_rejects_bundle_relative_path_escape(
    tmp_path: Path,
) -> None:
    artifact_names = {
        "checkpoint",
        "config",
        "g8_report",
        "training_report",
        *(f"train_episode_{index:02d}" for index in range(2)),
        *(f"train_episode_manifest_{index:02d}" for index in range(2)),
        *(f"validation_episode_{index:02d}" for index in range(2)),
        *(f"validation_episode_manifest_{index:02d}" for index in range(2)),
    }
    payload = {
        "schema_version": "so101_wam.robot_free_pipeline.v1",
        "result": "pass",
        "mode": "robot_free",
        "evidence_level": "simulation",
        "evidence_inputs": ["offline", "simulation"],
        "trained": False,
        "deployment_ready": False,
        "real_output_authorized": False,
        "real_output_rejection_reason": "offline candidate rejected",
        "candidate_checkpoint_sha256": "0" * 64,
        "g8_report_sha256": "0" * 64,
        "synthetic_data": {
            "fps": 30.0,
            "duration_s": 3.0,
            "frame_count": 91,
            "resolution": {"height": 24, "width": 32},
            "action_source": "synthetic_deterministic_no_robot_no_goal_write",
            "train_episode_count": 2,
            "validation_episode_count": 2,
            "train_task_count": 1,
            "validation_task_count": 1,
            "task_split": "task_disjoint",
        },
        "training": {},
        "g8": {},
        "artifacts": {
            name: {"path": "../outside", "sha256": "0" * 64} for name in artifact_names
        },
        "limitations": list(ROBOT_FREE_LIMITATIONS),
    }
    result_path = tmp_path / "robot_free_result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RobotFreeVerificationError, match="bundle-relative"):
        verify_robot_free_result(result_path)


def test_robot_free_verifier_rejects_overflowed_json_number(tmp_path: Path) -> None:
    result_path = tmp_path / "robot_free_result.json"
    result_path.write_text('{"value": 1e999}', encoding="utf-8")

    with pytest.raises(RobotFreeVerificationError, match="non-finite JSON number"):
        verify_robot_free_result(result_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "actuation_enabled = true",
            "actuation_enabled = false",
            "actuation_enabled=true",
        ),
        ("camera_width = 32", "camera_width = 64", "resolution mismatch"),
        (
            "forbid_collisions = true",
            "forbid_collisions = false",
            "forbid_collisions=true",
        ),
    ],
)
def test_robot_free_config_mismatch_fails_before_writing_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    message: str,
) -> None:
    from so101_wam import robot_free_cli

    config_text = (ROOT / "configs" / "mujoco_robot_free.toml").read_text(
        encoding="utf-8"
    )
    assert old in config_text
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(config_text.replace(old, new, 1), encoding="utf-8")
    output_dir = tmp_path / "robot_free"

    def unexpected_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("invalid config must fail before episode generation")

    monkeypatch.setattr(robot_free_cli, "_write_synthetic_episodes", unexpected_run)

    with pytest.raises(robot_free_cli.RobotFreePipelineError, match=message):
        robot_free_cli.run_robot_free_pipeline(output_dir, config_path=config_path)

    assert not output_dir.exists()


def test_robot_free_e2e_generates_deterministic_synthetic_episodes(
    tmp_path: Path,
) -> None:
    from so101_wam import robot_free_cli

    first = tmp_path / "first"
    second = tmp_path / "second"

    for output_dir in (first, second):
        robot_free_cli._write_synthetic_episodes(
            train_dir=output_dir / "train",
            validation_dir=output_dir / "validation",
            seed=23,
        )

    assert _episode_fingerprints(first / "train") == _episode_fingerprints(
        second / "train"
    )
    assert _episode_fingerprints(first / "validation") == _episode_fingerprints(
        second / "validation"
    )
