from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
from typing import Any

import pytest

from so101_wam import reference_learning as study
from so101_wam.config import ProjectConfig
from so101_wam.seen_task_training import SeenTaskArtifacts


@pytest.mark.parametrize("hold_success", [False, True])
def test_failed_gate_skips_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hold_success: bool
) -> None:
    calls: list[str] = []

    def run(trial: Any) -> dict[str, Any]:
        calls.append(trial.kind.value)
        return {
            "status": "scored",
            "object_success": hold_success,
            "reference_success": hold_success if trial.kind is study.TrialKind.REFERENCE else None,
        }

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("learning must not start when controls fail")

    monkeypatch.setattr(study, "run_sim_trial", run)
    monkeypatch.setattr(study, "train_seen_task_candidate", unexpected)
    monkeypatch.setattr(study, "_source_hashes", lambda: {})
    report = study.run_reference_learning(tmp_path / "run")
    assert len(calls) == 4
    assert report["learning"]["status"] == "not_attempted"
    assert report["generalization"]["status"] == "not_attempted"
    assert report["result"] == "reference_gate_failed"
    saved = json.loads((tmp_path / "run" / "reference-learning-report.json").read_text())
    assert saved == report


def test_study_refuses_overwrite(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("keep")
    with pytest.raises(study.ReferenceLearningError, match="empty"):
        study.run_reference_learning(tmp_path)
    assert (tmp_path / "existing.txt").read_text() == "keep"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH), runtime=replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH).runtime, backend="fake")),
            "mujoco",
        ),
        (
            replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH), runtime=replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH).runtime, actuation_enabled=False)),
            "actuation_enabled",
        ),
        (
            replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH), mujoco=replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH).mujoco, forbid_collisions=False)),
            "forbid_collisions",
        ),
        (
            replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH), runtime=replace(ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH).runtime, action_horizon=9)),
            "action_horizon",
        ),
    ],
)
def test_gate_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: ProjectConfig,
    match: str,
) -> None:
    monkeypatch.setattr(study.ProjectConfig, "load", lambda path: config)
    monkeypatch.setattr(study, "run_sim_trial", _unexpected_run)

    with pytest.raises(study.ReferenceLearningError, match=match):
        study.run_reference_learning(tmp_path / "run")

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mode", ["unbounded", "affine_tanh", "normalized_clamp"])
def test_gate_trains_two_learned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    calls: list[str] = []
    trained: list[dict[str, Any]] = []

    def run(trial: Any) -> dict[str, Any]:
        calls.append(trial.kind.value)
        success = trial.kind is not study.TrialKind.HOLD
        outcome: dict[str, Any] = {"status": "scored", "object_success": success}
        if trial.kind is study.TrialKind.REFERENCE:
            outcome["reference_success"] = True
            episode = trial.output_dir / "episode.npz"
            manifest = trial.output_dir / "episode.json"
            episode.parent.mkdir(parents=True, exist_ok=True)
            episode.write_bytes(f"episode:{trial.seed}".encode("utf-8"))
            manifest.write_text(f"manifest:{trial.seed}", encoding="utf-8")
            outcome["episode"] = {
                "npz_path": str(episode),
                "npz_sha256": study.file_sha256(episode),
                "manifest_path": str(manifest),
                "manifest_sha256": study.file_sha256(manifest),
            }
        return outcome

    def train(records: Any, **kwargs: Any) -> Any:
        trained.append({"records": records, **kwargs})
        checkpoint = Path(kwargs["checkpoint_path"])
        report = Path(kwargs["report_path"])
        checkpoint.write_bytes(b"checkpoint")
        report.write_text("{}", encoding="utf-8")
        return SeenTaskArtifacts(
            checkpoint_path=str(checkpoint),
            report_path=str(report),
            checkpoint_id=kwargs["checkpoint_id"],
            training_evidence_sha256="a" * 64,
            episode_count=2,
            window_count=4,
            optimizer_steps=3,
            trained=False,
            deployment_ready=False,
        )

    monkeypatch.setattr(study, "run_sim_trial", run)
    monkeypatch.setattr(study, "train_seen_task_candidate", train)
    monkeypatch.setattr(study, "load_episode_records", lambda paths: tuple(paths))
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "f" * 64})

    if mode != "unbounded":
        from so101_wam.model import ActionRangeConstraint

        assert mode in {item.value for item in ActionRangeConstraint}
        constraint = ActionRangeConstraint(mode)
        report = study.run_reference_learning(
            tmp_path / "run", action_range_constraint=constraint,
        )
        assert trained[0]["action_range_constraint"] is constraint
        config = ProjectConfig.load(study.DEFAULT_ROBOT_FREE_CONFIG_PATH)
        assert trained[0]["joint_lower"] == config.safety.joint_lower
        assert trained[0]["joint_upper"] == config.safety.joint_upper
        protocol = json.loads((tmp_path / "run" / "protocol.json").read_text())
        assert protocol["action_output"]["mode"] == mode
        assert protocol["training"]["stage1_steps"] == 200
        assert protocol["training"]["stage2_steps"] == 800
        assert protocol["action_output"]["joint_lower"] == list(config.safety.joint_lower)
    else:
        report = study.run_reference_learning(tmp_path / "run")

    assert calls.count(study.TrialKind.REFERENCE.value) == 2
    assert calls.count(study.TrialKind.HOLD.value) == 2
    assert calls.count(study.TrialKind.LEARNED.value) == 2
    assert len(trained) == 1
    assert trained[0]["device"] == "cpu"
    assert report["result"] == "seen_task_success"
    assert report["trained"] is False
    assert report["deployment_ready"] is False
    assert report["zero_shot_claimed"] is False
    assert report["real_world_success_claimed"] is False
    assert report["learning"]["status"] == "completed"
    assert report["learning"]["success_count"] == 2
    assert report["learned_trials_not_attempted"] == []


def test_training_failure_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "run_sim_trial", _run_success)
    monkeypatch.setattr(study, "load_episode_records", lambda paths: tuple(paths))
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "f" * 64})

    def fail_train(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("training exploded")

    monkeypatch.setattr(study, "train_seen_task_candidate", fail_train)

    report = study.run_reference_learning(tmp_path / "run")

    assert report["result"] == "execution_failure"
    assert report["learning"]["status"] == "failed"
    assert report["learning"]["reason"] == "training exploded"
    assert report["failure"] == {
        "type": "RuntimeError",
        "message": "training exploded",
    }
    assert len(report["trials"]) == 4
    assert report["learned_trials_not_attempted"] == list(study.INITIAL_CONDITIONS)


def test_input_drift_stops_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def drift(paths: Any) -> tuple[str, ...]:
        first = Path(next(iter(paths)))
        first.write_bytes(b"changed")
        return tuple(paths)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("trainer must not run after input drift")

    monkeypatch.setattr(study, "run_sim_trial", _run_success)
    monkeypatch.setattr(study, "load_episode_records", drift)
    monkeypatch.setattr(study, "train_seen_task_candidate", unexpected)
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "f" * 64})

    report = study.run_reference_learning(tmp_path / "run")

    assert report["result"] == "execution_failure"
    assert report["failure"]["type"] == "ReferenceLearningError"
    assert "input changed" in report["failure"]["message"]
    assert report["learning"]["status"] == "failed"


def test_source_drift_stops_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def source() -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls < 7:
            return {"source.py": "f" * 64}
        return {"source.py": "e" * 64}

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("trainer must not run after source drift")

    monkeypatch.setattr(study, "run_sim_trial", _run_success)
    monkeypatch.setattr(study, "load_episode_records", lambda paths: tuple(paths))
    monkeypatch.setattr(study, "train_seen_task_candidate", unexpected)
    monkeypatch.setattr(study, "_source_hashes", source)

    report = study.run_reference_learning(tmp_path / "run")

    assert report["result"] == "execution_failure"
    assert report["failure"]["type"] == "ReferenceLearningError"
    assert "source changed" in report["failure"]["message"]


def test_protocol_written_before_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"

    def train(*args: Any, **kwargs: Any) -> None:
        assert (root / "training-protocol.json").is_file()
        protocol = json.loads((root / "training-protocol.json").read_text())
        assert protocol["training"]["stage1_steps"] == 200
        assert protocol["training"]["stage2_steps"] == 800
        assert protocol["training"]["learning_rate"] == 1e-3
        raise RuntimeError("stop after protocol check")

    monkeypatch.setattr(study, "run_sim_trial", _run_success)
    monkeypatch.setattr(study, "load_episode_records", lambda paths: tuple(paths))
    monkeypatch.setattr(study, "train_seen_task_candidate", train)
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "f" * 64})

    report = study.run_reference_learning(root)

    assert report["result"] == "execution_failure"
    assert report["trained"] is False
    assert report["deployment_ready"] is False
    assert report["zero_shot_claimed"] is False


def test_trial_exception_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def run(trial: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("slot exploded")
        return _run_success(trial)

    monkeypatch.setattr(study, "run_sim_trial", run)
    monkeypatch.setattr(study, "train_seen_task_candidate", _unexpected_train)
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "f" * 64})

    report = study.run_reference_learning(tmp_path / "run")

    assert report["result"] == "execution_failure"
    assert report["failure"] == {
        "type": "RuntimeError",
        "message": "slot exploded",
    }
    assert len(report["trials"]) == 2
    assert report["trials"][0]["outcome"]["status"] == "scored"
    assert report["trials"][1]["outcome"]["failure_reason"] == "trial_exception"


@pytest.mark.parametrize(
    ("outcome", "match"),
    [
        ({"reference_success": False}, "reference_success"),
        ({"reference_success": "yes"}, "reference_success"),
        ({"drop_episode": True}, "episode"),
        ({"drop_reference_success": True}, "reference_success"),
    ],
)
def test_reference_marker_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: dict[str, object],
    match: str,
) -> None:
    def run(trial: Any) -> dict[str, Any]:
        if trial.kind is not study.TrialKind.REFERENCE:
            return _run_success(trial)
        return _run_success(trial, reference_update=outcome)

    monkeypatch.setattr(study, "run_sim_trial", run)
    monkeypatch.setattr(study, "train_seen_task_candidate", _unexpected_train)
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "f" * 64})

    report = study.run_reference_learning(tmp_path / "run")

    assert report["result"] == "execution_failure"
    assert report["failure"]["type"] == "ReferenceLearningError"
    assert match in report["failure"]["message"]


def test_hold_marker_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(trial: Any) -> dict[str, Any]:
        if trial.kind is study.TrialKind.HOLD:
            return {
                "status": "scored",
                "object_success": False,
                "reference_success": True,
            }
        return _run_success(trial)

    monkeypatch.setattr(study, "run_sim_trial", run)
    monkeypatch.setattr(study, "train_seen_task_candidate", _unexpected_train)
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "f" * 64})

    report = study.run_reference_learning(tmp_path / "run")

    assert report["result"] == "execution_failure"
    assert report["failure"]["type"] == "ReferenceLearningError"
    assert "hold" in report["failure"]["message"]


def _run_success(
    trial: Any,
    *,
    reference_update: dict[str, object] | None = None,
) -> dict[str, Any]:
    success = trial.kind is not study.TrialKind.HOLD
    outcome: dict[str, Any] = {"status": "scored", "object_success": success}
    if trial.kind is study.TrialKind.REFERENCE:
        outcome["reference_success"] = True
        episode = trial.output_dir / "episode.npz"
        manifest = trial.output_dir / "episode.json"
        episode.parent.mkdir(parents=True, exist_ok=True)
        episode.write_bytes(f"episode:{trial.seed}".encode("utf-8"))
        manifest.write_text(f"manifest:{trial.seed}", encoding="utf-8")
        outcome["episode"] = {
            "npz_path": str(episode),
            "npz_sha256": study.file_sha256(episode),
            "manifest_path": str(manifest),
            "manifest_sha256": study.file_sha256(manifest),
        }
        if reference_update:
            if reference_update.get("drop_episode") is True:
                del outcome["episode"]
            if reference_update.get("drop_reference_success") is True:
                del outcome["reference_success"]
            for key, value in reference_update.items():
                if key.startswith("drop_"):
                    continue
                outcome[key] = value
    return outcome


def _unexpected_train(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("trainer must not run")


def _unexpected_run(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("trials must not run before config gate")
