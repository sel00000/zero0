from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from so101_wam.checkpoint import load_compact_wam_bundle
from so101_wam.dataset import load_episode
from so101_wam.deployment import file_sha256
from so101_wam.mujoco_semantic_suite import SEMANTIC_SUITE_REPORT_SCHEMA


MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
requires_mujoco = pytest.mark.skipif(
    not MUJOCO_AVAILABLE, reason="optional mujoco dependency is absent"
)


def test_robot_free_semantic_rejects_nonempty_output_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import robot_free_semantic_cli

    output_dir = tmp_path / "robot_free_semantic"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def unexpected_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("semantic episode generation must not start")

    monkeypatch.setattr(
        robot_free_semantic_cli,
        "_write_semantic_episodes",
        unexpected_run,
    )

    with pytest.raises(
        robot_free_semantic_cli.RobotFreePipelineError,
        match="must be empty",
    ):
        robot_free_semantic_cli.run_robot_free_semantic_demo(output_dir)

    assert tuple(output_dir.iterdir()) == (sentinel,)


def test_robot_free_semantic_cli_prints_runner_result_as_stdout_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from so101_wam import robot_free_semantic_cli

    result = {
        "schema_version": SEMANTIC_SUITE_REPORT_SCHEMA,
        "result": "complete",
        "mode": "mujoco",
        "robot_used": False,
    }

    def fake_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return result

    monkeypatch.setattr(
        robot_free_semantic_cli,
        "run_robot_free_semantic_demo",
        fake_run,
    )

    exit_code = robot_free_semantic_cli.main(
        ["--output-dir", str(tmp_path / "robot_free_semantic")]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == result


@requires_mujoco
def test_robot_free_semantic_e2e_publishes_real_multi_object_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from so101_wam import robot_free_semantic_cli

    output_dir = tmp_path / "robot_free_semantic"
    monkeypatch.chdir(tmp_path)

    exit_code = robot_free_semantic_cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report_path = output_dir / "semantic-suite-report.json"
    checkpoint_path = output_dir / "candidate.pt"
    training_report_path = output_dir / "candidate.training.json"
    train_dir = output_dir / "episodes" / "train"
    validation_dir = output_dir / "episodes" / "validation"

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    assert len(tuple(train_dir.glob("*.npz"))) == 2
    assert len(tuple(validation_dir.glob("*.npz"))) == 4
    assert {
        (load_episode(path).task_index, load_episode(path).task)
        for path in train_dir.glob("*.npz")
    } == {(101, "synthetic-train-reach")}
    assert {
        (load_episode(path).task_index, load_episode(path).task)
        for path in validation_dir.glob("*.npz")
    } == {
        (201, "synthetic-place-left"),
        (202, "synthetic-place-right"),
    }

    bundle = load_compact_wam_bundle(checkpoint_path)
    assert bundle.metadata["offline_trained"] is True
    assert bundle.metadata["trained"] is False
    assert bundle.metadata["deployment_ready"] is False

    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    assert training_report["data"]["train_task_count"] == 1
    assert training_report["data"]["validation_task_count"] == 2
    assert training_report["data"]["validation_tasks"] == [
        {"task_index": 201, "task": "synthetic-place-left"},
        {"task_index": 202, "task": "synthetic-place-right"},
    ]

    assert payload["schema_version"] == SEMANTIC_SUITE_REPORT_SCHEMA
    assert payload["result"] == "complete"
    assert payload["case_count"] == 2
    assert payload["distinct_object_task_count"] == 2
    assert payload["distinct_object_body_count"] == 2
    assert payload["distinct_object_physical_profile_count"] == 2
    assert payload["physical_object_diversity_observed"] is True
    assert payload["mapping_status_counts"] == {"unverified": 2}
    assert payload["independent_mapping_verified"] is False
    assert payload["summary"]["total_trial_count"] == 6
    assert payload["summary"]["scored_trial_count"] == 6
    assert payload["summary"]["execution_failure_count"] == 0
    assert payload["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert payload["training_report_sha256"] == file_sha256(training_report_path)
    assert payload["robot_used"] is False
    assert payload["semantic_heldout_success_claimed"] is False
    assert payload["prompt_causality_claimed"] is False
    assert payload["real_world_success_claimed"] is False
    assert payload["official_zero_wam_claimed"] is False

    assert [case["case_id"] for case in payload["cases"]] == [
        "block-left",
        "cylinder-right",
    ]
    assert [case["dataset_task_index"] for case in payload["cases"]] == [
        201,
        202,
    ]
    assert [case["object_body"] for case in payload["cases"]] == [
        "task_block",
        "task_cylinder",
    ]
    assert all(case["semantic_match"] == "unverified" for case in payload["cases"])

    for case in payload["cases"]:
        nested_path = output_dir / "semantic-artifacts" / case["benchmark_report"]
        assert file_sha256(nested_path) == case["benchmark_report_sha256"]
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        assert nested["prompt_task_match"] == "dataset_identity_verified"
        assert nested["checkpoint_task_split"] == "training_report_verified"
        assert nested["summary"]["trial_count"] == 3
