from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch

from so101_wam.semantic_pilot import main, run_semantic_pilot


def _prep_meta() -> dict[str, Any]:
    return {
        "config_path": Path("config.toml"),
        "train_dir": Path("train"),
        "validation_dir": Path("validation"),
        "suite_path": Path("suite.json"),
        "mapping_path": Path("mapping.json"),
        "config_sha256": "c" * 64,
        "suite_sha256": "d" * 64,
        "mapping_sha256": "e" * 64,
        "input_fingerprints": ["input-a", "input-b"],
        "input_artifacts": {"episodes/train/episode_000000.npz": "3" * 64},
        "train_digest": "a" * 64,
        "validation_digest": "b" * 64,
        "source_hashes": {"semantic_pilot.py": "f" * 64},
        "shared_hashes": {"pilot-protocol.json": "2" * 64},
        "runtime": {
            "threads": 1,
            "deterministic": True,
            "deterministic_warn_only": False,
            "versions": {
                "python": "test",
                "numpy": "test",
                "torch": "test",
            },
        },
        "protocol_sha256": "1" * 64,
    }


def _slot_report(seed: int, success: int = 1) -> dict[str, Any]:
    return {
        "result": "complete",
        "summary": {
            "total_trial_count": 12,
            "scored_trial_count": 11,
            "execution_failure_count": 1,
            "success_count": success,
            "object_position_error_mean_m": 0.12,
            "object_position_error_max_m": 0.2,
            "failure_counts": {"position_tolerance": 10, "safety_watchdog": 1},
        },
        "case_count": 4,
        "distinct_object_task_count": 4,
        "distinct_object_physical_profile_count": 2,
        "mapping_status_counts": {"unverified": 4},
        "independent_mapping_verified": False,
        "semantic_heldout_success_claimed": False,
        "prompt_causality_claimed": False,
        "real_world_success_claimed": False,
        "official_zero_wam_claimed": False,
        "robot_used": False,
        "checkpoint_sha256": str(seed) * 64,
        "training_report_sha256": str(seed + 1) * 64,
        "suite_report_path": f"slots/seed-{seed}/semantic-suite-report.json",
        "suite_report_sha256": str(seed + 2) * 64,
    }


def test_pilot_shared_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import semantic_pilot

    prepared: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    def prep(*args: Any, **kwargs: Any) -> dict[str, Any]:
        prepared.append({"args": args, "kwargs": kwargs})
        return _prep_meta()

    def run_slot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, "kwargs": kwargs})
        return _slot_report(int(kwargs["model_seed"]))

    monkeypatch.setattr(semantic_pilot, "_prep_inputs", prep)
    monkeypatch.setattr(semantic_pilot, "_run_slot", run_slot)
    monkeypatch.setattr(semantic_pilot, "_check_frozen", lambda prepared: None)

    report = run_semantic_pilot(tmp_path / "pilot")

    assert len(prepared) == 1
    assert [call["kwargs"]["model_seed"] for call in calls] == [3, 7, 11]
    assert {call["kwargs"]["data_seed"] for call in calls} == {7}
    assert report["result"] == "complete"
    assert report["aggregate"] is not None
    assert report["model_seeds"] == [3, 7, 11]
    assert report["data_seed"] == 7
    assert report["rollout_seeds"] == [7, 13, 29]
    assert report["policy_steps"] == 10
    assert all(slot["input_fingerprints"] == ["input-a", "input-b"] for slot in report["slots"])
    assert report["aggregate"]["slot_count"] == 3
    assert report["aggregate"]["total_trial_count"] == 36
    assert report["aggregate"]["scored_trial_count"] == 33
    assert report["aggregate"]["execution_failure_count"] == 3
    assert report["aggregate"]["success_by_seed"] == {"3": 1, "7": 1, "11": 1}
    assert report["semantic_heldout_success_claimed"] is False
    assert report["real_world_success_claimed"] is False


def test_pilot_failure_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import semantic_pilot

    def prep(*args: Any, **kwargs: Any) -> dict[str, Any]:
        prepared = _prep_meta()
        prepared["input_fingerprints"] = ["shared"]
        return prepared

    def fail_mid(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs["model_seed"] == 7:
            raise RuntimeError("training failed")
        return _slot_report(int(kwargs["model_seed"]))

    monkeypatch.setattr(semantic_pilot, "_prep_inputs", prep)
    monkeypatch.setattr(semantic_pilot, "_run_slot", fail_mid)
    monkeypatch.setattr(semantic_pilot, "_check_frozen", lambda prepared: None)

    report = run_semantic_pilot(tmp_path / "pilot")

    assert report["result"] == "incomplete"
    assert report["aggregate"] is None
    assert [slot["status"] for slot in report["slots"]] == [
        "complete",
        "failed",
        "complete",
    ]
    assert "training failed" in report["slots"][1]["reason"]


def test_pilot_interrupt_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import semantic_pilot

    def prep(*args: Any, **kwargs: Any) -> dict[str, Any]:
        prepared = _prep_meta()
        prepared["input_fingerprints"] = ["shared"]
        return prepared

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs["model_seed"] == 7:
            raise KeyboardInterrupt
        return _slot_report(int(kwargs["model_seed"]))

    monkeypatch.setattr(semantic_pilot, "_prep_inputs", prep)
    monkeypatch.setattr(semantic_pilot, "_run_slot", interrupt)
    monkeypatch.setattr(semantic_pilot, "_check_frozen", lambda prepared: None)

    report = run_semantic_pilot(tmp_path / "pilot")

    assert report["result"] == "incomplete"
    assert report["aggregate"] is None
    assert [slot["status"] for slot in report["slots"]] == [
        "complete",
        "failed",
        "not_attempted",
    ]
    assert "interrupted" in report["slots"][1]["reason"]
    assert report["slots"][2]["reason"] == report["slots"][1]["reason"]


def test_pilot_fixed_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed model and data seeds"):
        run_semantic_pilot(tmp_path / "seeds", model_seeds=(3, 7))
    with pytest.raises(ValueError, match="fixed model and data seeds"):
        run_semantic_pilot(tmp_path / "float-seeds", model_seeds=(3.0, 7.0, 11.0))
    with pytest.raises(ValueError, match="fixed model and data seeds"):
        run_semantic_pilot(tmp_path / "float-data", data_seed=7.0)

    with pytest.raises(ValueError, match="device='cpu'"):
        run_semantic_pilot(tmp_path / "device", device="cuda")


def test_pilot_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import semantic_pilot

    output = tmp_path / "pilot"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    def unexpected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("pilot prep must not start")

    monkeypatch.setattr(semantic_pilot, "_prep_inputs", unexpected)

    with pytest.raises(semantic_pilot.RobotFreePipelineError, match="must be empty"):
        run_semantic_pilot(output)


def test_pilot_restores_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import semantic_pilot

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    before_random = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state()

    def mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        random.random()
        np.random.random()
        torch.rand(1)
        return _slot_report(int(kwargs["model_seed"]))

    monkeypatch.setattr(semantic_pilot, "_prep_inputs", lambda *a, **k: _prep_meta())
    monkeypatch.setattr(semantic_pilot, "_run_slot", mutate)
    monkeypatch.setattr(semantic_pilot, "_check_frozen", lambda prepared: None)

    run_semantic_pilot(tmp_path / "pilot")

    assert random.getstate() == before_random
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)


def test_pilot_drift_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import semantic_pilot

    checks = 0

    def check(prepared: dict[str, Any]) -> None:
        nonlocal checks
        del prepared
        checks += 1
        if checks == 2:
            raise semantic_pilot.SemanticPilotError("source drift")

    monkeypatch.setattr(semantic_pilot, "_prep_inputs", lambda *a, **k: _prep_meta())
    monkeypatch.setattr(
        semantic_pilot,
        "_run_slot",
        lambda *a, **k: _slot_report(int(k["model_seed"])),
    )
    monkeypatch.setattr(semantic_pilot, "_check_frozen", check)

    report = run_semantic_pilot(tmp_path / "pilot")

    assert report["aggregate"] is None
    assert [slot["status"] for slot in report["slots"]] == [
        "failed",
        "not_attempted",
        "not_attempted",
    ]
    assert "source drift" in report["slots"][0]["reason"]
    assert all(slot["reason"] == "source drift" for slot in report["slots"])


def test_pilot_input_loss_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import semantic_pilot

    calls = 0
    prepared = _prep_meta()
    prepared["train_dir"] = tmp_path / "missing-train"
    prepared["validation_dir"] = tmp_path / "missing-validation"

    def run_slot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _slot_report(int(kwargs["model_seed"]))

    monkeypatch.setattr(semantic_pilot, "_run_slot", run_slot)

    slots = semantic_pilot._run_slots(
        tmp_path,
        prepared=prepared,
        model_seeds=(3, 7, 11),
        data_seed=7,
        device="cpu",
    )

    assert calls == 0
    assert [slot["status"] for slot in slots] == [
        "failed",
        "not_attempted",
        "not_attempted",
    ]
    assert "frozen input check failed" in slots[0]["reason"]


def test_pilot_weighted_errors() -> None:
    from so101_wam import semantic_pilot

    slots = []
    for seed, scored, error in ((3, 1, 0.3), (7, 3, 0.1)):
        report = _slot_report(seed)
        report["summary"]["scored_trial_count"] = scored
        report["summary"]["object_position_error_mean_m"] = error
        slots.append(semantic_pilot._slot(seed, seed, report=report))

    aggregate = semantic_pilot._aggregate(slots)

    assert aggregate["object_position_error_mean_m"] == pytest.approx(0.15)
    assert aggregate["macro_seed_error_mean_m"] == pytest.approx(0.2)


def test_prep_writes_four_cases(tmp_path: Path) -> None:
    from so101_wam import semantic_pilot

    output = tmp_path / "pilot"
    output.mkdir()

    with semantic_pilot._fixed_runtime():
        prepared = semantic_pilot._prep_inputs(output)

    validation_dir = output / "shared" / "episodes" / "validation"
    suite = json.loads(Path(prepared["suite_path"]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(prepared["mapping_path"]).read_text(encoding="utf-8"))

    assert len(tuple(validation_dir.glob("*.npz"))) == 8
    assert [case["case_id"] for case in suite["cases"]] == [
        "block-left",
        "block-right",
        "cylinder-left",
        "cylinder-right",
    ]
    assert [item["dataset_task"]["task_index"] for item in mapping["mappings"]] == [
        201,
        202,
        203,
        204,
    ]
    assert prepared["shared_hashes"] == semantic_pilot._file_tree_hashes(
        output / "shared"
    )


def test_pilot_cli_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from so101_wam import semantic_pilot

    payload = {
        "schema_version": "so101_wam.semantic_pilot.v1",
        "result": "complete",
        "slots": [],
    }

    def fake_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(semantic_pilot, "run_semantic_pilot", fake_run)

    assert main(["--output-dir", str(tmp_path / "pilot")]) == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_pilot_cli_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from so101_wam import semantic_pilot

    payload = {
        "schema_version": "so101_wam.semantic_pilot.v1",
        "result": "incomplete",
        "slots": [],
    }

    monkeypatch.setattr(semantic_pilot, "run_semantic_pilot", lambda *a, **k: payload)

    assert main(["--output-dir", str(tmp_path / "pilot")]) == 1
    assert json.loads(capsys.readouterr().out) == payload
