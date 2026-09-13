from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pytest
import torch

from so101_wam import decoder_ablation
from so101_wam.checkpoint import load_compact_wam_bytes
from so101_wam.dataset import save_episode
from so101_wam.decoder_evaluation import named_tensor_hashes
from so101_wam.decoder_evidence import (
    STUDY_MODES,
    STUDY_SCHEMA,
    STUDY_SEEDS,
    artifact_ref,
    build_comparison,
    complete_run,
    freeze_protocol,
    publish_json,
    read_json,
    resolve_artifact,
    study_protocol,
    study_records,
    verify_comparison,
    verify_runtime,
)
from so101_wam.deployment import canonical_json_sha256
from so101_wam.model import ActionDecoder
from so101_wam.training import CompactWAMTrainingConfig, train_offline_candidate
from so101_wam.training_data import EpisodeRecord, TrainingDataError


@contextmanager
def _deterministic():
    threads = torch.get_num_threads()
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def _inputs(records: tuple[EpisodeRecord, ...]) -> tuple[Path, ...]:
    return tuple(record.path for record in records)


def _freeze(
    tmp_path: Path,
    records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    config: CompactWAMTrainingConfig,
) -> dict[str, Any]:
    return freeze_protocol(
        tmp_path / "study",
        _inputs(records[0]),
        _inputs(records[1]),
        config,
    )


def _core_digest(report: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {
            key: value
            for key, value in report.items()
            if key not in {"artifacts", "training_evidence_sha256"}
        }
    )


@pytest.mark.parametrize("clone_scope", ["cross_split", "within_group"])
def test_freeze_rejects_clones(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    clone_scope: str,
) -> None:
    train, validation = decoder_records
    clone_dir = tmp_path / "clones"
    if clone_scope == "cross_split":
        for record in train:
            save_episode(
                replace(record.data, task="relabelled", task_index=99),
                clone_dir,
            )
        train_inputs = _inputs(train)
        validation_inputs = (clone_dir,)
    else:
        for record in train:
            save_episode(
                replace(train[0].data, episode_index=record.data.episode_index),
                clone_dir,
            )
        train_inputs = (clone_dir,)
        validation_inputs = _inputs(validation)

    root = tmp_path / "study"
    with _deterministic(), pytest.raises(TrainingDataError, match="content"):
        freeze_protocol(root, train_inputs, validation_inputs, decoder_config)

    assert not (root / "protocol.json").exists()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.chmod(0o644)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _rebind_checkpoint(path: Path, digest: str) -> str:
    payload = torch.load(BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
    payload["metadata"]["training_evidence_sha256"] = digest
    stream = BytesIO()
    torch.save(payload, stream)
    path.chmod(0o644)
    path.write_bytes(stream.getvalue())
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def completed_study(
    tmp_path_factory: pytest.TempPathFactory,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> Path:
    root = tmp_path_factory.mktemp("decoder-completed") / "study"

    decoder_ablation.run_decoder_study(
        root,
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    return root


def _rewrite(path: Path, value: dict[str, Any]) -> None:
    path.chmod(0o644)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _rebind(root: Path, mutation: str) -> None:
    name = "seed-3-mean_repeat_control"
    directory = root / "runs" / name
    checkpoint = directory / "candidate.pt"
    report_path = directory / "training.json"
    status_path = directory / "status.json"
    report = read_json(report_path)
    status = read_json(status_path)
    payload = torch.load(
        BytesIO(checkpoint.read_bytes()),
        map_location="cpu",
        weights_only=True,
    )

    if mutation == "mode":
        payload["architecture"]["action_decoder"] = ActionDecoder.ORDERED_CONCAT.value
    elif mutation == "protocol":
        payload["metadata"]["study_sha256"] = "b" * 64
    elif mutation == "config":
        report["optimization"]["learning_rate"] = 0.02
    elif mutation == "shared":
        audit = report["decoder_audit"]
        key = next(
            key for key in audit["initial_tensors"] if key.startswith("encoder.")
        )
        audit["initial_tensors"][key] = "b" * 64
        audit["stage1_tensors"][key] = "b" * 64
    elif mutation == "schedule":
        report["decoder_audit"]["stage_schedule_sha256"]["stage1"] = "b" * 64
    elif mutation == "summary":
        report["decoder_audit"]["stage2"]["summary"]["task_macro_mse"] += 1
    elif mutation == "nonfinite":
        key = next(
            name for name in payload["state_dict"] if name.startswith("encoder.")
        )
        stream = BytesIO()
        torch.save(payload, stream)
        model = load_compact_wam_bytes(stream.getvalue(), device="cpu").model
        payload["state_dict"][key].flatten()[0] = float("nan")
        model.load_state_dict(payload["state_dict"], strict=True)
        report["decoder_audit"]["final_tensors"] = named_tensor_hashes(model)

    digest = _core_digest(report)
    report["training_evidence_sha256"] = digest
    payload["metadata"]["training_evidence_sha256"] = digest
    stream = BytesIO()
    torch.save(payload, stream)
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(stream.getvalue())
    report["artifacts"]["checkpoint_sha256"] = artifact_ref(root, checkpoint)[
        "sha256"
    ]
    _rewrite(report_path, report)
    status["checkpoint"] = artifact_ref(root, checkpoint)
    status["training_report"] = artifact_ref(root, report_path)
    _rewrite(status_path, status)


def test_json_is_immutable(tmp_path: Path) -> None:
    target = tmp_path / "doc.json"

    publish_json(target, {"ok": 1})

    assert target.stat().st_mode & 0o777 == 0o444
    assert target.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        publish_json(target, {"ok": 2})


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a": 1, "a": 2}\n',
        b'{"a": NaN}\n',
        b'{"a": Infinity}\n',
        b'{"a": 1e999}\n',
        b"[1]\n",
    ],
)
def test_json_rejects_bad(raw: bytes, tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_bytes(raw)

    with pytest.raises(ValueError):
        read_json(target)


def test_artifact_path_checks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    (root / "file.txt").write_text("x", encoding="utf-8")
    os.symlink(outside, root / "link.txt")

    with pytest.raises(ValueError):
        resolve_artifact(root, "../outside.txt")
    with pytest.raises(ValueError):
        resolve_artifact(root, "link.txt")
    with pytest.raises(ValueError):
        resolve_artifact(root, "a/./b")
    with pytest.raises(ValueError):
        resolve_artifact(root, "a//b")
    assert resolve_artifact(root, "future/status.json") == root / "future/status.json"


def test_symlink_root_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    root = tmp_path / "link-root"
    os.symlink(real, root)

    with pytest.raises(ValueError):
        resolve_artifact(root, "file.txt")


def test_ref_rejects_symlink_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "file.txt").write_text("x", encoding="utf-8")
    os.symlink(real, root / "alias")

    with pytest.raises(ValueError):
        artifact_ref(root, root / "alias" / "file.txt")


def test_freeze_accepts_dirs(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    envelope = freeze_protocol(
        tmp_path / "study",
        (decoder_records[0][0].path.parent,),
        (decoder_records[1][0].path.parent,),
        decoder_config,
    )

    assert envelope["protocol"]["data"]["train_episode_count"] == 2


def test_relative_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    monkeypatch.chdir(tmp_path)

    envelope = freeze_protocol(
        "study",
        _inputs(decoder_records[0]),
        _inputs(decoder_records[1]),
        decoder_config,
    )

    assert envelope["protocol"]["inputs"][0]["path"] == (
        "inputs/train/episode_000000.npz"
    )


def test_freeze_round_trip(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    with _deterministic():
        first = _freeze(tmp_path, decoder_records, decoder_config)
        second = study_protocol(tmp_path / "study")
        runtime_ok = verify_runtime(first["protocol"])

    protocol = first["protocol"]
    assert first == second
    assert protocol["schema_version"] == STUDY_SCHEMA
    assert protocol["seeds"] == list(STUDY_SEEDS)
    assert protocol["modes"] == [mode.value for mode in STUDY_MODES]
    assert len(protocol["slots"]) == 9
    assert protocol["runtime"]["device"] == "cpu"
    assert protocol["runtime"]["threads"] == 1
    assert protocol["runtime"]["deterministic"] is True
    assert torch.are_deterministic_algorithms_enabled() is False
    assert protocol["real_output_authorized"] is False
    assert protocol["data"]["validation_task_count"] == 1
    assert len(protocol["data"]["validation_tasks"]) == protocol["data"][
        "validation_window_count"
    ]
    assert all(set(ref) == {"path", "sha256"} for ref in protocol["inputs"])
    assert all(ref["path"].startswith("inputs/") for ref in protocol["inputs"])
    assert first["protocol_sha256"] == canonical_json_sha256(protocol)
    assert second["protocol_artifact"] == artifact_ref(
        tmp_path / "study",
        tmp_path / "study" / "protocol.json",
    )
    assert runtime_ok is None


def test_freeze_rejects_f1(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    config = replace(decoder_config, future_steps=1)

    with pytest.raises(ValueError, match="future_steps"):
        _freeze(tmp_path, decoder_records, config)


def test_tamper_rejected(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    _freeze(tmp_path, decoder_records, decoder_config)
    target = tmp_path / "study" / "inputs" / "train" / "episode_000000.npz"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b"x")

    with pytest.raises(ValueError, match="sha256"):
        study_protocol(tmp_path / "study")


def test_bool_seed_rejected(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    envelope = _freeze(tmp_path, decoder_records, decoder_config)
    envelope["protocol"]["diagnostic_seed"] = False
    payload = {
        "protocol": envelope["protocol"],
        "protocol_sha256": canonical_json_sha256(envelope["protocol"]),
    }
    target = tmp_path / "study" / "protocol.json"
    target.chmod(0o644)
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic seed"):
        study_protocol(tmp_path / "study")


def test_upper_revision_bad(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    envelope = _freeze(tmp_path, decoder_records, decoder_config)
    envelope["protocol"]["source_revision"] = envelope["protocol"][
        "source_revision"
    ].upper()
    payload = {
        "protocol": envelope["protocol"],
        "protocol_sha256": canonical_json_sha256(envelope["protocol"]),
    }
    target = tmp_path / "study" / "protocol.json"
    target.chmod(0o644)
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source revision"):
        study_protocol(tmp_path / "study")


def test_records_use_frozen_paths(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    envelope = _freeze(tmp_path, decoder_records, decoder_config)

    train, validation = study_records(
        tmp_path / "study",
        envelope["protocol"]["inputs"],
    )

    assert all(record.path.is_relative_to(tmp_path / "study") for record in train)
    assert all(record.path.is_relative_to(tmp_path / "study") for record in validation)
    assert train[0].data.fingerprint == decoder_records[0][0].data.fingerprint


def test_alias_rejected(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    _freeze(tmp_path, decoder_records, decoder_config)
    source = tmp_path / "study" / "inputs" / "train" / "episode_000000.npz"
    os.link(source, tmp_path / "study" / "inputs" / "train" / "alias.npz")

    with pytest.raises(ValueError, match="inventory"):
        study_protocol(tmp_path / "study")


def test_partial_comparison(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    with _deterministic():
        envelope = _freeze(tmp_path, decoder_records, decoder_config)
        slot = envelope["protocol"]["slots"][0]
        run_root = tmp_path / "study" / "runs" / slot["id"]
        run_root.mkdir(parents=True)
        train, validation = study_records(
            tmp_path / "study",
            envelope["protocol"]["inputs"],
        )
        config = replace(
            decoder_config,
            seed=slot["seed"],
            action_decoder=ActionDecoder(slot["mode"]),
        )
        train_offline_candidate(
            train,
            validation,
            checkpoint_path=run_root / "candidate.pt",
            report_path=run_root / "training.json",
            checkpoint_id=slot["id"],
            config=config,
            study_sha256=envelope["protocol_sha256"],
            device="cpu",
        )
        publish_json(
            run_root / "status.json",
            {
                "id": slot["id"],
                "seed": slot["seed"],
                "mode": slot["mode"],
                "status": "complete",
                "reason": None,
                "checkpoint": artifact_ref(tmp_path / "study", run_root / "candidate.pt"),
                "training_report": artifact_ref(
                    tmp_path / "study",
                    run_root / "training.json",
                ),
            },
        )

    comparison = build_comparison(tmp_path / "study")
    complete = [run for run in comparison["runs"] if run["status"] == "complete"]

    assert len(complete) == 1
    assert comparison["summary"] is None
    assert comparison["status"] == "incomplete"
    assert set(complete[0]["metrics"]) == {
        "stage1",
        "stage2",
        "seconds",
        "head_parameters",
        "model_parameters",
    }

    report = run_root / "training.json"
    report.chmod(0o644)
    report.write_bytes(report.read_bytes() + b" \n")
    with pytest.raises(ValueError, match="sha256"):
        build_comparison(tmp_path / "study")


def test_relative_compare_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    monkeypatch.chdir(tmp_path)
    with _deterministic():
        envelope = freeze_protocol(
            "study",
            _inputs(decoder_records[0]),
            _inputs(decoder_records[1]),
            decoder_config,
        )
        slot = envelope["protocol"]["slots"][0]
        run_root = Path("study") / "runs" / slot["id"]
        run_root.mkdir(parents=True)
        train, validation = study_records("study", envelope["protocol"]["inputs"])
        train_offline_candidate(
            train,
            validation,
            checkpoint_path=run_root / "candidate.pt",
            report_path=run_root / "training.json",
            checkpoint_id=slot["id"],
            config=replace(
                decoder_config,
                seed=slot["seed"],
                action_decoder=ActionDecoder(slot["mode"]),
            ),
            study_sha256=envelope["protocol_sha256"],
            device="cpu",
        )
        publish_json(
            run_root / "status.json",
            {
                "id": slot["id"],
                "seed": slot["seed"],
                "mode": slot["mode"],
                "status": "complete",
                "reason": None,
                "checkpoint": artifact_ref("study", run_root / "candidate.pt"),
                "training_report": artifact_ref("study", run_root / "training.json"),
            },
        )

    comparison = build_comparison("study")
    publish_json(Path("study") / "comparison.json", comparison)

    assert verify_comparison("study") == comparison


def test_bool_config_bad(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    with _deterministic():
        envelope = _freeze(tmp_path, decoder_records, decoder_config)
        slot = envelope["protocol"]["slots"][0]
        run_root = tmp_path / "study" / "runs" / slot["id"]
        run_root.mkdir(parents=True)
        train, validation = study_records(
            tmp_path / "study",
            envelope["protocol"]["inputs"],
        )
        train_offline_candidate(
            train,
            validation,
            checkpoint_path=run_root / "candidate.pt",
            report_path=run_root / "training.json",
            checkpoint_id=slot["id"],
            config=replace(decoder_config, seed=slot["seed"]),
            study_sha256=envelope["protocol_sha256"],
            device="cpu",
        )

    report_path = run_root / "training.json"
    report = read_json(report_path)
    report["optimization"]["stage1_steps"] = True
    report["training_evidence_sha256"] = _core_digest(report)
    checkpoint_hash = _rebind_checkpoint(
        run_root / "candidate.pt",
        report["training_evidence_sha256"],
    )
    report["artifacts"]["checkpoint_sha256"] = checkpoint_hash
    _write_report(report_path, report)
    publish_json(
        run_root / "status.json",
        {
            "id": slot["id"],
            "seed": slot["seed"],
            "mode": slot["mode"],
            "status": "complete",
            "reason": None,
            "checkpoint": artifact_ref(tmp_path / "study", run_root / "candidate.pt"),
            "training_report": artifact_ref(
                tmp_path / "study",
                run_root / "training.json",
            ),
        },
    )

    with pytest.raises(ValueError, match="stage1_steps"):
        build_comparison(tmp_path / "study")


def test_complete_run_status(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    with _deterministic():
        envelope = _freeze(tmp_path, decoder_records, decoder_config)
        slot = envelope["protocol"]["slots"][0]
        run_root = tmp_path / "study" / "runs" / slot["id"]
        train, validation = study_records(
            tmp_path / "study",
            envelope["protocol"]["inputs"],
        )
        train_offline_candidate(
            train,
            validation,
            checkpoint_path=run_root / "candidate.pt",
            report_path=run_root / "training.json",
            checkpoint_id=slot["id"],
            config=replace(
                decoder_config,
                seed=slot["seed"],
                action_decoder=ActionDecoder(slot["mode"]),
            ),
            study_sha256=envelope["protocol_sha256"],
            device="cpu",
        )
        complete_run(tmp_path / "study", slot, envelope)

    status_path = run_root / "status.json"
    status = read_json(status_path)
    assert set(status) == {
        "id",
        "seed",
        "mode",
        "status",
        "reason",
        "checkpoint",
        "training_report",
    }
    assert status["status"] == "complete"
    assert status_path.stat().st_mode & 0o777 == 0o444
    assert (run_root / "candidate.pt").stat().st_mode & 0o777 == 0o444
    assert (run_root / "training.json").stat().st_mode & 0o777 == 0o444


def test_relative_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> None:
    monkeypatch.chdir(tmp_path)

    with _deterministic():
        envelope = freeze_protocol(
            "study",
            _inputs(decoder_records[0]),
            _inputs(decoder_records[1]),
            decoder_config,
        )
        slot = envelope["protocol"]["slots"][0]
        run_root = Path("study") / "runs" / slot["id"]
        train, validation = study_records("study", envelope["protocol"]["inputs"])
        train_offline_candidate(
            train,
            validation,
            checkpoint_path=run_root / "candidate.pt",
            report_path=run_root / "training.json",
            checkpoint_id=slot["id"],
            config=replace(
                decoder_config,
                seed=slot["seed"],
                action_decoder=ActionDecoder(slot["mode"]),
            ),
            study_sha256=envelope["protocol_sha256"],
            device="cpu",
        )
        complete_run("study", slot, envelope)

    assert build_comparison("study")["runs"][0]["status"] == "complete"


def test_nine_run_completion(completed_study: Path) -> None:
    result = verify_comparison(completed_study)

    assert result["status"] == "complete"
    assert len(result["runs"]) == 9
    assert all(run["status"] == "complete" for run in result["runs"])
    paired = result["summary"]["paired_differences"]
    assert len(paired) == 3
    assert result["summary"]["primary"] == (
        "ordered_concat_minus_mean_repeat_control"
    )
    assert all(set(value["per_seed"]) == {"3", "7", "11"} for value in paired.values())
    assert all(value["sample_std"] >= 0 for value in paired.values())


def test_failures_keep_all_slots(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(decoder_ablation, "train_offline_candidate", fail)
    root = tmp_path / "failed"

    result = decoder_ablation.run_decoder_study(
        root,
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert calls == 9
    assert result["status"] == "incomplete"
    assert result["summary"] is None
    assert len(result["runs"]) == 9
    assert all(run["status"] == "failed" for run in result["runs"])
    assert all(run["metrics"] is None for run in result["runs"])
    assert verify_comparison(root) == result


def test_interrupt_slots(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def interrupt(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(decoder_ablation, "train_offline_candidate", interrupt)
    root = tmp_path / "interrupted"

    result = decoder_ablation.run_decoder_study(
        root,
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert calls == 1
    assert [run["status"] for run in result["runs"]] == ["failed"] + [
        "not_attempted"
    ] * 8
    assert all(run["metrics"] is None for run in result["runs"])
    assert verify_comparison(root) == result


def test_assertion_failure_runs(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("asserted failure")

    monkeypatch.setattr(decoder_ablation, "train_offline_candidate", fail)

    result = decoder_ablation.run_decoder_study(
        tmp_path / "assertion",
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert calls == 9
    assert [run["status"] for run in result["runs"]] == ["failed"] * 9
    assert all(run["metrics"] is None for run in result["runs"])


def test_cli_verify_codes(
    tmp_path: Path,
    completed_study: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(shutil.copytree(completed_study, tmp_path / "copy"))
    monkeypatch.chdir(tmp_path)

    assert decoder_ablation.main(["verify", "--bundle", "copy"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"

    (root / "runs/seed-3-ordered_concat/status.json").unlink()
    comparison = build_comparison(root)
    (root / "comparison.json").chmod(0o644)
    (root / "comparison.json").unlink()
    publish_json(root / "comparison.json", comparison)

    assert decoder_ablation.main(["verify", "--bundle", "copy"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"


def test_cli_rejection_format(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        decoder_ablation.main(["verify", "--bundle", "missing"])

    assert error.value.code == 2
    assert "decoder study rejected:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "artifact",
    ["candidate.pt", "training.json"],
)
def test_changed_bytes_rejected(
    tmp_path: Path,
    completed_study: Path,
    artifact: str,
) -> None:
    root = Path(shutil.copytree(completed_study, tmp_path / "copy"))
    path = root / "runs/seed-3-ordered_concat" / artifact
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="sha256|hash"):
        build_comparison(root)


@pytest.mark.parametrize(
    "mutation",
    ["mode", "protocol", "config", "shared", "schedule", "summary", "nonfinite"],
)
def test_rebound_bad_evidence(
    tmp_path: Path,
    completed_study: Path,
    mutation: str,
) -> None:
    root = Path(shutil.copytree(completed_study, tmp_path / "copy"))

    _rebind(root, mutation)

    with pytest.raises(ValueError):
        build_comparison(root)


def test_false_complete_rejected(tmp_path: Path, completed_study: Path) -> None:
    root = Path(shutil.copytree(completed_study, tmp_path / "copy"))
    (root / "runs/seed-3-ordered_concat/status.json").unlink()
    result = build_comparison(root)

    assert result["status"] == "incomplete"
    assert result["summary"] is None
    assert result["runs"][2]["metrics"] is None
    with pytest.raises(ValueError, match="falsely complete"):
        verify_comparison(root)


def test_report_nan_rejected(tmp_path: Path, completed_study: Path) -> None:
    root = Path(shutil.copytree(completed_study, tmp_path / "copy"))
    report_path = root / "runs/seed-3-ordered_concat/training.json"
    report = read_json(report_path)
    report["validation"]["action_mse_normalized"] = float("nan")
    _rewrite(report_path, report)
    status_path = report_path.with_name("status.json")
    status = read_json(status_path)
    status["training_report"] = artifact_ref(root, report_path)
    _rewrite(status_path, status)

    with pytest.raises(ValueError):
        build_comparison(root)


def test_protocol_space_rejects(
    tmp_path: Path,
    completed_study: Path,
) -> None:
    root = Path(shutil.copytree(completed_study, tmp_path / "copy"))
    protocol_path = root / "protocol.json"
    protocol_path.chmod(0o644)
    protocol_path.write_bytes(protocol_path.read_bytes() + b" \n")

    with pytest.raises(ValueError):
        verify_comparison(root)


def test_relocated_bundle_valid(tmp_path: Path, completed_study: Path) -> None:
    root = Path(shutil.copytree(completed_study, tmp_path / "relocated"))

    assert verify_comparison(root)["status"] == "complete"


def test_precondition_stops(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_train = decoder_ablation.train_offline_candidate

    def mutate_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        result = real_train(*args, **kwargs)
        protocol_path = tmp_path / "precondition" / "protocol.json"
        protocol_path.chmod(0o644)
        protocol_path.write_bytes(protocol_path.read_bytes() + b" \n")
        return result

    monkeypatch.setattr(decoder_ablation, "train_offline_candidate", mutate_once)

    result = decoder_ablation.run_decoder_study(
        tmp_path / "precondition",
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert calls == 1
    assert [run["status"] for run in result["runs"]] == ["failed"] + [
        "not_attempted"
    ] * 8
    assert result["summary"] is None
    assert verify_comparison(tmp_path / "precondition") == result


def test_final_precondition_stops(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_train = decoder_ablation.train_offline_candidate

    def mutate_last(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        result = real_train(*args, **kwargs)
        if calls == 9:
            protocol_path = tmp_path / "final-precondition" / "protocol.json"
            protocol_path.chmod(0o644)
            protocol_path.write_bytes(protocol_path.read_bytes() + b" \n")
        return result

    monkeypatch.setattr(decoder_ablation, "train_offline_candidate", mutate_last)

    result = decoder_ablation.run_decoder_study(
        tmp_path / "final-precondition",
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert calls == 9
    assert [run["status"] for run in result["runs"]] == ["complete"] * 8 + ["failed"]
    assert result["summary"] is None
    assert verify_comparison(tmp_path / "final-precondition") == result


def test_precondition_error_marks(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_protocol = decoder_ablation.study_protocol

    def fail_once(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AssertionError("guard failed")
        return real_protocol(root)

    monkeypatch.setattr(decoder_ablation, "study_protocol", fail_once)

    result = decoder_ablation.run_decoder_study(
        tmp_path / "guard",
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert calls == 1
    assert [run["status"] for run in result["runs"]] == ["not_attempted"] * 9
    assert all("AssertionError" in run["reason"] for run in result["runs"])


def test_freeze_error_restores(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("freeze failed")

    monkeypatch.setattr(decoder_ablation, "freeze_protocol", fail)

    with pytest.raises(RuntimeError, match="freeze failed"):
        decoder_ablation.run_decoder_study(
            tmp_path / "restore",
            train_inputs=(decoder_records[0][0].path.parent,),
            validation_inputs=(decoder_records[1][0].path.parent,),
            config=decoder_config,
        )

    assert torch.get_num_threads() == old_threads
    assert torch.are_deterministic_algorithms_enabled() is old_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is old_warn
