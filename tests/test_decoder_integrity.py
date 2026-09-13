from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest
import torch

from so101_wam.decoder_evidence import (
    artifact_ref,
    build_comparison,
    complete_run,
    freeze_protocol,
    read_json,
    study_records,
)
from so101_wam.deployment import canonical_json_sha256
from so101_wam.model import ActionDecoder
from so101_wam.training import CompactWAMTrainingConfig, train_offline_candidate
from so101_wam.training_data import EpisodeRecord


@contextmanager
def _deterministic():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def _core_digest(report: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {
            key: value
            for key, value in report.items()
            if key not in {"artifacts", "training_evidence_sha256"}
        }
    )


def _write(path: Path, value: dict[str, Any]) -> None:
    path.chmod(0o644)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _rebind_ck(path: Path, digest: str, **metadata: object) -> str:
    payload = torch.load(
        BytesIO(path.read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    payload["metadata"].update(metadata)
    payload["metadata"]["training_evidence_sha256"] = digest
    stream = BytesIO()
    torch.save(payload, stream)
    path.chmod(0o644)
    path.write_bytes(stream.getvalue())
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def complete_study(
    tmp_path_factory: pytest.TempPathFactory,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("decoder-integrity") / "study"
    with _deterministic():
        envelope = freeze_protocol(
            root,
            tuple(record.path for record in decoder_records[0]),
            tuple(record.path for record in decoder_records[1]),
            decoder_config,
        )
        slot = envelope["protocol"]["slots"][0]
        run_root = root / "runs" / slot["id"]
        train, validation = study_records(root, envelope["protocol"]["inputs"])
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
        complete_run(root, slot, envelope)
    return root, slot


def _tamper_run(
    root: Path,
    slot: dict[str, Any],
    change: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    run_root = root / "runs" / slot["id"]
    checkpoint = run_root / "candidate.pt"
    report_path = run_root / "training.json"
    status_path = run_root / "status.json"
    report = read_json(report_path)
    metadata: dict[str, Any] = {}

    change(report, metadata)

    digest = _core_digest(report)
    report["training_evidence_sha256"] = digest
    report["artifacts"]["checkpoint_sha256"] = _rebind_ck(
        checkpoint,
        digest,
        **metadata,
    )
    _write(report_path, report)

    status = read_json(status_path)
    status["checkpoint"] = artifact_ref(root, checkpoint)
    status["training_report"] = artifact_ref(root, report_path)
    _write(status_path, status)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("split", "random"),
        ("prompt_pairing", "self_asserted_pair"),
        ("sampling_strategy", "window_shuffle"),
        ("stage1", "skipped"),
        ("stage2", "skipped"),
        ("ifp_architecture", "fused_modules"),
        ("ifp_stride", 2.0),
        ("ifp_loss_weights", [1.0, 1.0]),
        ("wrist_rgb_preprocess", "center_crop"),
        ("action_target_source", "real_goal_position_commands"),
        ("action_row_zero_timing", "future_offset"),
    ],
)
def test_proto_rebind_rejected(
    tmp_path: Path,
    complete_study: tuple[Path, dict[str, Any]],
    name: str,
    value: object,
) -> None:
    root, slot = complete_study
    root = Path(shutil.copytree(root, tmp_path / "copy"))

    _tamper_run(
        root,
        slot,
        lambda report, _metadata: report["protocol"].__setitem__(name, value),
    )

    with pytest.raises(ValueError, match=name):
        build_comparison(root)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("artifact_kind", "compact_wam_deployment"),
        ("evidence_level", "real"),
    ],
)
def test_report_rebind_rejected(
    tmp_path: Path,
    complete_study: tuple[Path, dict[str, Any]],
    name: str,
    value: object,
) -> None:
    root, slot = complete_study
    root = Path(shutil.copytree(root, tmp_path / "copy"))

    _tamper_run(
        root,
        slot,
        lambda report, _metadata: report.__setitem__(name, value),
    )

    with pytest.raises(ValueError, match=name):
        build_comparison(root)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("artifact_kind", "compact_wam_deployment"),
        ("evidence_level", "real"),
        ("training_objective", "compact_future_latent_action_mse"),
        ("training_ifp_steps", 4),
        ("ifp_architecture", "fused_modules"),
        ("inference_ifp_module_present", False),
        ("action_target_source", "real_goal_position_commands"),
        ("train_task_count", 999),
        ("train_task_count", True),
        ("validation_task_count", 999),
        ("sampling_strategy", "window_shuffle"),
        ("validation_action_mse_normalized", 999.0),
    ],
)
def test_meta_rebind_rejected(
    tmp_path: Path,
    complete_study: tuple[Path, dict[str, Any]],
    name: str,
    value: object,
) -> None:
    root, slot = complete_study
    root = Path(shutil.copytree(root, tmp_path / "copy"))

    _tamper_run(
        root,
        slot,
        lambda _report, metadata: metadata.__setitem__(name, value),
    )

    with pytest.raises(ValueError, match=name):
        build_comparison(root)
