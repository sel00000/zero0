from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any

import pytest
import torch

from so101_wam import decoder_ablation
from so101_wam.decoder_evaluation import reduce_decoder_rows
from so101_wam.decoder_evidence import (
    artifact_ref,
    build_comparison,
    publish_json,
    read_json,
    verify_comparison,
)
from so101_wam.decoder_replay import _compare, replay_decoder_study
from so101_wam.deployment import canonical_json_sha256
from so101_wam.training import CompactWAMTrainingConfig
from so101_wam.training_data import EpisodeRecord


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


@pytest.fixture(scope="module")
def replay_study(
    tmp_path_factory: pytest.TempPathFactory,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
) -> Path:
    root = tmp_path_factory.mktemp("decoder-replay") / "study"

    decoder_ablation.run_decoder_study(
        root,
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    return root


def test_replay_final_stage2(replay_study: Path) -> None:
    before = _file_hashes(replay_study)

    with _deterministic():
        result = replay_decoder_study(replay_study)

    assert result["status"] == "complete"
    assert result["comparison_status"] == "complete"
    assert result["stage1_replayed"] is False
    assert result["stage2_replayed"] is True
    assert result["tolerances"] == {"rtol": 1e-5, "atol": 1e-6}
    assert result["provenance"]["current_evaluator_sha256"]
    assert result["provenance"]["current_replay_sha256"]
    assert len(result["runs"]) == 9
    assert all(run["status"] == "matched" for run in result["runs"])
    assert _file_hashes(replay_study) == before


def test_replay_rebound_metric(
    tmp_path: Path,
    replay_study: Path,
) -> None:
    root = Path(shutil.copytree(replay_study, tmp_path / "copy"))
    _rebind_stage2_metrics(root)

    assert verify_comparison(root)["status"] == "complete"
    with _deterministic(), pytest.raises(ValueError, match="numerical replay"):
        replay_decoder_study(root)


def test_replay_missing_slot(tmp_path: Path, replay_study: Path) -> None:
    root = Path(shutil.copytree(replay_study, tmp_path / "copy"))
    (root / "runs" / "seed-3-ordered_concat" / "status.json").unlink()
    comparison_path = root / "comparison.json"
    comparison_path.chmod(0o644)
    comparison_path.unlink()
    publish_json(comparison_path, build_comparison(root))

    with _deterministic(), pytest.raises(ValueError, match="complete comparison"):
        replay_decoder_study(root)


def test_replay_restores_state(replay_study: Path) -> None:
    torch.manual_seed(123)
    rng = torch.random.get_rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    replay_decoder_study(replay_study)

    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only


def test_replay_restores_on_error(tmp_path: Path, replay_study: Path) -> None:
    root = Path(shutil.copytree(replay_study, tmp_path / "copy"))
    _rebind_stage2_metrics(root)
    torch.manual_seed(321)
    rng = torch.random.get_rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    with pytest.raises(ValueError, match="numerical replay"):
        replay_decoder_study(root)

    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only


def test_compare_rejects_types() -> None:
    with pytest.raises(ValueError, match="type mismatch"):
        _compare({"count": 1}, {"count": 1.0}, path="root")


def test_compare_rejects_delta() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        _compare({"mse": 1.0}, {"mse": 1.1}, path="root")


def test_cli_replay_codes(
    tmp_path: Path,
    replay_study: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(shutil.copytree(replay_study, tmp_path / "copy"))
    monkeypatch.chdir(tmp_path)

    assert decoder_ablation.main(["replay", "--bundle", "copy"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"

    _rebind_stage2_metrics(root)

    with pytest.raises(SystemExit) as error:
        decoder_ablation.main(["replay", "--bundle", "copy"])

    assert error.value.code == 2
    assert "decoder study rejected:" in capsys.readouterr().err


def _rebind_stage2_metrics(root: Path) -> None:
    run_root = root / "runs" / "seed-3-mean_repeat_control"
    checkpoint_path = run_root / "candidate.pt"
    report_path = run_root / "training.json"
    status_path = run_root / "status.json"
    comparison_path = root / "comparison.json"
    report = read_json(report_path)
    status = read_json(status_path)
    payload = torch.load(
        BytesIO(checkpoint_path.read_bytes()),
        map_location="cpu",
        weights_only=True,
    )

    stage2 = report["decoder_audit"]["stage2"]
    changed_rows = deepcopy(stage2["rows"])
    changed_rows[0]["mse"] = float(changed_rows[0]["mse"]) + 0.01
    changed_rows[0]["mae_native"] = [
        float(value) + 0.01 for value in changed_rows[0]["mae_native"]
    ]
    stage2["rows"] = changed_rows
    stage2["summary"] = reduce_decoder_rows(changed_rows)
    digest = _core_digest(report)
    report["training_evidence_sha256"] = digest
    payload["metadata"]["training_evidence_sha256"] = digest
    checkpoint_sha = _rewrite_checkpoint(checkpoint_path, payload)
    report["artifacts"]["checkpoint_sha256"] = checkpoint_sha
    _rewrite_json(report_path, report)
    status["checkpoint"] = artifact_ref(root, checkpoint_path)
    status["training_report"] = artifact_ref(root, report_path)
    _rewrite_json(status_path, status)
    rebound = build_comparison(root)
    comparison_path.chmod(0o644)
    comparison_path.unlink()
    publish_json(comparison_path, rebound)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _core_digest(report: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {
            key: value
            for key, value in report.items()
            if key not in {"artifacts", "training_evidence_sha256"}
        }
    )


def _rewrite_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    stream = BytesIO()
    torch.save(payload, stream)
    path.chmod(0o644)
    path.write_bytes(stream.getvalue())
    return sha256(path.read_bytes()).hexdigest()


def _rewrite_json(path: Path, value: dict[str, Any]) -> None:
    path.chmod(0o644)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
