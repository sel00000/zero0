from __future__ import annotations

from pathlib import Path

import pytest
import torch

from so101_wam import decoder_ablation
from so101_wam.decoder_evidence import STUDY_MODES, STUDY_SEEDS, verify_comparison
from so101_wam.training import CompactWAMTrainingConfig
from so101_wam.training_data import EpisodeRecord


RUN_COUNT = len(STUDY_SEEDS) * len(STUDY_MODES)


@pytest.mark.parametrize("changed_slot", [1, RUN_COUNT])
@pytest.mark.parametrize("setting", ["threads", "deterministic", "warn_only"])
def test_post_train_runtime_drift(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
    changed_slot: int,
    setting: str,
) -> None:
    calls = 0
    real_train = decoder_ablation.train_offline_candidate
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def change_runtime(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        result = real_train(*args, **kwargs)
        if calls != changed_slot:
            return result

        # Change live settings after valid artifacts exist, before publication.
        if setting == "threads":
            torch.set_num_threads(2)
        elif setting == "deterministic":
            torch.use_deterministic_algorithms(False)
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)
        return result

    monkeypatch.setattr(decoder_ablation, "train_offline_candidate", change_runtime)
    root = tmp_path / "runtime-drift"

    result = decoder_ablation.run_decoder_study(
        root,
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert calls == changed_slot
    assert [run["status"] for run in result["runs"]] == (
        ["complete"] * (changed_slot - 1)
        + ["failed"]
        + ["not_attempted"] * (RUN_COUNT - changed_slot)
    )
    assert "mismatch" in result["runs"][changed_slot - 1]["reason"]
    assert result["status"] == "incomplete"
    assert result["summary"] is None
    assert verify_comparison(root) == result
    assert torch.get_num_threads() == old_threads
    assert torch.are_deterministic_algorithms_enabled() is old_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is old_warn


@pytest.mark.parametrize("boundary", ["study_protocol", "verify_runtime", "study_records"])
def test_preflight_interrupt(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    calls = 0
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    def unexpected_train(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("interrupted preflight must not train")

    monkeypatch.setattr(decoder_ablation, boundary, interrupt)
    monkeypatch.setattr(decoder_ablation, "train_offline_candidate", unexpected_train)
    root = tmp_path / "preflight-interrupt"

    try:
        result = decoder_ablation.run_decoder_study(
            root,
            train_inputs=(decoder_records[0][0].path.parent,),
            validation_inputs=(decoder_records[1][0].path.parent,),
            config=decoder_config,
        )
    except KeyboardInterrupt:
        pytest.fail("preflight interrupt lost the frozen study's slot statuses")

    assert calls == 0
    assert [run["status"] for run in result["runs"]] == ["not_attempted"] * RUN_COUNT
    assert all(run["reason"] == decoder_ablation.INTERRUPT_REASON for run in result["runs"])
    assert result["status"] == "incomplete"
    assert result["summary"] is None
    assert verify_comparison(root) == result
    assert torch.get_num_threads() == old_threads
    assert torch.are_deterministic_algorithms_enabled() is old_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is old_warn


def test_post_guard_failure_stops(
    tmp_path: Path,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config: CompactWAMTrainingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = 0
    real_verify = decoder_ablation.verify_runtime

    def reject_post_check(protocol: dict[str, object]) -> None:
        nonlocal checks
        checks += 1
        real_verify(protocol)
        if checks == 2:
            raise ValueError("live source inventory mismatch")

    monkeypatch.setattr(decoder_ablation, "verify_runtime", reject_post_check)
    root = tmp_path / "post-guard-failure"

    result = decoder_ablation.run_decoder_study(
        root,
        train_inputs=(decoder_records[0][0].path.parent,),
        validation_inputs=(decoder_records[1][0].path.parent,),
        config=decoder_config,
    )

    assert checks == 2
    assert [run["status"] for run in result["runs"]] == (
        ["failed"] + ["not_attempted"] * (RUN_COUNT - 1)
    )
    assert result["summary"] is None
    assert verify_comparison(root) == result
