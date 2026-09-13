"""Run the bounded ordered-decoder ablation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import torch

from .decoder_evidence import (
    build_comparison,
    complete_run,
    freeze_protocol,
    publish_json,
    resolve_artifact,
    study_protocol,
    study_records,
    verify_comparison,
    verify_runtime,
)
from .decoder_replay import replay_decoder_study
from .model import ActionDecoder
from .training import CompactWAMTrainingConfig, train_offline_candidate


INTERRUPT_REASON = "no automatic retry or budget extension"


def run_decoder_study(
    root: str | Path,
    *,
    train_inputs: Sequence[str | Path],
    validation_inputs: Sequence[str | Path],
    config: CompactWAMTrainingConfig = CompactWAMTrainingConfig(),
) -> dict[str, Any]:
    base = Path(root).absolute()
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        document = freeze_protocol(base, train_inputs, validation_inputs, config)
        original = dict(document)
        slots = list(document["protocol"]["slots"])

        for index, slot in enumerate(slots):
            try:
                _check_preconditions(base, original)
                train, validation = study_records(base, original["protocol"]["inputs"])
            except KeyboardInterrupt:
                _mark_left(base, slots[index:], INTERRUPT_REASON)
                break
            except Exception as error:
                _mark_left(base, slots[index:], _reason(error))
                break
            run_root = resolve_artifact(base, f"runs/{slot['id']}")
            slot_config = replace(
                config,
                seed=slot["seed"],
                action_decoder=ActionDecoder(slot["mode"]),
            )

            try:
                train_offline_candidate(
                    train,
                    validation,
                    checkpoint_path=run_root / "candidate.pt",
                    report_path=run_root / "training.json",
                    checkpoint_id=slot["id"],
                    config=slot_config,
                    device="cpu",
                    study_sha256=original["protocol_sha256"],
                )
                # Recheck before publishing, including the final study slot.
                try:
                    _check_preconditions(base, original)
                except Exception as error:
                    reason = _reason(error)
                    _mark_failed(base, slot, reason)
                    _mark_left(base, slots[index + 1 :], reason)
                    break
                complete_run(base, slot, original)
            except KeyboardInterrupt:
                _mark_failed(base, slot, INTERRUPT_REASON)
                _mark_left(base, slots[index + 1 :], INTERRUPT_REASON)
                break
            except Exception as error:
                _mark_failed(base, slot, _reason(error))

        comparison = build_comparison(base)
        publish_json(base / "comparison.json", comparison)
        return verify_comparison(base)
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
        torch.set_num_threads(old_threads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="so101-wam-decoder-ablation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--train-episodes", nargs="+", type=Path, required=True)
    run.add_argument("--validation-episodes", nargs="+", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            result = run_decoder_study(
                args.output_dir,
                train_inputs=args.train_episodes,
                validation_inputs=args.validation_episodes,
            )
        elif args.command == "verify":
            result = verify_comparison(args.bundle)
        else:
            result = replay_decoder_study(args.bundle)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(f"decoder study rejected: {error}")

    payload = result if args.command == "replay" else _cli_result(result)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


def _cli_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "protocol_sha256": result["protocol_sha256"],
        "summary": result["summary"],
    }


def _check_preconditions(root: Path, document: dict[str, Any]) -> None:
    if study_protocol(root) != document:
        raise ValueError("protocol precondition changed")
    verify_runtime(document["protocol"])


def _mark_left(
    root: Path,
    slots: Sequence[dict[str, Any]],
    reason: str,
) -> None:
    for slot in slots:
        _mark_unattempted(root, slot, reason)


def _mark_failed(root: Path, slot: dict[str, Any], reason: str) -> None:
    _publish_status(root, slot, "failed", reason)


def _mark_unattempted(root: Path, slot: dict[str, Any], reason: str) -> None:
    _publish_status(root, slot, "not_attempted", reason)


def _publish_status(
    root: Path,
    slot: dict[str, Any],
    status: str,
    reason: str,
) -> None:
    run_root = resolve_artifact(root, f"runs/{slot['id']}")
    run_root.mkdir(parents=True, exist_ok=True)
    publish_json(
        run_root / "status.json",
        {
            "id": slot["id"],
            "seed": slot["seed"],
            "mode": slot["mode"],
            "status": status,
            "reason": reason,
            "checkpoint": None,
            "training_report": None,
        },
    )


def _reason(error: BaseException) -> str:
    text = str(error)
    name = type(error).__name__
    return f"{name}: {text}" if text else name


if __name__ == "__main__":
    raise SystemExit(main())
