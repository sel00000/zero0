"""Read-only numerical replay for completed decoder studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from hashlib import sha256
from importlib import metadata
import json
from math import isclose, isfinite
from pathlib import Path
import platform
import subprocess
from typing import Any

import torch

from .checkpoint import load_compact_wam_bytes
from .decoder_evaluation import (
    FutureSource,
    MEAN_ATOL,
    MEAN_RTOL,
    evaluate_decoder,
    order_schedule,
)
from .decoder_evidence import study_protocol, study_records, verify_comparison
from .decoder_evidence import resolve_artifact
from .model import ActionDecoder
from .training import (
    CompactWAMTrainingConfig,
    IFPArchitecture,
    SamplingStrategy,
)
from .training_data import build_training_windows


REPLAY_SCHEMA = "so101_wam.decoder_replay.v1"


def replay_decoder_study(root: str | Path) -> dict[str, Any]:
    """Replay final stage-2 metrics from stored checkpoints without bundle writes."""

    base = Path(root).absolute()
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_rng = torch.random.get_rng_state()

    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        return _replay(base)
    finally:
        torch.random.set_rng_state(old_rng)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
        torch.set_num_threads(old_threads)


def _replay(root: Path) -> dict[str, Any]:
    source_inventory = _inventory_digest()
    comparison = verify_comparison(root)
    if comparison["status"] != "complete":
        raise ValueError("numerical replay requires a complete comparison")

    envelope = study_protocol(root)
    protocol = _mapping(envelope["protocol"], "protocol")
    train, validation = study_records(root, _sequence(protocol["inputs"], "inputs"))
    del train
    windows = build_training_windows(
        validation,
        policy_hz=float(protocol["config"]["policy_hz"]),
        servo_hz=float(protocol["config"]["servo_hz"]),
        action_history_steps=int(protocol["config"]["action_history_steps"]),
        future_steps=int(protocol["config"]["future_steps"]),
        action_horizon=int(protocol["config"]["action_horizon"]),
        ifp_steps=_ifp_steps(protocol["config"]),
        ifp_stride=int(protocol["config"]["ifp_stride"]),
        max_context_steps=int(protocol["config"]["max_context_steps"]),
    )
    expected_order = _sequence(
        _mapping(protocol["data"], "data")["validation_order"],
        "validation_order",
    )
    if order_schedule(windows) != list(expected_order):
        raise ValueError("numerical replay validation schedule mismatch")

    runs = []
    for run in _sequence(comparison["runs"], "runs"):
        item = _mapping(run, "run")
        if item["status"] != "complete":
            raise ValueError("numerical replay requires complete slots")
        runs.append(_replay_run(root, item, windows))

    if _inventory_digest() != source_inventory:
        raise ValueError("numerical replay source changed during execution")

    return {
        "schema_version": REPLAY_SCHEMA,
        "status": "complete",
        "comparison_status": comparison["status"],
        "protocol_sha256": comparison["protocol_sha256"],
        "stage1_replayed": False,
        "stage2_replayed": True,
        "tolerances": {"rtol": MEAN_RTOL, "atol": MEAN_ATOL},
        "provenance": _provenance(protocol, source_inventory),
        "runs": runs,
        "limitations": [
            "replays final stage-2 checkpoints only; stage-1 checkpoints were not saved",
            "uses current evaluator source, not historical execution attestation",
            "offline numerical replay only; no hardware or zero-shot success claim",
        ],
    }


def _ifp_steps(config: Mapping[str, Any]) -> int:
    cfg = _config(config)
    return cfg.effective_ifp_window_steps


def _config(config: Mapping[str, Any]) -> CompactWAMTrainingConfig:
    values = dict(config)
    values["action_decoder"] = ActionDecoder(values["action_decoder"])
    values["ifp_architecture"] = IFPArchitecture(values["ifp_architecture"])
    values["sampling_strategy"] = SamplingStrategy(values["sampling_strategy"])
    names = {field.name for field in fields(CompactWAMTrainingConfig)}
    return CompactWAMTrainingConfig(**{key: values[key] for key in names})


def _replay_run(
    root: Path,
    run: Mapping[str, Any],
    windows: Sequence[Any],
) -> dict[str, Any]:
    checkpoint_bytes = _artifact_bytes(
        root,
        _mapping(run["checkpoint"], "checkpoint"),
    )
    report_bytes = _artifact_bytes(
        root,
        _mapping(run["training_report"], "training_report"),
    )
    report = _json_from_bytes(report_bytes)
    audit = _mapping(report["decoder_audit"], "decoder_audit")
    stored = _mapping(audit["stage2"], "stage2")
    bundle = load_compact_wam_bytes(checkpoint_bytes, device="cpu")
    replayed = evaluate_decoder(bundle.model, windows, source=FutureSource.PREDICTED)
    _compare_eval(stored, replayed)

    return {
        "id": run["id"],
        "seed": run["seed"],
        "mode": run["mode"],
        "status": "matched",
        "row_count": len(_sequence(replayed["rows"], "rows")),
        "stage2_summary": replayed["summary"],
    }


def _artifact_bytes(root: Path, ref: Mapping[str, Any]) -> bytes:
    path = resolve_artifact(root, str(ref["path"]))
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != ref["sha256"]:
        raise ValueError("numerical replay artifact hash mismatch")
    return data


def _json_from_bytes(data: bytes) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_const(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_const,
        )
    except UnicodeDecodeError as error:
        raise ValueError("invalid UTF-8 JSON in numerical replay") from error
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object in numerical replay")
    return value


def _compare_eval(stored: Mapping[str, Any], replayed: Mapping[str, Any]) -> None:
    _compare(stored, replayed, path="stage2")


def _compare(stored: Any, replayed: Any, *, path: str) -> None:
    if isinstance(stored, Mapping) and isinstance(replayed, Mapping):
        if set(stored) != set(replayed):
            raise ValueError(f"numerical replay key mismatch at {path}")
        for key in sorted(stored):
            _compare(stored[key], replayed[key], path=f"{path}.{key}")
        return

    if isinstance(stored, list) and isinstance(replayed, list):
        if len(stored) != len(replayed):
            raise ValueError(f"numerical replay length mismatch at {path}")
        for index, (left, right) in enumerate(zip(stored, replayed, strict=True)):
            _compare(left, right, path=f"{path}[{index}]")
        return

    if _float_like(stored) and _float_like(replayed):
        left = float(stored)
        right = float(replayed)
        if not isfinite(left) or not isfinite(right):
            raise ValueError(f"numerical replay non-finite number at {path}")
        if not isclose(left, right, rel_tol=MEAN_RTOL, abs_tol=MEAN_ATOL):
            raise ValueError(f"numerical replay mismatch at {path}")
        return

    if type(stored) is not type(replayed):
        raise ValueError(f"numerical replay type mismatch at {path}")
    if stored != replayed:
        raise ValueError(f"numerical replay identity mismatch at {path}")


def _float_like(value: Any) -> bool:
    return type(value) is float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _provenance(
    protocol: Mapping[str, Any],
    source_inventory: str,
) -> dict[str, Any]:
    return {
        "frozen_source_revision": protocol["source_revision"],
        "frozen_source_status": protocol["source_status"],
        "current_source_revision": _git(["rev-parse", "HEAD"]),
        "current_source_status": _git(["status", "--short"]),
        "runtime": {
            "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "versions": _versions(),
        },
        "current_source_inventory_sha256": source_inventory,
        "current_evaluator_sha256": _source_digest("decoder_evaluation.py"),
        "current_replay_sha256": _source_digest("decoder_replay.py"),
    }


def _git(args: Sequence[str]) -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def _source_digest(name: str) -> str:
    path = Path(__file__).resolve().with_name(name)
    return sha256(path.read_bytes()).hexdigest()


def _inventory_digest() -> str:
    source_root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for path in _source_paths(source_root):
        data = path.read_bytes()
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(sha256(data).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _source_paths(source_root: Path) -> list[Path]:
    paths = [source_root / "pyproject.toml"]
    paths.extend(sorted((source_root / "src").rglob("*.py")))
    return [path for path in paths if path.is_file()]


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _pkg_version("torch"),
        "numpy": _pkg_version("numpy"),
        "mujoco": _pkg_version("mujoco"),
        "so101-wam": _pkg_version("so101-wam"),
    }


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


__all__ = ["REPLAY_SCHEMA", "replay_decoder_study"]
