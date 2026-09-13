"""Immutable evidence for the ordered decoder ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import statistics
import tempfile
from typing import Any, cast

import numpy as np
import torch

from .checkpoint import load_compact_wam_bytes
from .decoder_evaluation import (
    DECODER_AUDIT_SCHEMA,
    DIAGNOSTIC_SEED,
    MEAN_ATOL,
    MEAN_RTOL,
    FutureSource,
    named_tensor_hashes,
    order_schedule,
    reduce_decoder_rows,
)
from .deployment import canonical_json_sha256
from .model import ActionDecoder, CompactWAM
from .training import (
    PAPER_IFP_LOSS_WEIGHTS,
    PAPER_IFP_STEPS,
    TRAINING_REPORT_SCHEMA,
    CompactWAMTrainingConfig,
    IFPArchitecture,
    SamplingStrategy,
)
from .training_data import (
    EpisodeRecord,
    action_source_summary,
    build_training_windows,
    discover_episode_paths,
    load_episode_records,
    training_axis_statistics,
    validate_task_disjoint_split,
)
from .vision import COMPACT_IMAGE_MAX_SIDE


STUDY_SCHEMA = "so101_wam.decoder_ablation.v1"
STUDY_SEEDS = (3, 7, 11)
STUDY_MODES = (
    ActionDecoder.LEGACY_MEAN,
    ActionDecoder.MEAN_REPEAT_CONTROL,
    ActionDecoder.ORDERED_CONCAT,
)

_HEX64 = frozenset("0123456789abcdef")
_STATUS_KEYS = {
    "id",
    "seed",
    "mode",
    "status",
    "reason",
    "checkpoint",
    "training_report",
}
_LIMITATIONS = (
    "offline teacher-forced decoder diagnostic, not zero-shot/task success",
    "three seeds are not significance and overlapping windows are not replicas",
    "parameter matching != information/capacity/optimization geometry",
    "predictor fixed time offsets may cause order sensitivity without understanding",
    "synthetic task IDs are not independent semantics and native axes retain units",
)


def freeze_protocol(
    root: str | Path,
    train_inputs: Sequence[str | Path],
    validation_inputs: Sequence[str | Path],
    config: CompactWAMTrainingConfig,
) -> dict[str, Any]:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=False)
    cfg = _valid_config(asdict(config))
    _require_study_cfg(cfg)

    train_refs = _copy_inputs(target, "train", train_inputs)
    val_refs = _copy_inputs(target, "validation", validation_inputs)
    sources = _copy_sources(target)
    inputs = [*train_refs, *val_refs]
    train_records, val_records = study_records(target, inputs)
    data = _data_contract(train_records, val_records, cfg)
    protocol = {
        "schema_version": STUDY_SCHEMA,
        "config": asdict(cfg),
        "seeds": list(STUDY_SEEDS),
        "modes": [mode.value for mode in STUDY_MODES],
        "slots": _slots(),
        "runtime": _runtime(),
        "source_revision": _git_head(),
        "source_status": _git_status(),
        "sources": sources,
        "inputs": inputs,
        "data": data,
        "preprocessing": (
            f"aspect_preserving_nearest_max_side_{COMPACT_IMAGE_MAX_SIDE}"
        ),
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "mean_rtol": MEAN_RTOL,
        "mean_atol": MEAN_ATOL,
        "real_output_authorized": False,
    }
    envelope = {
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
    }
    publish_json(target / "protocol.json", envelope)
    return study_protocol(target)


def publish_json(target: str | Path, document: Mapping[str, Any]) -> None:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)

    encoded = _json_bytes(document)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp_path, path)
        os.chmod(path, 0o444)
    except BaseException:
        try:
            if path.exists() and path.stat().st_ino == tmp_path.stat().st_ino:
                path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    document, _ = _read_json_bytes(Path(path))
    return document


def resolve_artifact(root: str | Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("artifact path must be a string")
    if "\\" in name or name == "":
        raise ValueError("artifact path must be bundle-relative POSIX")

    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact path must not contain empty/dot segments")

    pure = PurePosixPath(name)
    if pure.is_absolute():
        raise ValueError("artifact path must be relative")

    raw_root = Path(root)
    if raw_root.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    base = raw_root.resolve(strict=True)
    current = base
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact path must not contain symlinks")

    absolute = base.joinpath(*pure.parts)
    if not absolute.is_relative_to(base):
        raise ValueError("artifact path escapes the bundle")
    return absolute


def artifact_ref(root: str | Path, path: str | Path) -> dict[str, str]:
    raw_root = Path(root)
    if raw_root.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    base = raw_root.resolve(strict=True)
    raw_target = Path(path)
    target = raw_target if raw_target.is_absolute() else Path.cwd() / raw_target
    target = target.absolute()
    if not target.is_relative_to(base):
        raise ValueError("artifact is outside bundle")
    name = target.relative_to(base).as_posix()
    resolved = resolve_artifact(base, name)
    return {"path": name, "sha256": _sha_bytes(resolved.read_bytes())}


def study_protocol(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    envelope, raw = _read_json_bytes(resolve_artifact(base, "protocol.json"))
    if set(envelope) != {"protocol", "protocol_sha256"}:
        raise ValueError("protocol envelope keys do not match schema")
    protocol = _mapping(envelope["protocol"], "protocol")
    if envelope["protocol_sha256"] != canonical_json_sha256(protocol):
        raise ValueError("protocol_sha256 mismatch")

    _check_protocol(protocol)
    _check_inventory(base, protocol)
    train, validation = study_records(
        base,
        cast(Sequence[Mapping[str, Any]], protocol["inputs"]),
    )
    expected_data = _data_contract(train, validation, _config_from_doc(protocol))
    if canonical_json_sha256(expected_data) != canonical_json_sha256(protocol["data"]):
        raise ValueError("protocol data contract mismatch")

    return {
        **envelope,
        "protocol_artifact": {
            "path": "protocol.json",
            "sha256": _sha_bytes(raw),
        },
    }


def study_records(
    root: str | Path,
    references: Sequence[Mapping[str, Any]],
) -> tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]]:
    inputs = _input_pairs(root, references)
    records: list[tuple[EpisodeRecord, ...]] = []
    for split in ("train", "validation"):
        pairs = []
        for index, item in enumerate(inputs[split]):
            npz_ref = item["npz"]
            json_ref = item["json"]
            npz_path = resolve_artifact(root, str(npz_ref["path"]))
            npz = _ref_bytes(root, npz_ref)
            sidecar = _ref_bytes(root, json_ref)
            pairs.append((index, npz_path, npz, sidecar))
        records.append(_load_verified(pairs))
    return cast(tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]], tuple(records))


def verify_runtime(protocol: Mapping[str, Any]) -> None:
    runtime = _mapping(protocol["runtime"], "runtime")
    if runtime["device"] != "cpu":
        raise ValueError("runtime device mismatch")
    if torch.get_num_threads() != runtime["threads"]:
        raise ValueError("torch thread count mismatch")
    if torch.are_deterministic_algorithms_enabled() is not runtime["deterministic"]:
        raise ValueError("torch deterministic mode mismatch")
    if (
        torch.is_deterministic_algorithms_warn_only_enabled()
        is not runtime["deterministic_warn_only"]
    ):
        raise ValueError("torch deterministic warn mode mismatch")
    if _versions() != runtime["versions"]:
        raise ValueError("runtime version mismatch")

    source_root = Path(__file__).resolve().parents[2]
    refs = _source_refs(protocol)
    live = {path: digest for path, digest in _source_inventory(source_root)}
    if live != refs:
        raise ValueError("live source inventory mismatch")


def build_comparison(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    envelope = study_protocol(base)
    protocol = _mapping(envelope["protocol"], "protocol")
    runs = []
    complete = []
    for slot in _sequence(protocol["slots"], "slots"):
        item = _slot_run(base, _mapping(slot, "slot"), envelope)
        runs.append(item)
        if item["status"] == "complete":
            complete.append(item)

    _check_comparable(complete)
    public_runs = [
        {key: value for key, value in run.items() if key != "_report"}
        for run in runs
    ]
    summary = _summary(complete) if len(complete) == len(STUDY_SEEDS) * len(STUDY_MODES) else None
    return {
        "schema_version": STUDY_SCHEMA,
        "protocol": protocol,
        "protocol_sha256": envelope["protocol_sha256"],
        "protocol_artifact": envelope["protocol_artifact"],
        "status": "complete" if summary is not None else "incomplete",
        "runs": public_runs,
        "summary": summary,
        "limitations": list(_LIMITATIONS),
    }


def verify_comparison(root: str | Path) -> dict[str, Any]:
    stored = read_json(resolve_artifact(root, "comparison.json"))
    rebuilt = build_comparison(root)
    if canonical_json_sha256(stored) != canonical_json_sha256(rebuilt):
        raise ValueError(
            "comparison.json is malformed, stale, or falsely complete"
        )
    return rebuilt


def complete_run(
    root: str | Path,
    slot: Mapping[str, Any],
    document: Mapping[str, Any],
) -> None:
    base = Path(root).absolute()
    run_root = resolve_artifact(base, f"runs/{slot['id']}")
    status = {
        "id": slot["id"],
        "seed": slot["seed"],
        "mode": slot["mode"],
        "status": "complete",
        "reason": None,
        "checkpoint": artifact_ref(base, run_root / "candidate.pt"),
        "training_report": artifact_ref(base, run_root / "training.json"),
    }
    _run_metrics(base, run_root, slot, status, document)
    os.chmod(run_root / "candidate.pt", 0o444)
    os.chmod(run_root / "training.json", 0o444)
    publish_json(run_root / "status.json", status)


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return _json_from_bytes(raw), raw


def _json_from_bytes(raw: bytes) -> dict[str, Any]:

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
        document = json.loads(
            raw,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_const,
        )
    except ValueError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("JSON root must be an object")
    _reject_nonfinite(document)
    return document


def _reject_nonfinite(value: object) -> None:
    if type(value) is float and not np.isfinite(value):
        raise ValueError("JSON contains non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def _sha_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _valid_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX64:
        raise ValueError(f"{name} must be 64 lowercase hex")
    return value


def _same_json(left: object, right: object) -> bool:
    return canonical_json_sha256(left) == canonical_json_sha256(right)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _copy_inputs(
    root: Path,
    split: str,
    inputs: Sequence[str | Path],
) -> list[dict[str, str]]:
    paths = discover_episode_paths(inputs)
    _check_aliases(paths)
    refs: list[dict[str, str]] = []
    out_dir = root / "inputs" / split
    out_dir.mkdir(parents=True)
    for index, source in enumerate(paths):
        sidecar = source.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"missing episode sidecar: {sidecar}")
        stem = f"episode_{index:06d}"
        npz_target = out_dir / f"{stem}.npz"
        json_target = out_dir / f"{stem}.json"
        _copy_unchanged(source, npz_target)
        _copy_unchanged(sidecar, json_target)
        refs.append(artifact_ref(root, npz_target))
        refs.append(artifact_ref(root, json_target))
    return refs


def _copy_unchanged(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    data = source.read_bytes()
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp_path, target)
        os.chmod(target, 0o444)
    finally:
        tmp_path.unlink(missing_ok=True)


def _input_pairs(
    root: str | Path,
    references: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {
        "train": {},
        "validation": {},
    }
    for raw in references:
        ref = _valid_ref(raw)
        path = str(ref["path"])
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "inputs":
            raise ValueError("input reference path mismatch")
        split = parts[1]
        if split not in grouped:
            raise ValueError("input reference split mismatch")
        suffix = Path(parts[2]).suffix
        if suffix not in {".npz", ".json"}:
            raise ValueError("input reference suffix mismatch")
        stem = Path(parts[2]).stem
        grouped[split].setdefault(stem, {})[suffix[1:]] = ref
        _ref_bytes(root, ref)

    result: dict[str, list[dict[str, Mapping[str, Any]]]] = {
        "train": [],
        "validation": [],
    }
    for split, stems in grouped.items():
        for stem in sorted(stems):
            pair = stems[stem]
            if set(pair) != {"npz", "json"}:
                raise ValueError("input reference pair mismatch")
            result[split].append({"npz": pair["npz"], "json": pair["json"]})
    return result


def _copy_sources(root: Path) -> list[dict[str, str]]:
    source_root = Path(__file__).resolve().parents[2]
    refs: list[dict[str, str]] = []
    for rel, _ in _source_inventory(source_root):
        source = source_root / rel
        target = root / "source" / rel
        _copy_unchanged(source, target)
        refs.append(artifact_ref(root, target))
    return refs


def _source_inventory(root: Path) -> list[tuple[str, str]]:
    paths = [root / "pyproject.toml"]
    paths.extend(sorted((root / "src").rglob("*.py")))
    result = []
    for path in paths:
        if path.is_file():
            data = path.read_bytes()
            result.append((path.relative_to(root).as_posix(), _sha_bytes(data)))
    return result


def _git_head() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) not in {40, 64} or set(value) - _HEX64:
        raise ValueError("source revision is not hex")
    return value


def _git_status() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--",
            "src",
            "pyproject.toml",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _runtime() -> dict[str, Any]:
    return {
        "device": "cpu",
        "threads": 1,
        "deterministic": True,
        "deterministic_warn_only": False,
        "versions": _versions(),
    }


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


def _slots() -> list[dict[str, str | int]]:
    slots: list[dict[str, str | int]] = []
    for seed in STUDY_SEEDS:
        for mode in STUDY_MODES:
            slots.append(
                {
                    "id": f"seed-{seed}-{mode.value}",
                    "seed": seed,
                    "mode": mode.value,
                }
            )
    return slots


def _valid_config(value: Mapping[str, Any]) -> CompactWAMTrainingConfig:
    expected = {field.name for field in fields(CompactWAMTrainingConfig)}
    if set(value) != expected:
        raise ValueError("training config keys do not match schema")
    data = dict(value)
    data["action_decoder"] = ActionDecoder(data["action_decoder"])
    data["ifp_architecture"] = IFPArchitecture(data["ifp_architecture"])
    data["sampling_strategy"] = SamplingStrategy(data["sampling_strategy"])
    return CompactWAMTrainingConfig(**data)


def _config_from_doc(protocol: Mapping[str, Any]) -> CompactWAMTrainingConfig:
    return _valid_config(_mapping(protocol["config"], "config"))


def _require_study_cfg(config: CompactWAMTrainingConfig) -> None:
    if config.future_steps < 2:
        raise ValueError("future_steps must be at least 2")
    if config.stage1_steps < 1 or config.stage2_steps < 1:
        raise ValueError("both training stages are required")
    if config.action_decoder is not ActionDecoder.LEGACY_MEAN:
        raise ValueError("base action_decoder must be legacy_mean")
    if config.sampling_strategy is not SamplingStrategy.TASK_BALANCED:
        raise ValueError("sampling_strategy must be task_balanced")
    if config.ifp_architecture is not IFPArchitecture.COMPACT_LINEAR:
        raise ValueError("ifp_architecture must be compact_linear")
    if config.ifp_steps != 2 or config.ifp_stride != 2:
        raise ValueError("study requires K2 stride2")
    if config.ifp_window_steps is not None:
        raise ValueError("ifp_window_steps must be None")


def _data_contract(
    train: Sequence[EpisodeRecord],
    validation: Sequence[EpisodeRecord],
    config: CompactWAMTrainingConfig,
) -> dict[str, Any]:
    split = validate_task_disjoint_split(train, validation)
    train_windows = _windows(split.train, config)
    val_windows = _windows(split.validation, config)
    axis_mean, axis_scale = training_axis_statistics(split.train)
    val_tasks = [
        f"{spec.pair.target.data.task_index}:{spec.pair.target.data.task}"
        for spec in val_windows
    ]
    return {
        "train_split_sha256": split.train_digest,
        "validation_split_sha256": split.validation_digest,
        "train_episode_count": len(split.train),
        "validation_episode_count": len(split.validation),
        "train_task_count": len({record.data.task_index for record in split.train}),
        "validation_task_count": len(
            {record.data.task_index for record in split.validation}
        ),
        "train_window_count": len(train_windows),
        "validation_window_count": len(val_windows),
        "train_action_source": action_source_summary(split.train),
        "validation_action_source": action_source_summary(split.validation),
        "normalization": {
            "axis_mean": [float(value) for value in axis_mean],
            "axis_scale": [float(value) for value in axis_scale],
        },
        "validation_order": order_schedule(val_windows),
        "validation_tasks": val_tasks,
    }


def _windows(
    records: Sequence[EpisodeRecord],
    config: CompactWAMTrainingConfig,
):
    return build_training_windows(
        records,
        policy_hz=config.policy_hz,
        servo_hz=config.servo_hz,
        action_history_steps=config.action_history_steps,
        future_steps=config.future_steps,
        action_horizon=config.action_horizon,
        ifp_steps=config.effective_ifp_window_steps,
        ifp_stride=config.ifp_stride,
        max_context_steps=config.max_context_steps,
    )


def _check_protocol(protocol: Mapping[str, Any]) -> None:
    keys = {
        "schema_version",
        "config",
        "seeds",
        "modes",
        "slots",
        "runtime",
        "source_revision",
        "source_status",
        "sources",
        "inputs",
        "data",
        "preprocessing",
        "diagnostic_seed",
        "mean_rtol",
        "mean_atol",
        "real_output_authorized",
    }
    if set(protocol) != keys:
        raise ValueError("protocol keys do not match schema")
    if protocol["schema_version"] != STUDY_SCHEMA:
        raise ValueError("protocol schema mismatch")
    _require_study_cfg(_config_from_doc(protocol))
    if protocol["seeds"] != list(STUDY_SEEDS):
        raise ValueError("study seeds mismatch")
    if protocol["modes"] != [mode.value for mode in STUDY_MODES]:
        raise ValueError("study modes mismatch")
    if protocol["slots"] != _slots():
        raise ValueError("study slots mismatch")
    runtime = _mapping(protocol["runtime"], "runtime")
    if set(runtime) != {
        "device",
        "threads",
        "deterministic",
        "deterministic_warn_only",
        "versions",
    }:
        raise ValueError("runtime keys do not match schema")
    if type(runtime["threads"]) is not int:
        raise ValueError("runtime threads must be an integer")
    if type(runtime["deterministic"]) is not bool:
        raise ValueError("runtime deterministic must be a boolean")
    if type(runtime["deterministic_warn_only"]) is not bool:
        raise ValueError("runtime warn flag must be a boolean")
    if runtime != _runtime():
        raise ValueError("runtime values mismatch")
    revision = protocol["source_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) not in {40, 64}
        or set(revision) - _HEX64
    ):
        raise ValueError("source revision must be 40 or 64 hex")
    if not isinstance(protocol["source_status"], str):
        raise ValueError("source_status must be a string")
    if protocol["preprocessing"] != (
        f"aspect_preserving_nearest_max_side_{COMPACT_IMAGE_MAX_SIDE}"
    ):
        raise ValueError("preprocessing mismatch")
    if type(protocol["diagnostic_seed"]) is not int:
        raise ValueError("diagnostic seed must be an integer")
    if protocol["diagnostic_seed"] != DIAGNOSTIC_SEED:
        raise ValueError("diagnostic seed mismatch")
    if protocol["mean_rtol"] != MEAN_RTOL or protocol["mean_atol"] != MEAN_ATOL:
        raise ValueError("mean tolerance mismatch")
    if protocol["real_output_authorized"] is not False:
        raise ValueError("real output must not be authorized")


def _check_inventory(root: Path, protocol: Mapping[str, Any]) -> None:
    refs = [_valid_ref(ref) for ref in _sequence(protocol["inputs"], "inputs")]
    refs.extend(_valid_ref(ref) for ref in _sequence(protocol["sources"], "sources"))
    _check_refs(root, refs)

    input_paths = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "inputs").rglob("*")
        if path.is_file()
    )
    expected_inputs = sorted(str(ref["path"]) for ref in refs if str(ref["path"]).startswith("inputs/"))
    if input_paths != expected_inputs:
        raise ValueError("input inventory mismatch")

    source_paths = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "source").rglob("*")
        if path.is_file()
    )
    source_refs = sorted(str(ref["path"]) for ref in _sequence(protocol["sources"], "sources"))
    if source_paths != source_refs:
        raise ValueError("source inventory mismatch")
    if "source/pyproject.toml" not in source_refs:
        raise ValueError("source inventory missing pyproject.toml")
    if "source/src/so101_wam/model.py" not in source_refs:
        raise ValueError("source inventory missing model.py")


def _source_refs(protocol: Mapping[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for ref in _sequence(protocol["sources"], "sources"):
        item = _valid_ref(ref)
        path = str(item["path"])
        if not path.startswith("source/"):
            raise ValueError("source reference outside source/")
        refs[path.removeprefix("source/")] = _valid_hex(item["sha256"], "sha256")
    return refs


def _check_refs(root: Path, refs: Sequence[Mapping[str, Any]]) -> None:
    paths = []
    inodes = set()
    names = set()
    for ref in refs:
        item = _valid_ref(ref)
        name = item["path"]
        if name in names:
            raise ValueError("duplicate artifact path")
        names.add(name)
        path = resolve_artifact(root, name)
        paths.append(path)
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in inodes:
            raise ValueError("duplicate artifact inode alias")
        inodes.add(inode)
        if _sha_bytes(path.read_bytes()) != item["sha256"]:
            raise ValueError(f"artifact sha256 mismatch: {name}")
    _check_aliases(paths)


def _check_aliases(paths: Sequence[Path]) -> None:
    seen = set()
    for path in paths:
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen:
            raise ValueError("duplicate artifact inode alias")
        seen.add(inode)


def _ref_bytes(root: str | Path, ref: Mapping[str, Any]) -> bytes:
    item = _valid_ref(ref)
    path = resolve_artifact(root, item["path"])
    data = path.read_bytes()
    if _sha_bytes(data) != item["sha256"]:
        raise ValueError(f"artifact sha256 mismatch: {item['path']}")
    return data


def _valid_ref(ref: object) -> Mapping[str, str]:
    item = _mapping(ref, "artifact reference")
    if set(item) != {"path", "sha256"}:
        raise ValueError("artifact reference keys do not match schema")
    if not isinstance(item["path"], str) or not isinstance(item["sha256"], str):
        raise ValueError("artifact reference values must be strings")
    _valid_hex(item["sha256"], "sha256")
    return cast(Mapping[str, str], item)


def _load_verified(
    pairs: Sequence[tuple[int, Path, bytes, bytes]],
) -> tuple[EpisodeRecord, ...]:
    records: list[EpisodeRecord] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for index, original, npz, sidecar in pairs:
            npz_path = tmp_root / f"episode_{index:06d}.npz"
            json_path = tmp_root / f"episode_{index:06d}.json"
            npz_path.write_bytes(npz)
            json_path.write_bytes(sidecar)
            data = load_episode_records((npz_path,))[0].data
            records.append(EpisodeRecord(original, data))
    return tuple(records)


def _slot_run(
    root: Path,
    slot: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = root / "runs" / str(slot["id"])
    status_path = resolve_artifact(root, f"runs/{slot['id']}/status.json")
    if not status_path.exists():
        return {
            "id": slot["id"],
            "seed": slot["seed"],
            "mode": slot["mode"],
            "status": "not_attempted",
            "reason": "no published run status",
            "checkpoint": None,
            "training_report": None,
            "metrics": None,
            "status_artifact": None,
        }

    status, raw = _read_json_bytes(status_path)
    if set(status) != _STATUS_KEYS:
        raise ValueError("run status keys do not match schema")
    _native_int(status["seed"], "run status seed")
    if status["id"] != slot["id"] or status["seed"] != slot["seed"]:
        raise ValueError("run status identity mismatch")
    if status["mode"] != slot["mode"]:
        raise ValueError("run status mode mismatch")
    if status["status"] not in {"complete", "failed", "not_attempted"}:
        raise ValueError("run status value mismatch")
    if status["status"] != "complete":
        if not isinstance(status["reason"], str) or not status["reason"]:
            raise ValueError("incomplete run status requires reason")
        if status["checkpoint"] is not None or status["training_report"] is not None:
            raise ValueError("incomplete run status must not have refs")
        metrics = None
    else:
        if status["reason"] is not None:
            raise ValueError("complete run status must not have reason")
        metrics, report = _run_metrics(root, run_root, slot, status, envelope)
    return {
        **status,
        "metrics": metrics,
        "status_artifact": {
            "path": status_path.relative_to(root.resolve()).as_posix(),
            "sha256": _sha_bytes(raw),
        },
        **({"_report": report} if status["status"] == "complete" else {}),
    }


def _run_metrics(
    root: Path,
    run_root: Path,
    slot: Mapping[str, Any],
    status: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    ck_ref = _mapping(status["checkpoint"], "checkpoint")
    tr_ref = _mapping(status["training_report"], "training_report")
    expected_ck = (run_root / "candidate.pt").relative_to(root).as_posix()
    expected_tr = (run_root / "training.json").relative_to(root).as_posix()
    if ck_ref.get("path") != expected_ck or tr_ref.get("path") != expected_tr:
        raise ValueError("run references must point at fixed artifact names")

    checkpoint_bytes = _ref_bytes(root, ck_ref)
    report_bytes = _ref_bytes(root, tr_ref)
    report = _json_from_bytes(report_bytes)
    bundle = load_compact_wam_bytes(checkpoint_bytes, device="cpu")
    artifacts = _mapping(report["artifacts"], "artifacts")
    if artifacts["checkpoint_sha256"] != ck_ref["sha256"]:
        raise ValueError("report checkpoint sha256 mismatch")
    _check_report(report, bundle.metadata, bundle.architecture, slot, envelope)
    _check_state(bundle.model)
    _check_norm(bundle.model, _mapping(envelope["protocol"], "protocol"))
    _check_audit(report, bundle.model, slot, envelope)
    metrics = {
        "stage1": report["decoder_audit"]["stage1"]["summary"],
        "stage2": report["decoder_audit"]["stage2"]["summary"],
        "seconds": report["decoder_audit"]["seconds"],
        "head_parameters": report["decoder_audit"]["head_parameters"],
        "model_parameters": report["decoder_audit"]["model_parameters"],
    }
    return metrics, report


def _check_report(
    report: Mapping[str, Any],
    metadata_map: Mapping[str, Any],
    architecture: Mapping[str, Any],
    slot: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> None:
    protocol = _mapping(envelope["protocol"], "protocol")
    if (
        report["schema_version"] != TRAINING_REPORT_SCHEMA
        or report["result"] != "pass"
    ):
        raise ValueError("training report schema/result mismatch")
    _check_value(
        report["artifact_kind"],
        "compact_wam_candidate",
        "training report artifact_kind",
    )
    _check_value(
        report["evidence_level"],
        "offline",
        "training report evidence_level",
    )
    if report["checkpoint_id"] != slot["id"]:
        raise ValueError("training report id mismatch")
    if not _same_json(report["model"], dict(architecture)):
        raise ValueError("training report architecture mismatch")
    expected = _expected_arch(protocol, slot)
    if not _same_json(dict(architecture), expected):
        raise ValueError("checkpoint architecture mismatch")
    if metadata_map.get("checkpoint_id") != slot["id"]:
        raise ValueError("checkpoint metadata id mismatch")
    if metadata_map.get("action_decoder") != slot["mode"]:
        raise ValueError("checkpoint metadata mode mismatch")
    if report["model"]["action_decoder"] != slot["mode"]:
        raise ValueError("training report mode mismatch")
    _check_flags(report)
    _check_flags(metadata_map)
    artifacts = _mapping(report["artifacts"], "artifacts")
    if set(artifacts) != {
        "checkpoint_sha256",
        "checkpoint_filename",
        "report_filename",
    }:
        raise ValueError("training report artifact keys mismatch")
    if artifacts["checkpoint_filename"] != "candidate.pt":
        raise ValueError("checkpoint filename mismatch")
    if artifacts["report_filename"] != "training.json":
        raise ValueError("report filename mismatch")

    core = {
        key: value
        for key, value in report.items()
        if key not in {"artifacts", "training_evidence_sha256"}
    }
    digest = canonical_json_sha256(core)
    if report["training_evidence_sha256"] != digest:
        raise ValueError("training evidence digest mismatch")
    if metadata_map.get("training_evidence_sha256") != digest:
        raise ValueError("metadata evidence digest mismatch")
    if metadata_map.get("study_sha256") != envelope["protocol_sha256"]:
        raise ValueError("metadata study digest mismatch")
    _native_int(slot["seed"], "slot seed")
    if _native_int(metadata_map.get("seed"), "metadata seed") != slot["seed"]:
        raise ValueError("metadata seed mismatch")
    config = _config_from_doc(protocol)
    steps = config.stage1_steps + config.stage2_steps
    if _native_int(metadata_map.get("optimizer_steps"), "optimizer_steps") != steps:
        raise ValueError("optimizer step count mismatch")
    if report["optimization"]["optimizer"] != "AdamW":
        raise ValueError("optimizer mismatch")
    if report["optimization"]["device"] != "cpu":
        raise ValueError("optimization device mismatch")
    report_protocol = _mapping(report["protocol"], "training protocol")
    _check_rpt_proto(report_protocol, protocol, config, architecture)
    _check_config(report, config, slot)
    _check_data(report, protocol)
    _check_meta(metadata_map, report, config, slot, envelope, architecture)
    _valid_hex(report["training_schedule_sha256"], "training_schedule_sha256")
    _valid_hex(metadata_map["sampling_schedule_sha256"], "sampling_schedule_sha256")
    if metadata_map["sampling_schedule_sha256"] != report["training_schedule_sha256"]:
        raise ValueError("metadata sampling schedule mismatch")


def _check_rpt_proto(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    config: CompactWAMTrainingConfig,
    architecture: Mapping[str, Any],
) -> None:
    expected = {
        "prompt_pairing": "same_task_different_episode",
        "split": "task_disjoint",
        "sampling_strategy": config.sampling_strategy.value,
        "stage1": "ground_truth_future_latent_inverse_dynamics",
        "stage2": "predicted_future_latent_end_to_end_with_ifp",
        "ifp_architecture": config.ifp_architecture.value,
        "ifp_stride": config.ifp_stride,
        "ifp_loss_weights": list(_ifp_weights(config.ifp_steps)),
        "ifp_module_removed_from_inference": architecture["ifp_steps"] == 0,
        "wrist_rgb_preprocess": (
            f"aspect_preserving_nearest_max_side_{COMPACT_IMAGE_MAX_SIDE}"
        ),
        "action_target_source": _action_src(protocol),
        "action_row_zero_timing": "policy_anchor_immediate",
        "real_output_authorized": False,
    }
    if set(report) != set(expected):
        raise ValueError("training protocol keys mismatch")
    for key, value in expected.items():
        _check_value(report[key], value, f"training protocol {key}")


def _check_meta(
    metadata_map: Mapping[str, Any],
    report: Mapping[str, Any],
    config: CompactWAMTrainingConfig,
    slot: Mapping[str, Any],
    envelope: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> None:
    data = _mapping(report["data"], "data")
    validation = _mapping(report["validation"], "validation")
    expected = {
        "artifact_kind": "compact_wam_candidate",
        "evidence_level": "offline",
        "checkpoint_id": slot["id"],
        "trained": False,
        "offline_trained": True,
        "deployment_ready": False,
        "training_evidence_sha256": report["training_evidence_sha256"],
        "training_objective": _training_obj(config),
        "training_ifp_steps": config.ifp_steps,
        "ifp_architecture": config.ifp_architecture.value,
        "inference_ifp_module_present": architecture["ifp_steps"] > 0,
        "action_target_source": _action_src(envelope["protocol"]),
        "train_task_count": data["train_task_count"],
        "validation_task_count": data["validation_task_count"],
        "optimizer_steps": config.stage1_steps + config.stage2_steps,
        "seed": slot["seed"],
        "sampling_strategy": config.sampling_strategy.value,
        "sampling_schedule_sha256": report["training_schedule_sha256"],
        "validation_action_mse_normalized": validation[
            "action_mse_normalized"
        ],
        "action_decoder": slot["mode"],
        "study_sha256": envelope["protocol_sha256"],
    }
    if set(metadata_map) != set(expected):
        raise ValueError("checkpoint metadata keys mismatch")
    for key, value in expected.items():
        _check_value(metadata_map[key], value, f"checkpoint metadata {key}")


def _check_value(actual: object, expected: object, name: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{name} mismatch")
    if type(expected) is int:
        _native_int(actual, name)
    if not _same_json(actual, expected):
        raise ValueError(f"{name} mismatch")


def _action_src(protocol: Mapping[str, Any]) -> str:
    data = _mapping(protocol["data"], "data")
    return (
        "servo_rate_linear_interpolation_of_episode_action;"
        f"train={data['train_action_source']};"
        f"validation={data['validation_action_source']}"
    )


def _training_obj(config: CompactWAMTrainingConfig) -> str:
    if config.ifp_steps == 0:
        return "compact_future_latent_action_mse"
    if config.ifp_architecture is IFPArchitecture.FUSED_MODULES:
        return "compact_future_latent_action_fused_ifp_mse"
    return "compact_future_latent_action_ifp_mse"


def _ifp_weights(ifp_steps: int) -> tuple[float, ...]:
    if ifp_steps == 0:
        return ()
    if ifp_steps == PAPER_IFP_STEPS:
        return PAPER_IFP_LOSS_WEIGHTS
    weight = 1.0 / ifp_steps
    return (weight,) * ifp_steps


def _check_flags(value: Mapping[str, Any]) -> None:
    if value.get("offline_trained") is not True:
        raise ValueError("offline_trained flag mismatch")
    if value.get("trained") is not False:
        raise ValueError("trained flag mismatch")
    if value.get("deployment_ready") is not False:
        raise ValueError("deployment_ready flag mismatch")


def _expected_arch(
    protocol: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> dict[str, Any]:
    config = _config_from_doc(protocol)
    return {
        "latent_dim": config.latent_dim,
        "transformer_layers": config.transformer_layers,
        "transformer_heads": config.transformer_heads,
        "future_steps": config.future_steps,
        "action_horizon": config.action_horizon,
        "action_history_steps": config.action_history_steps,
        "ifp_steps": config.ifp_steps,
        "max_context_steps": config.max_context_steps,
        "action_decoder": slot["mode"],
    }


def _check_config(
    report: Mapping[str, Any],
    config: CompactWAMTrainingConfig,
    slot: Mapping[str, Any],
) -> None:
    expected = asdict(config)
    expected["seed"] = slot["seed"]
    expected["action_decoder"] = slot["mode"]
    for key, value in expected.items():
        if not _same_json(report["optimization"].get(key), value):
            raise ValueError(f"optimization config mismatch: {key}")


def _check_data(report: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    data = _mapping(report["data"], "data")
    frozen = _mapping(protocol["data"], "data")
    pairs = {
        "train_episode_count": "train_episode_count",
        "validation_episode_count": "validation_episode_count",
        "train_task_count": "train_task_count",
        "validation_task_count": "validation_task_count",
        "train_window_count": "train_window_count",
        "validation_window_count": "validation_window_count",
        "train_split_sha256": "train_split_sha256",
        "validation_split_sha256": "validation_split_sha256",
        "train_action_source": "train_action_source",
        "validation_action_source": "validation_action_source",
    }
    for report_key, frozen_key in pairs.items():
        if report_key.endswith("_count"):
            _native_int(data[report_key], report_key)
        if not _same_json(data[report_key], frozen[frozen_key]):
            raise ValueError(f"data contract mismatch: {report_key}")
    if not _same_json(report["normalization"], frozen["normalization"]):
        raise ValueError("normalization mismatch")
    for key in (
        "train_episode_count",
        "validation_episode_count",
        "train_task_count",
        "validation_task_count",
        "train_window_count",
        "validation_window_count",
    ):
        _native_int(data[key], key)


def _check_state(model: torch.nn.Module) -> None:
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint tensor is non-finite: {name}")


def _check_norm(model: torch.nn.Module, protocol: Mapping[str, Any]) -> None:
    frozen = _mapping(_mapping(protocol["data"], "data")["normalization"], "norm")
    axis_mean = cast(torch.Tensor, getattr(model, "axis_mean")).detach().cpu().tolist()
    axis_scale = cast(torch.Tensor, getattr(model, "axis_scale")).detach().cpu().tolist()
    if not _same_json(axis_mean, frozen["axis_mean"]) or not _same_json(
        axis_scale,
        frozen["axis_scale"],
    ):
        raise ValueError("checkpoint normalization mismatch")


def _check_audit(
    report: Mapping[str, Any],
    model: CompactWAM,
    slot: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> None:
    audit = _mapping(report["decoder_audit"], "decoder_audit")
    if audit["schema_version"] != DECODER_AUDIT_SCHEMA:
        raise ValueError("decoder audit schema mismatch")
    if audit["study_sha256"] != envelope["protocol_sha256"]:
        raise ValueError("decoder audit study digest mismatch")
    expected_keys = set(model.state_dict())
    for name in ("initial_tensors", "stage1_tensors", "final_tensors"):
        values = _mapping(audit[name], name)
        if set(values) != expected_keys:
            raise ValueError(f"{name} keys mismatch")
        for digest in values.values():
            _valid_hex(digest, name)
    if dict(_mapping(audit["final_tensors"], "final_tensors")) != named_tensor_hashes(model):
        raise ValueError("final tensor hashes mismatch")
    for key, digest in _mapping(audit["initial_tensors"], "initial").items():
        if key.startswith("action_head."):
            continue
        if _mapping(audit["stage1_tensors"], "stage1").get(key) != digest:
            raise ValueError("stage1 modified non-head tensor")
    head_count = sum(parameter.numel() for parameter in model.action_head.parameters())
    model_count = sum(parameter.numel() for parameter in model.parameters())
    if audit["head_parameters"] != head_count:
        raise ValueError("head parameter count mismatch")
    if audit["model_parameters"] != model_count:
        raise ValueError("model parameter count mismatch")
    seconds = _mapping(audit["seconds"], "seconds")
    if set(seconds) != {
        "stage1_train",
        "stage1_evaluation",
        "stage2_train",
        "stage2_evaluation",
    }:
        raise ValueError("decoder audit timing keys mismatch")
    for value in seconds.values():
        _finite(value, "seconds")
    schedules = _mapping(audit["stage_schedule_sha256"], "stage_schedule_sha256")
    if set(schedules) != {"stage1", "stage2"}:
        raise ValueError("stage schedule digest keys mismatch")
    _valid_hex(schedules["stage1"], "stage1 schedule")
    _valid_hex(schedules["stage2"], "stage2 schedule")
    _check_eval(audit["stage1"], FutureSource.ORACLE, slot, envelope)
    _check_eval(audit["stage2"], FutureSource.PREDICTED, slot, envelope)


def _check_eval(
    value: object,
    source: FutureSource,
    slot: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> None:
    item = _mapping(value, "decoder evaluation")
    protocol = _mapping(envelope["protocol"], "protocol")
    data = _mapping(protocol["data"], "data")
    if item["source"] != source.value:
        raise ValueError("decoder evaluation source mismatch")
    if type(item["diagnostic_seed"]) is not int:
        raise ValueError("decoder evaluation seed must be an integer")
    if item["diagnostic_seed"] != DIAGNOSTIC_SEED:
        raise ValueError("decoder evaluation seed mismatch")
    if item["permutation_sha256"] != canonical_json_sha256(data["validation_order"]):
        raise ValueError("decoder evaluation schedule mismatch")
    rows = _sequence(item["rows"], "rows")
    keys = [row["key"] for row in data["validation_order"]]
    if [row["key"] for row in rows] != keys:
        raise ValueError("decoder evaluation row keys mismatch")
    tasks = list(data["validation_tasks"])
    if [_mapping(row, "row")["task"] for row in rows] != tasks:
        raise ValueError("decoder evaluation task mismatch")
    expected = reduce_decoder_rows(cast(Sequence[Mapping[str, Any]], rows))
    if canonical_json_sha256(expected) != canonical_json_sha256(item["summary"]):
        raise ValueError("decoder evaluation summary mismatch")


def _native_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float and type(value) is not int:
        raise ValueError(f"{name} must be a native number")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite non-negative")
    return result


def _check_comparable(runs: Sequence[Mapping[str, Any]]) -> None:
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for run in runs:
        by_seed.setdefault(int(run["seed"]), []).append(run)
    for items in by_seed.values():
        if len(items) < 2:
            continue
        base = _metrics_report(items[0])
        for item in items[1:]:
            current = _metrics_report(item)
            if _non_head(base["decoder_audit"]["initial_tensors"]) != _non_head(
                current["decoder_audit"]["initial_tensors"]
            ):
                raise ValueError("non-head initial tensors are incomparable")
            for key in ("normalization", "data", "training_schedule_sha256"):
                if base[key] != current[key]:
                    raise ValueError(f"run comparability mismatch: {key}")
            if (
                base["decoder_audit"]["stage_schedule_sha256"]
                != current["decoder_audit"]["stage_schedule_sha256"]
            ):
                raise ValueError("stage schedule hashes are incomparable")
        wide = [
            _metrics_report(item)
            for item in items
            if item["mode"] in {
                ActionDecoder.MEAN_REPEAT_CONTROL.value,
                ActionDecoder.ORDERED_CONCAT.value,
            }
        ]
        if len(wide) == 2:
            left, right = wide
            if _head(left) != _head(right):
                raise ValueError("wide head initial tensors are incomparable")
            if (
                left["decoder_audit"]["head_parameters"]
                != right["decoder_audit"]["head_parameters"]
            ):
                raise ValueError("wide head parameter counts are incomparable")


def _metrics_report(run: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(run["_report"], "report") if "_report" in run else {}


def _non_head(items: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in items.items() if not key.startswith("action_head.")}


def _head(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report["decoder_audit"]["initial_tensors"].items()
        if key.startswith("action_head.")
    }


def _summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_seed_mode = {(str(run["seed"]), run["mode"]): run for run in runs}
    pairs = {
        "ordered_concat_minus_mean_repeat_control": (
            ActionDecoder.ORDERED_CONCAT.value,
            ActionDecoder.MEAN_REPEAT_CONTROL.value,
        ),
        "ordered_concat_minus_legacy_mean": (
            ActionDecoder.ORDERED_CONCAT.value,
            ActionDecoder.LEGACY_MEAN.value,
        ),
        "mean_repeat_control_minus_legacy_mean": (
            ActionDecoder.MEAN_REPEAT_CONTROL.value,
            ActionDecoder.LEGACY_MEAN.value,
        ),
    }
    result = {}
    for name, (left, right) in pairs.items():
        per_seed = {}
        for seed in map(str, STUDY_SEEDS):
            left_mse = by_seed_mode[(seed, left)]["metrics"]["stage2"]["task_macro_mse"]
            right_mse = by_seed_mode[(seed, right)]["metrics"]["stage2"]["task_macro_mse"]
            per_seed[seed] = left_mse - right_mse
        values = list(per_seed.values())
        result[name] = {
            "per_seed": per_seed,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
    return {
        "primary": "ordered_concat_minus_mean_repeat_control",
        "paired_differences": result,
    }


__all__ = [
    "STUDY_MODES",
    "STUDY_SCHEMA",
    "STUDY_SEEDS",
    "artifact_ref",
    "build_comparison",
    "complete_run",
    "freeze_protocol",
    "publish_json",
    "read_json",
    "resolve_artifact",
    "study_protocol",
    "study_records",
    "verify_comparison",
    "verify_runtime",
]
