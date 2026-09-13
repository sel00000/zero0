"""Provenance-preserving human-video/robot-episode pair manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from .dataset import DatasetError, load_episode
from .training_data import EpisodeRecord


PAIRED_DATA_SCHEMA = "so101_wam.paired_data.v1"
PAIRED_DATA_KIND = "compatible_human_robot_pairs_not_official_humangen"
LEROBOT_V3_COMPATIBLE = "lerobot_dataset_v3_compatible_manifest"
HUMAN_TASK_SPEC_FORMAT = "so101_wam.human_video_prompt_npz.v1"
SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_PROVENANCE_FIELDS = (
    "source_dataset",
    "source_layout",
    "license",
    "transformation",
)


class PairedDataError(ValueError):
    """Raised when a paired-data manifest loses task or provenance evidence."""


class SemanticMatchStatus(StrEnum):
    """Record whether a human independently reviewed the declared task match."""

    UNVERIFIED = "unverified"
    HUMAN_REVIEWED = "human_reviewed"


@dataclass(frozen=True, slots=True)
class HumanTaskSpecArtifact:
    """Checksum-bound RGB/timestamp extraction from the source human video."""

    path: Path
    sha256: str
    format: str = HUMAN_TASK_SPEC_FORMAT


@dataclass(frozen=True, slots=True)
class HumanRobotPairRecord:
    """One human-video task prompt paired with one executable robot episode."""

    pair_id: str
    task: str
    task_index: int
    semantic_match: SemanticMatchStatus
    source_format: str
    source_url: str
    human_video_path: Path
    human_video_sha256: str
    human_view: str
    human_task_spec: HumanTaskSpecArtifact | None
    robot_episode: EpisodeRecord
    robot_episode_sha256: str
    provenance: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        payload = {
            "human_video_sha256": self.human_video_sha256,
            "human_view": self.human_view,
            "human_task_spec": (
                None
                if self.human_task_spec is None
                else {
                    "format": self.human_task_spec.format,
                    "sha256": self.human_task_spec.sha256,
                }
            ),
            "pair_id": self.pair_id,
            "provenance": dict(self.provenance),
            "robot_episode": self.robot_episode.fingerprint,
            "robot_episode_sha256": self.robot_episode_sha256,
            "semantic_match": self.semantic_match.value,
            "source_format": self.source_format,
            "source_url": self.source_url,
            "task": self.task,
            "task_index": self.task_index,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HumanRobotPairSplit:
    """Task-disjoint paired records with stable split digests."""

    train: tuple[HumanRobotPairRecord, ...]
    validation: tuple[HumanRobotPairRecord, ...]
    train_digest: str
    validation_digest: str


def load_human_robot_pairs(path: str | Path) -> tuple[HumanRobotPairRecord, ...]:
    """Load a local manifest that preserves official pair semantics.

    The reader accepts already-exported local NPZ robot episodes. It does not
    decode LeRobot v3 Parquet or MP4 shards directly.
    """

    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    payload = _read_json(manifest_path)
    if payload.get("schema_version") != PAIRED_DATA_SCHEMA:
        raise PairedDataError("unsupported paired-data schema version")
    if payload.get("artifact_kind") != PAIRED_DATA_KIND:
        raise PairedDataError("paired-data artifact kind is not a local compatible pair")
    source_format = _string(payload.get("source_format"), name="source_format")
    if source_format != LEROBOT_V3_COMPATIBLE:
        raise PairedDataError(
            f"source_format must be {LEROBOT_V3_COMPATIBLE!r}"
        )
    source_url = _string(payload.get("source_url"), name="source_url")
    pairs = payload.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise PairedDataError("pairs must be a non-empty array")
    if not pairs:
        raise PairedDataError("pairs must be a non-empty array")

    records: list[HumanRobotPairRecord] = []
    seen_ids: set[str] = set()
    for item in pairs:
        if not isinstance(item, Mapping):
            raise PairedDataError("each pair must be a JSON object")
        record = _load_pair(
            item,
            root=root,
            source_format=source_format,
            source_url=source_url,
        )
        if record.pair_id in seen_ids:
            raise PairedDataError(f"duplicate pair_id: {record.pair_id}")
        seen_ids.add(record.pair_id)
        records.append(record)
    return tuple(records)


def paired_episode_records(
    pairs: Sequence[HumanRobotPairRecord],
) -> tuple[EpisodeRecord, ...]:
    """Extract robot supervision records after pair validation."""

    records = tuple(pair.robot_episode for pair in pairs)
    if len({record.fingerprint for record in records}) != len(records):
        raise PairedDataError("duplicate robot episode in paired records")
    return records


def paired_data_audit(
    pairs: Sequence[HumanRobotPairRecord],
) -> Mapping[str, Any]:
    """Summarize integrity and declared semantics without success claims."""

    records = _validate_pair_group(tuple(pairs), split_name="audit")
    semantic_counts: dict[str, int] = {}
    for pair in records:
        key = pair.semantic_match.value
        semantic_counts[key] = semantic_counts.get(key, 0) + 1
    payload = {
        "schema_version": "so101_wam.paired_data_audit.v1",
        "result": "pass",
        "evidence_level": "offline",
        "artifact_kind": PAIRED_DATA_KIND,
        "pair_count": len(records),
        "task_count": len({(pair.task_index, pair.task) for pair in records}),
        "semantic_match_counts": dict(sorted(semantic_counts.items())),
        "pair_digest": _pair_digest(records),
        "source_formats": sorted({pair.source_format for pair in records}),
        "source_urls": sorted({pair.source_url for pair in records}),
        "human_task_spec_declared_count": sum(
            pair.human_task_spec is not None for pair in records
        ),
        "official_humangen": False,
        "full_lerobot_v3_reader": False,
        "human_video_task_success_evaluated": False,
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "pair_fingerprint": pair.fingerprint,
                "task": pair.task,
                "task_index": pair.task_index,
                "semantic_match": pair.semantic_match.value,
                "human_video_sha256": pair.human_video_sha256,
                "human_task_spec_sha256": (
                    None
                    if pair.human_task_spec is None
                    else pair.human_task_spec.sha256
                ),
                "robot_episode_sha256": pair.robot_episode_sha256,
                "robot_episode_fingerprint": pair.robot_episode.fingerprint,
            }
            for pair in sorted(records, key=lambda item: item.pair_id)
        ],
    }
    return MappingProxyType(payload)


def paired_manifest_as_json(
    pairs: Sequence[HumanRobotPairRecord],
    *,
    root: str | Path,
) -> Mapping[str, Any]:
    """Serialize records as one canonical bundle-relative manifest."""

    records = _validate_pair_group(tuple(pairs), split_name="manifest")
    source_formats = {pair.source_format for pair in records}
    source_urls = {pair.source_url for pair in records}
    if len(source_formats) != 1 or len(source_urls) != 1:
        raise PairedDataError("one manifest requires one source format and URL")

    bundle_root = Path(root).resolve()
    payload = {
        "schema_version": PAIRED_DATA_SCHEMA,
        "artifact_kind": PAIRED_DATA_KIND,
        "source_format": records[0].source_format,
        "source_url": records[0].source_url,
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "task": pair.task,
                "task_index": pair.task_index,
                "semantic_match": pair.semantic_match.value,
                "robot_episode": _relative_path(
                    pair.robot_episode.path,
                    root=bundle_root,
                    name="robot_episode",
                ),
                "robot_episode_sha256": pair.robot_episode_sha256,
                "human_video": _human_video_as_json(pair, root=bundle_root),
                "provenance": dict(pair.provenance),
            }
            for pair in records
        ],
    }
    return payload


def validate_pair_disjoint_split(
    train: Sequence[HumanRobotPairRecord],
    validation: Sequence[HumanRobotPairRecord],
) -> HumanRobotPairSplit:
    """Reject pair, source artifact, and task leakage across a split."""

    train_records = _validate_pair_group(tuple(train), split_name="train")
    validation_records = _validate_pair_group(
        tuple(validation),
        split_name="validation",
    )
    _reject_overlap(
        {pair.fingerprint for pair in train_records},
        {pair.fingerprint for pair in validation_records},
        name="pair fingerprint",
    )
    _reject_overlap(
        {pair.pair_id for pair in train_records},
        {pair.pair_id for pair in validation_records},
        name="pair_id",
    )
    _reject_overlap(
        {pair.human_video_sha256 for pair in train_records},
        {pair.human_video_sha256 for pair in validation_records},
        name="human video checksum",
    )
    _reject_overlap(
        {
            pair.human_task_spec.sha256
            for pair in train_records
            if pair.human_task_spec is not None
        },
        {
            pair.human_task_spec.sha256
            for pair in validation_records
            if pair.human_task_spec is not None
        },
        name="human task-spec checksum",
    )
    _reject_overlap(
        {pair.robot_episode_sha256 for pair in train_records},
        {pair.robot_episode_sha256 for pair in validation_records},
        name="robot episode checksum",
    )
    _reject_overlap(
        {pair.robot_episode.fingerprint for pair in train_records},
        {pair.robot_episode.fingerprint for pair in validation_records},
        name="robot episode fingerprint",
    )

    train_task_ids = {pair.task_index for pair in train_records}
    validation_task_ids = {pair.task_index for pair in validation_records}
    if train_task_ids & validation_task_ids:
        raise PairedDataError("train/validation task_index sets must be disjoint")
    train_tasks = {pair.task for pair in train_records}
    validation_tasks = {pair.task for pair in validation_records}
    if train_tasks & validation_tasks:
        raise PairedDataError("train/validation task labels must be disjoint")

    return HumanRobotPairSplit(
        train=train_records,
        validation=validation_records,
        train_digest=_pair_digest(train_records),
        validation_digest=_pair_digest(validation_records),
    )


def _load_pair(
    item: Mapping[str, Any],
    *,
    root: Path,
    source_format: str,
    source_url: str,
) -> HumanRobotPairRecord:
    pair_id = _string(item.get("pair_id"), name="pair_id")
    task = _string(item.get("task"), name="task")
    task_index = _non_negative_int(item.get("task_index"), name="task_index")
    semantic_match_text = _string(
        item.get("semantic_match"),
        name="semantic_match",
    )
    try:
        semantic_match = SemanticMatchStatus(semantic_match_text)
    except ValueError as error:
        raise PairedDataError(
            "semantic_match must be unverified or human_reviewed"
        ) from error
    robot_path = _bundle_path(item.get("robot_episode"), root=root, name="robot_episode")
    robot_episode_sha256 = _sha256_hex(
        item.get("robot_episode_sha256"),
        name="robot_episode_sha256",
    )
    if _file_sha256(robot_path) != robot_episode_sha256:
        raise PairedDataError("robot episode checksum mismatch")
    try:
        robot = EpisodeRecord(path=robot_path, data=load_episode(robot_path))
    except DatasetError as error:
        raise PairedDataError(f"robot episode is invalid: {error}") from error
    if robot.data.task != task or robot.data.task_index != task_index:
        raise PairedDataError("pair task identity does not match robot episode")

    human_video = item.get("human_video")
    if not isinstance(human_video, Mapping):
        raise PairedDataError("human_video must be a JSON object")
    human_path = _bundle_path(
        human_video.get("path"),
        root=root,
        name="human_video.path",
    )
    expected_sha256 = _sha256_hex(
        human_video.get("sha256"),
        name="human_video.sha256",
    )
    if _file_sha256(human_path) != expected_sha256:
        raise PairedDataError("human video checksum mismatch")
    human_view = _string(human_video.get("view"), name="human_video.view")
    human_task_spec = _load_human_task_spec(
        human_video.get("task_spec"),
        root=root,
    )
    provenance = item.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PairedDataError("provenance must be a JSON object")
    safe_provenance = _json_object(provenance, name="provenance")
    for field in REQUIRED_PROVENANCE_FIELDS:
        safe_provenance[field] = _string(
            safe_provenance.get(field),
            name=f"provenance.{field}",
        )

    return HumanRobotPairRecord(
        pair_id=pair_id,
        task=task,
        task_index=task_index,
        semantic_match=semantic_match,
        source_format=source_format,
        source_url=source_url,
        human_video_path=human_path,
        human_video_sha256=expected_sha256,
        human_view=human_view,
        human_task_spec=human_task_spec,
        robot_episode=robot,
        robot_episode_sha256=robot_episode_sha256,
        provenance=MappingProxyType(safe_provenance),
    )


def _load_human_task_spec(
    value: object,
    *,
    root: Path,
) -> HumanTaskSpecArtifact | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PairedDataError("human_video.task_spec must be a JSON object")
    task_spec_format = _string(
        value.get("format"),
        name="human_video.task_spec.format",
    )
    if task_spec_format != HUMAN_TASK_SPEC_FORMAT:
        raise PairedDataError(
            f"human_video.task_spec.format must be {HUMAN_TASK_SPEC_FORMAT!r}"
        )
    path = _bundle_path(
        value.get("path"),
        root=root,
        name="human_video.task_spec.path",
    )
    expected_sha256 = _sha256_hex(
        value.get("sha256"),
        name="human_video.task_spec.sha256",
    )
    if _file_sha256(path) != expected_sha256:
        raise PairedDataError("human task-spec checksum mismatch")
    return HumanTaskSpecArtifact(path=path, sha256=expected_sha256)


def _human_video_as_json(
    pair: HumanRobotPairRecord,
    *,
    root: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": _relative_path(
            pair.human_video_path,
            root=root,
            name="human_video.path",
        ),
        "sha256": pair.human_video_sha256,
        "view": pair.human_view,
    }
    if pair.human_task_spec is not None:
        payload["task_spec"] = {
            "format": pair.human_task_spec.format,
            "path": _relative_path(
                pair.human_task_spec.path,
                root=root,
                name="human_video.task_spec.path",
            ),
            "sha256": pair.human_task_spec.sha256,
        }
    return payload


def _validate_pair_group(
    pairs: tuple[HumanRobotPairRecord, ...],
    *,
    split_name: str,
) -> tuple[HumanRobotPairRecord, ...]:
    if not pairs:
        raise PairedDataError(f"{split_name} pairs must be non-empty")
    pair_ids = [pair.pair_id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise PairedDataError(f"{split_name} contains duplicate pair_id")
    fingerprints = [pair.fingerprint for pair in pairs]
    if len(set(fingerprints)) != len(fingerprints):
        raise PairedDataError(f"{split_name} contains duplicate pair fingerprint")

    index_to_task: dict[int, str] = {}
    task_to_index: dict[str, int] = {}
    for pair in pairs:
        previous_task = index_to_task.setdefault(pair.task_index, pair.task)
        previous_index = task_to_index.setdefault(pair.task, pair.task_index)
        if previous_task != pair.task or previous_index != pair.task_index:
            raise PairedDataError(
                f"{split_name} task_index and task labels must map one-to-one"
            )
    return pairs


def _reject_overlap(
    train: set[Any],
    validation: set[Any],
    *,
    name: str,
) -> None:
    overlap = train & validation
    if overlap:
        raise PairedDataError(f"train/validation {name} leakage: {sorted(overlap)!r}")


def _pair_digest(pairs: Sequence[HumanRobotPairRecord]) -> str:
    encoded = json.dumps(
        sorted(pair.fingerprint for pair in pairs),
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _relative_path(path: Path, *, root: Path, name: str) -> str:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as error:
        raise PairedDataError(f"{name} must stay inside the manifest bundle") from error
    return relative.as_posix()


def _bundle_path(value: object, *, root: Path, name: str) -> Path:
    raw = _string(value, name=name)
    path = Path(raw)
    if path.is_absolute() or path.as_posix() != raw or ".." in path.parts:
        raise PairedDataError(f"{name} must be a normalized bundle-relative path")
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise PairedDataError(f"{name} must stay inside the manifest bundle") from error
    if not target.is_file():
        raise PairedDataError(f"{name} is missing: {raw}")
    return target


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairedDataError(f"failed to read paired-data manifest: {error}") from error
    if not isinstance(value, dict):
        raise PairedDataError("paired-data manifest must be a JSON object")
    return value


def _json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            json.dumps(dict(value), allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise PairedDataError(f"{name} must be JSON serializable") from error
    if not isinstance(decoded, dict):
        raise PairedDataError(f"{name} must be a JSON object")
    return decoded


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PairedDataError(f"{name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PairedDataError(f"{name} must be a non-negative integer")
    return value


def _sha256_hex(value: object, *, name: str) -> str:
    text = _string(value, name=name)
    if SHA256_HEX_RE.fullmatch(text) is None:
        raise PairedDataError(f"{name} must be a SHA-256 hex digest")
    return text


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "HUMAN_TASK_SPEC_FORMAT",
    "HumanRobotPairRecord",
    "HumanRobotPairSplit",
    "HumanTaskSpecArtifact",
    "LEROBOT_V3_COMPATIBLE",
    "PAIRED_DATA_KIND",
    "PAIRED_DATA_SCHEMA",
    "PairedDataError",
    "SemanticMatchStatus",
    "load_human_robot_pairs",
    "paired_data_audit",
    "paired_manifest_as_json",
    "paired_episode_records",
    "validate_pair_disjoint_split",
]
