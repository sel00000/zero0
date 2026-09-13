from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.paired_data import (
    LEROBOT_V3_COMPATIBLE,
    PAIRED_DATA_KIND,
    PAIRED_DATA_SCHEMA,
    PairedDataError,
    SemanticMatchStatus,
    load_human_robot_pairs,
    paired_manifest_as_json,
    paired_episode_records,
    validate_pair_disjoint_split,
)
from so101_wam.paired_data_cli import main as paired_data_main


def _episode(directory: Path, *, task: str, task_index: int, index: int) -> Path:
    buffer = EpisodeBuffer(
        fps=30.0,
        task=task,
        task_index=task_index,
        episode_index=index,
        metadata={"action_source": "imported_lerobot_v3_action"},
    )
    for frame_index in range(91):
        value = np.full(ACTION_DIM, frame_index + index, dtype=np.float32)
        buffer.append(
            SensorimotorFrame(
                timestamp_s=frame_index / 30.0,
                images={
                    "left_wrist": np.full((8, 8, 3), frame_index, dtype=np.uint8),
                    "right_wrist": np.full((8, 8, 3), frame_index + 1, dtype=np.uint8),
                },
                joint_position=value,
                executed_action=value,
            )
        )
    path, _ = buffer.save(directory, stem=f"episode_{index:06d}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    directory: Path,
    *,
    task: str,
    task_index: int,
    index: int,
    pair_id: str,
    human_bytes: bytes,
) -> Path:
    episode_path = _episode(
        directory / "episodes",
        task=task,
        task_index=task_index,
        index=index,
    )
    human_path = directory / "human" / f"demo_{index}.mp4"
    human_path.parent.mkdir(parents=True)
    human_path.write_bytes(human_bytes)
    manifest_path = directory / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": PAIRED_DATA_SCHEMA,
                "artifact_kind": PAIRED_DATA_KIND,
                "source_format": LEROBOT_V3_COMPATIBLE,
                "source_url": "hf://datasets/example/repo",
                "pairs": [
                    {
                        "pair_id": pair_id,
                        "task": task,
                        "task_index": task_index,
                        "semantic_match": SemanticMatchStatus.UNVERIFIED.value,
                        "robot_episode": str(
                            episode_path.relative_to(directory).as_posix()
                        ),
                        "robot_episode_sha256": _sha256(episode_path),
                        "human_video": {
                            "path": human_path.relative_to(directory).as_posix(),
                            "sha256": _sha256(human_path),
                            "view": "third_person",
                        },
                        "provenance": {
                            "source_dataset": "example/repo",
                            "source_layout": "meta/info.json,data/,videos/",
                            "license": "apache-2.0",
                            "transformation": (
                                "robot episode converted to local NPZ; "
                                "human MP4 copied byte-for-byte"
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_paired_manifest_preserves_provenance_and_robot_episode(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        task="place block",
        task_index=9,
        index=3,
        pair_id="pair-0001",
        human_bytes=b"mp4-placeholder",
    )

    pairs = load_human_robot_pairs(manifest_path)
    records = paired_episode_records(pairs)

    assert pairs[0].pair_id == "pair-0001"
    assert pairs[0].task == "place block"
    assert pairs[0].task_index == 9
    assert pairs[0].source_format == LEROBOT_V3_COMPATIBLE
    assert pairs[0].semantic_match is SemanticMatchStatus.UNVERIFIED
    assert pairs[0].human_video_sha256 == _sha256(pairs[0].human_video_path)
    assert pairs[0].robot_episode_sha256 == _sha256(pairs[0].robot_episode.path)
    assert pairs[0].robot_episode.data.task_index == 9
    assert pairs[0].provenance["source_layout"] == "meta/info.json,data/,videos/"
    assert records == (pairs[0].robot_episode,)

    exported = paired_manifest_as_json(pairs, root=tmp_path)
    exported_path = tmp_path / "exported.json"
    exported_path.write_text(json.dumps(exported), encoding="utf-8")
    reloaded = load_human_robot_pairs(exported_path)
    assert [pair.fingerprint for pair in reloaded] == [pair.fingerprint for pair in pairs]


def test_paired_manifest_rejects_hash_and_task_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        task="pick",
        task_index=1,
        index=1,
        pair_id="pair-0001",
        human_bytes=b"demo",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["pairs"][0]["task"] = "place"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedDataError, match="task identity"):
        load_human_robot_pairs(manifest_path)

    payload["pairs"][0]["task"] = "pick"
    payload["pairs"][0]["human_video"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedDataError, match="human video checksum"):
        load_human_robot_pairs(manifest_path)

    human_path = tmp_path / payload["pairs"][0]["human_video"]["path"]
    payload["pairs"][0]["human_video"]["sha256"] = _sha256(human_path)
    payload["pairs"][0]["robot_episode_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedDataError, match="robot episode checksum"):
        load_human_robot_pairs(manifest_path)


def test_paired_manifest_rejects_bundle_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    manifest_path = _write_manifest(
        bundle,
        task="pick",
        task_index=1,
        index=1,
        pair_id="pair-0001",
        human_bytes=b"demo",
    )
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"demo")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["pairs"][0]["human_video"]["path"] = "../outside.mp4"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedDataError, match="normalized bundle-relative"):
        load_human_robot_pairs(manifest_path)


def test_paired_manifest_requires_review_status_and_provenance(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        task="pick",
        task_index=1,
        index=1,
        pair_id="pair-0001",
        human_bytes=b"demo",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["pairs"][0]["semantic_match"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedDataError, match="semantic_match"):
        load_human_robot_pairs(manifest_path)

    payload["pairs"][0]["semantic_match"] = "unverified"
    del payload["pairs"][0]["provenance"]["license"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedDataError, match="provenance.license"):
        load_human_robot_pairs(manifest_path)


def test_pair_split_rejects_human_and_task_leakage(tmp_path: Path) -> None:
    train = load_human_robot_pairs(
        _write_manifest(
            tmp_path / "train",
            task="pick",
            task_index=1,
            index=1,
            pair_id="train-pair",
            human_bytes=b"shared-human",
        )
    )
    shared_human_validation = load_human_robot_pairs(
        _write_manifest(
            tmp_path / "shared-human-validation",
            task="stack",
            task_index=2,
            index=2,
            pair_id="validation-pair",
            human_bytes=b"shared-human",
        )
    )

    with pytest.raises(PairedDataError, match="human video checksum leakage"):
        validate_pair_disjoint_split(train, shared_human_validation)

    task_validation = load_human_robot_pairs(
        _write_manifest(
            tmp_path / "task-validation",
            task="pick",
            task_index=1,
            index=3,
            pair_id="other-pair",
            human_bytes=b"other-human",
        )
    )
    with pytest.raises(PairedDataError, match="task_index sets must be disjoint"):
        validate_pair_disjoint_split(train, task_validation)


def test_pair_split_rejects_robot_payload_leakage(tmp_path: Path) -> None:
    train_manifest = _write_manifest(
        tmp_path / "train",
        task="pick",
        task_index=1,
        index=1,
        pair_id="train-pair",
        human_bytes=b"train-human",
    )
    validation_manifest = _write_manifest(
        tmp_path / "validation",
        task="stack",
        task_index=2,
        index=1,
        pair_id="validation-pair",
        human_bytes=b"validation-human",
    )
    train_payload = json.loads(train_manifest.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation_manifest.read_text(encoding="utf-8"))
    train_robot = tmp_path / "train" / train_payload["pairs"][0]["robot_episode"]
    validation_robot = (
        tmp_path / "validation" / validation_payload["pairs"][0]["robot_episode"]
    )
    shutil.copyfile(train_robot, validation_robot)
    robot_sha256 = _sha256(train_robot)
    validation_robot_manifest = validation_robot.with_suffix(".json")
    robot_metadata = json.loads(
        validation_robot_manifest.read_text(encoding="utf-8")
    )
    robot_metadata["npz_sha256"] = robot_sha256
    validation_robot_manifest.write_text(
        json.dumps(robot_metadata),
        encoding="utf-8",
    )
    validation_payload["pairs"][0]["robot_episode_sha256"] = robot_sha256
    validation_manifest.write_text(json.dumps(validation_payload), encoding="utf-8")

    train = load_human_robot_pairs(train_manifest)
    validation = load_human_robot_pairs(validation_manifest)
    assert train[0].robot_episode.fingerprint != validation[0].robot_episode.fingerprint

    with pytest.raises(PairedDataError, match="robot episode checksum leakage"):
        validate_pair_disjoint_split(train, validation)


def test_pair_split_accepts_held_out_tasks(tmp_path: Path) -> None:
    train = load_human_robot_pairs(
        _write_manifest(
            tmp_path / "train",
            task="pick",
            task_index=1,
            index=1,
            pair_id="train-pair",
            human_bytes=b"train-human",
        )
    )
    validation = load_human_robot_pairs(
        _write_manifest(
            tmp_path / "validation",
            task="stack",
            task_index=2,
            index=2,
            pair_id="validation-pair",
            human_bytes=b"validation-human",
        )
    )

    split = validate_pair_disjoint_split(train, validation)

    assert split.train == train
    assert split.validation == validation
    assert split.train_digest != split.validation_digest


def test_paired_data_cli_emits_negative_safe_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        task="pick",
        task_index=1,
        index=1,
        pair_id="pair-0001",
        human_bytes=b"demo",
    )

    result = paired_data_main(["--manifest", str(manifest_path)])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["result"] == "pass"
    assert payload["artifact_kind"] == PAIRED_DATA_KIND
    assert payload["pair_count"] == 1
    assert payload["human_task_spec_declared_count"] == 0
    assert payload["semantic_match_counts"] == {"unverified": 1}
    assert payload["official_humangen"] is False
    assert payload["full_lerobot_v3_reader"] is False
