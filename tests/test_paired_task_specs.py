from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.model import CompactWAM
from so101_wam.paired_data import (
    HUMAN_TASK_SPEC_FORMAT,
    LEROBOT_V3_COMPATIBLE,
    PAIRED_DATA_KIND,
    PAIRED_DATA_SCHEMA,
    PairedDataError,
    load_human_robot_pairs,
    paired_data_audit,
    paired_manifest_as_json,
)
from so101_wam.paired_task_specs import PairedTaskSpecError, human_prompt_from_pair
from so101_wam.policy import CompactWAMPolicy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    root: Path,
    *,
    prompt_extra: dict[str, np.ndarray] | None = None,
    prompt_timestamps: np.ndarray | None = None,
    include_task_spec: bool = True,
) -> Path:
    episode = EpisodeBuffer(
        fps=30.0,
        task="place block",
        task_index=9,
        episode_index=3,
        metadata={"action_source": "imported_lerobot_v3_action"},
    )
    for frame_index in range(91):
        axes = np.full(ACTION_DIM, frame_index, dtype=np.float32)
        episode.append(
            SensorimotorFrame(
                timestamp_s=frame_index / 30.0,
                images={
                    "left_wrist": np.full(
                        (8, 8, 3), frame_index, dtype=np.uint8
                    ),
                    "right_wrist": np.full(
                        (8, 8, 3), frame_index + 1, dtype=np.uint8
                    ),
                },
                joint_position=axes,
                executed_action=axes,
            )
        )
    robot_path, _ = episode.save(root / "episodes", stem="episode_000003")
    human_path = root / "human" / "place_block.mp4"
    human_path.parent.mkdir(parents=True)
    human_path.write_bytes(b"source-mp4-placeholder")

    human_video: dict[str, object] = {
        "path": human_path.relative_to(root).as_posix(),
        "sha256": _sha256(human_path),
        "view": "third_person",
    }
    if include_task_spec:
        prompt_path = root / "human" / "place_block.task_spec.npz"
        arrays = {
            "timestamp": (
                np.linspace(0.0, 3.0, 31, dtype=np.float64)
                if prompt_timestamps is None
                else prompt_timestamps
            ),
            "rgb": np.stack(
                [
                    np.full((8, 8, 3), frame_index, dtype=np.uint8)
                    for frame_index in range(31)
                ]
            ),
            **(prompt_extra or {}),
        }
        np.savez_compressed(prompt_path, **arrays)
        human_video["task_spec"] = {
            "format": HUMAN_TASK_SPEC_FORMAT,
            "path": prompt_path.relative_to(root).as_posix(),
            "sha256": _sha256(prompt_path),
        }

    manifest_path = root / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": PAIRED_DATA_SCHEMA,
                "artifact_kind": PAIRED_DATA_KIND,
                "source_format": LEROBOT_V3_COMPATIBLE,
                "source_url": "hf://datasets/example/repo",
                "pairs": [
                    {
                        "pair_id": "pair-0001",
                        "task": "place block",
                        "task_index": 9,
                        "semantic_match": "unverified",
                        "robot_episode": robot_path.relative_to(root).as_posix(),
                        "robot_episode_sha256": _sha256(robot_path),
                        "human_video": human_video,
                        "provenance": {
                            "source_dataset": "example/repo",
                            "source_layout": "meta/info.json,data/,videos/",
                            "license": "apache-2.0",
                            "transformation": (
                                "robot episode converted to local NPZ; human "
                                "frames extracted without robot signals"
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _live_frames() -> tuple[SensorimotorFrame, ...]:
    frames = []
    for index in range(4):
        axes = np.full(ACTION_DIM, index, dtype=np.float32)
        frames.append(
            SensorimotorFrame(
                timestamp_s=4.0 + index * 0.1,
                images={
                    "left_wrist": np.full((8, 8, 3), index, dtype=np.uint8),
                    "right_wrist": np.full((8, 8, 3), index + 1, dtype=np.uint8),
                },
                joint_position=axes,
                executed_action=axes,
            )
        )
    return tuple(frames)


def test_pair_task_spec_runs_through_offline_policy(tmp_path: Path) -> None:
    pair = load_human_robot_pairs(_write_bundle(tmp_path))[0]

    prompt = human_prompt_from_pair(pair)
    policy = CompactWAMPolicy(
        CompactWAM(
            latent_dim=8,
            transformer_heads=2,
            future_steps=1,
            action_horizon=2,
            action_history_steps=4,
        ),
        servo_hz=50.0,
    )
    chunk = policy.predict_task(prompt, _live_frames(), now_s=4.4)

    assert prompt.provenance.source_sha256 == pair.human_task_spec.sha256
    assert prompt.text_metadata == pair.task
    assert len(prompt.frames) == 31
    assert chunk.target_joint_position.shape == (2, ACTION_DIM)
    assert np.isfinite(chunk.target_joint_position).all()

    exported_path = tmp_path / "exported.json"
    exported_path.write_text(
        json.dumps(paired_manifest_as_json((pair,), root=tmp_path)),
        encoding="utf-8",
    )
    reloaded = load_human_robot_pairs(exported_path)[0]
    assert reloaded.human_task_spec == pair.human_task_spec


def test_pair_task_spec_rejects_robot_signal_arrays(tmp_path: Path) -> None:
    manifest_path = _write_bundle(
        tmp_path,
        prompt_extra={"action": np.zeros((31, ACTION_DIM), dtype=np.float32)},
    )
    pair = load_human_robot_pairs(manifest_path)[0]
    audit = paired_data_audit((pair,))

    assert audit["human_task_spec_declared_count"] == 1
    assert "human_task_spec_ready_count" not in audit
    with pytest.raises(PairedTaskSpecError, match="exactly timestamp and rgb"):
        human_prompt_from_pair(pair)


def test_pair_task_spec_requires_checksum_bound_artifact(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["pairs"][0]["human_video"]["task_spec"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PairedDataError, match="human task-spec checksum"):
        load_human_robot_pairs(manifest_path)

    no_task_spec = load_human_robot_pairs(
        _write_bundle(tmp_path / "without-task-spec", include_task_spec=False)
    )[0]
    with pytest.raises(PairedTaskSpecError, match="does not include"):
        human_prompt_from_pair(no_task_spec)


def test_pair_task_spec_rechecks_checksum_before_load(tmp_path: Path) -> None:
    pair = load_human_robot_pairs(_write_bundle(tmp_path))[0]
    assert pair.human_task_spec is not None

    pair.human_task_spec.path.write_bytes(b"changed-after-manifest-import")

    with pytest.raises(PairedTaskSpecError, match="checksum changed after import"):
        human_prompt_from_pair(pair)


def test_pair_task_spec_wraps_missing_artifact(tmp_path: Path) -> None:
    pair = load_human_robot_pairs(_write_bundle(tmp_path))[0]
    assert pair.human_task_spec is not None

    pair.human_task_spec.path.unlink()

    with pytest.raises(PairedTaskSpecError, match="failed to load"):
        human_prompt_from_pair(pair)


def test_pair_task_spec_rejects_complex_timestamps(tmp_path: Path) -> None:
    timestamps = np.linspace(0.0, 3.0, 31, dtype=np.complex128)
    pair = load_human_robot_pairs(
        _write_bundle(tmp_path, prompt_timestamps=timestamps)
    )[0]

    with pytest.raises(PairedTaskSpecError, match="timestamp"):
        human_prompt_from_pair(pair)
