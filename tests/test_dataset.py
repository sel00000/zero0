from __future__ import annotations

import hashlib
import json
import os
import zipfile

import numpy as np
import pytest

import so101_wam.dataset as dataset_module
from so101_wam.constants import ACTION_DIM, JOINT_KEYS, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import (
    ACTION_KEY,
    DatasetArtifactExistsError,
    DatasetError,
    EpisodeBuffer,
    EpisodeData,
    JOINT_STATE_KEY,
    SCHEMA_VERSION,
    TIMESTAMPS_KEY,
    WRIST_RGB_KEY,
    lerobot_v061_feature_mapping,
    load_episode,
    physical_prompt_from_episode,
    save_episode,
)


def make_frame(timestamp_s: float, *, include_action: bool = True, resolution: tuple[int, int] = (4, 5)) -> SensorimotorFrame:
    height, width = resolution
    action = np.full(ACTION_DIM, timestamp_s + 1.0, dtype=np.float32) if include_action else None
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={
            "left_wrist": np.full((height, width, 3), 11, dtype=np.uint8),
            "right_wrist": np.full((height, width, 3), 22, dtype=np.uint8),
        },
        joint_position=np.full(ACTION_DIM, timestamp_s, dtype=np.float32),
        executed_action=action,
    )


def test_episode_buffer_saves_loads_and_replays_round_trip(tmp_path) -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=7, task_index=3, metadata={"operator": "test"})
    buffer.append(make_frame(0.0))
    buffer.append(make_frame(0.5))
    buffer.append(make_frame(1.0))

    npz_path, manifest_path = buffer.save(tmp_path)
    loaded = load_episode(npz_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["camera_keys"] == list(PRIMARY_CAMERA_KEYS)
    assert manifest["joint_keys"] == list(JOINT_KEYS)
    assert manifest["action_keys"] == list(JOINT_KEYS)
    assert manifest["frame_count"] == 3
    assert loaded.wrist_rgb.shape == (3, 2, 4, 5, 3)
    assert loaded.joint_state.shape == (3, ACTION_DIM)
    assert loaded.action.shape == (3, ACTION_DIM)
    assert loaded.fingerprint == buffer.to_episode_data().fingerprint

    replayed = tuple(loaded.frames())
    assert [frame.timestamp_s for frame in replayed] == [0.0, 0.5, 1.0]
    np.testing.assert_array_equal(replayed[0].images["left_wrist"], np.full((4, 5, 3), 11, dtype=np.uint8))
    np.testing.assert_array_equal(replayed[-1].executed_action, np.full(ACTION_DIM, 2.0, dtype=np.float32))


def test_episode_rejects_missing_executed_action() -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=0)

    with pytest.raises(DatasetError, match="executed/sent action"):
        buffer.append(make_frame(0.0, include_action=False))


def test_episode_rejects_non_monotonic_or_wrong_fps_timestamps() -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=0)
    buffer.append(make_frame(0.5))

    with pytest.raises(DatasetError, match="strictly increasing"):
        buffer.append(make_frame(0.5))

    wrong_rate = EpisodeBuffer(fps=2.0, task="pick", episode_index=1)
    wrong_rate.append(make_frame(0.0))
    with pytest.raises(DatasetError, match="inconsistent with fps"):
        wrong_rate.append(make_frame(0.8))


def test_episode_rejects_resolution_changes() -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=0)
    buffer.append(make_frame(0.0, resolution=(4, 5)))

    with pytest.raises(DatasetError, match="resolution"):
        buffer.append(make_frame(0.5, resolution=(5, 5)))


def test_load_rejects_corrupted_npz_checksum(tmp_path) -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=0)
    buffer.append(make_frame(0.0))
    buffer.append(make_frame(0.5))
    npz_path, manifest_path = buffer.save(tmp_path)
    original = npz_path.read_bytes()
    npz_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))

    with pytest.raises(DatasetError, match="checksum mismatch"):
        load_episode(npz_path, manifest_path)


def test_no_overwrite_preserves_existing_episode_pair(tmp_path) -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=0)
    buffer.append(make_frame(0.0))
    buffer.append(make_frame(0.5))
    npz_path, manifest_path = buffer.save(tmp_path)
    original_npz = npz_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(DatasetArtifactExistsError, match="already exists"):
        buffer.save(tmp_path)

    assert npz_path.read_bytes() == original_npz
    assert manifest_path.read_bytes() == original_manifest


def test_no_overwrite_rolls_back_npz_when_manifest_publish_loses_race(
    tmp_path,
    monkeypatch,
) -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=0)
    buffer.append(make_frame(0.0))
    buffer.append(make_frame(0.5))
    real_link = os.link
    link_calls = 0

    def racing_link(source, target) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise FileExistsError(target)
        real_link(source, target)

    monkeypatch.setattr("so101_wam.dataset.os.link", racing_link)

    with pytest.raises(DatasetArtifactExistsError, match="already exists"):
        buffer.save(tmp_path)

    assert not (tmp_path / "episode_000000.npz").exists()
    assert not (tmp_path / "episode_000000.json").exists()
    assert not tuple(tmp_path.glob(".episode_000000.*"))


def test_manifest_metadata_is_covered_by_episode_fingerprint(tmp_path) -> None:
    buffer = EpisodeBuffer(
        fps=2.0,
        task="pick",
        episode_index=0,
        metadata={"action_source": "sent_goal_position"},
    )
    buffer.append(make_frame(0.0))
    buffer.append(make_frame(0.5))
    npz_path, manifest_path = buffer.save(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["action_source"] = "measured_position"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetError, match="fingerprint"):
        load_episode(npz_path, manifest_path)


@pytest.mark.parametrize("rgb_frames", [0, 1, 3])
def test_rgb_time_mismatch_ctor(rgb_frames: int) -> None:
    timestamps = np.array([0.0, 0.5], dtype=np.float64)

    with pytest.raises(DatasetError, match="wrist_rgb time dimension"):
        EpisodeData(
            timestamps_s=timestamps,
            wrist_rgb=np.zeros((rgb_frames, 2, 4, 5, 3), dtype=np.uint8),
            joint_state=np.zeros((2, ACTION_DIM), dtype=np.float32),
            action=np.zeros((2, ACTION_DIM), dtype=np.float32),
            fps=2.0,
            task="pick",
            episode_index=0,
        )


def test_rgb_time_mismatch_load(tmp_path) -> None:
    data = EpisodeData(
        timestamps_s=np.array([0.0, 0.5], dtype=np.float64),
        wrist_rgb=np.zeros((2, 2, 4, 5, 3), dtype=np.uint8),
        joint_state=np.zeros((2, ACTION_DIM), dtype=np.float32),
        action=np.zeros((2, ACTION_DIM), dtype=np.float32),
        fps=2.0,
        task="pick",
        episode_index=0,
    )
    payload_path, manifest_path = save_episode(data, tmp_path)
    rgb = np.zeros((3, 2, 4, 5, 3), dtype=np.uint8)
    with zipfile.ZipFile(payload_path, "w") as archive:
        for name, array in {
            TIMESTAMPS_KEY: data.timestamps_s,
            WRIST_RGB_KEY: rgb,
            JOINT_STATE_KEY: data.joint_state,
            ACTION_KEY: data.action,
        }.items():
            array_path = tmp_path / f"{name}.npy"
            np.save(array_path, array)
            archive.write(array_path, arcname=f"{name}.npy")
            array_path.unlink()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["npz_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest["fingerprint"] = dataset_module._array_fingerprint(
        data.timestamps_s,
        rgb,
        data.joint_state,
        data.action,
        extra={
            "fps": float(data.fps),
            "task": data.task,
            "episode_index": data.episode_index,
            "task_index": data.task_index,
            "metadata": dict(data.metadata or {}),
        },
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetError, match="wrist_rgb time dimension"):
        load_episode(payload_path, manifest_path)


def test_load_rejects_payload_missing_action_even_if_checksum_matches(tmp_path) -> None:
    data = EpisodeData(
        timestamps_s=np.array([0.0, 0.5], dtype=np.float64),
        wrist_rgb=np.zeros((2, 2, 4, 5, 3), dtype=np.uint8),
        joint_state=np.zeros((2, ACTION_DIM), dtype=np.float32),
        action=np.zeros((2, ACTION_DIM), dtype=np.float32),
        fps=2.0,
        task="pick",
        episode_index=0,
    )
    npz_path = tmp_path / "episode_000000.npz"
    manifest_path = tmp_path / "episode_000000.json"
    payload_path, manifest_path = save_episode(data, tmp_path)
    with zipfile.ZipFile(payload_path, "w") as archive:
        for name, array in {
            TIMESTAMPS_KEY: data.timestamps_s,
            WRIST_RGB_KEY: data.wrist_rgb,
            JOINT_STATE_KEY: data.joint_state,
        }.items():
            array_path = tmp_path / f"{name}.npy"
            np.save(array_path, array)
            archive.write(array_path, arcname=f"{name}.npy")
            array_path.unlink()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["npz_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert npz_path == payload_path
    with pytest.raises(DatasetError, match=ACTION_KEY):
        load_episode(payload_path, manifest_path)


def test_lerobot_v061_mapping_uses_verified_feature_names() -> None:
    buffer = EpisodeBuffer(fps=2.0, task="pick", episode_index=7, task_index=3)
    buffer.append(make_frame(0.0))
    data = buffer.to_episode_data()

    features = lerobot_v061_feature_mapping(data)

    assert features["observation.images.left_wrist"].shape == (1, 4, 5, 3)
    assert features["observation.images.right_wrist"].shape == (1, 4, 5, 3)
    assert features["observation.state"].shape == (1, ACTION_DIM)
    assert features["action"].shape == (1, ACTION_DIM)
    assert features["timestamp"].tolist() == [0.0]
    assert features["frame_index"].tolist() == [0]
    assert features["episode_index"].tolist() == [7]
    assert features["index"].tolist() == [0]
    assert features["task_index"].tolist() == [3]
    assert features["metadata"]["action_keys"] == JOINT_KEYS


def test_episode_downsamples_to_policy_rate_physical_prompt() -> None:
    buffer = EpisodeBuffer(fps=30.0, task="pick", episode_index=0)
    for index in range(91):
        buffer.append(make_frame(index / 30.0))

    prompt = physical_prompt_from_episode(buffer.to_episode_data(), policy_hz=10.0)

    assert len(prompt.frames) == 31
    assert prompt.duration_s == pytest.approx(3.0)
    assert prompt.frames[0].timestamp_s == pytest.approx(0.0)
    assert prompt.frames[-1].timestamp_s == pytest.approx(3.0)
    assert all(frame.executed_action is not None for frame in prompt.frames)


def test_episode_prompt_downsampling_rejects_upsampling_and_bad_duration() -> None:
    short = EpisodeBuffer(fps=2.0, task="pick", episode_index=0)
    for index in range(5):
        short.append(make_frame(index / 2.0))
    data = short.to_episode_data()

    with pytest.raises(DatasetError, match="cannot exceed"):
        physical_prompt_from_episode(data, policy_hz=3.0)
    with pytest.raises(DatasetError, match="cannot exceed"):
        physical_prompt_from_episode(data, policy_hz=2.01)
    with pytest.raises(DatasetError, match="physical prompt"):
        physical_prompt_from_episode(data, policy_hz=1.0)
