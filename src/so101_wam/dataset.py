"""Episode storage for synchronized dual-wrist sensorimotor trajectories."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import blake2b, sha256
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray

from .constants import ACTION_DIM, JOINT_KEYS, LEROBOT_VERSION_PIN, PRIMARY_CAMERA_KEYS
from .contracts import PhysicalPrompt, SensorimotorFrame

SCHEMA_VERSION = "so101_wam.episode.v1"
FPS_RELATIVE_TOLERANCE = 0.05

TIMESTAMPS_KEY = "timestamp"
WRIST_RGB_KEY = "wrist_rgb"
JOINT_STATE_KEY = "joint_state"
ACTION_KEY = "action"

LEROBOT_FEATURE_NAMES: Mapping[str, str] = {
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
    "state": "observation.state",
    "action": "action",
    "timestamp": "timestamp",
    "frame_index": "frame_index",
    "episode_index": "episode_index",
    "index": "index",
    "task_index": "task_index",
}


class DatasetError(ValueError):
    """Raised when an episode cannot be serialized or replayed safely."""


class DatasetArtifactExistsError(DatasetError):
    """Raised when no-overwrite publication finds an existing artifact."""


@dataclass(frozen=True, slots=True)
class EpisodeData:
    """Validated columnar episode arrays and JSON-serializable metadata."""

    timestamps_s: NDArray[np.float64]
    wrist_rgb: NDArray[np.uint8]
    joint_state: NDArray[np.float32]
    action: NDArray[np.float32]
    fps: float
    task: str
    episode_index: int
    task_index: int = 0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        timestamps = np.array(self.timestamps_s, dtype=np.float64, copy=True)
        wrist_rgb = np.array(self.wrist_rgb, copy=True)
        joint_state = np.array(self.joint_state, dtype=np.float32, copy=True)
        action = np.array(self.action, dtype=np.float32, copy=True)

        _validate_arrays(
            timestamps_s=timestamps,
            wrist_rgb=wrist_rgb,
            joint_state=joint_state,
            action=action,
            fps=self.fps,
        )
        if not isinstance(self.task, str) or not self.task:
            raise DatasetError("task must be a non-empty string")
        if int(self.episode_index) < 0:
            raise DatasetError("episode_index must be non-negative")
        if int(self.task_index) < 0:
            raise DatasetError("task_index must be non-negative")

        for array in (timestamps, wrist_rgb, joint_state, action):
            array.flags.writeable = False
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "wrist_rgb", wrist_rgb)
        object.__setattr__(self, "joint_state", joint_state)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "episode_index", int(self.episode_index))
        object.__setattr__(self, "task_index", int(self.task_index))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata or {}, name="metadata"))

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_s.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.wrist_rgb.shape[2]), int(self.wrist_rgb.shape[3])

    @property
    def fingerprint(self) -> str:
        return _array_fingerprint(
            self.timestamps_s,
            self.wrist_rgb,
            self.joint_state,
            self.action,
            extra={
                "fps": float(self.fps),
                "task": self.task,
                "episode_index": self.episode_index,
                "task_index": self.task_index,
                "metadata": dict(self.metadata or {}),
            },
        )

    @property
    def content_fingerprint(self) -> str:
        """Hash only sensorimotor content for duplicate trajectory checks."""

        return _array_fingerprint(
            self.timestamps_s,
            self.wrist_rgb,
            self.joint_state,
            self.action,
            extra={"fps": float(self.fps)},
        )

    def frames(self) -> Iterator[SensorimotorFrame]:
        for index, timestamp_s in enumerate(self.timestamps_s):
            yield SensorimotorFrame(
                timestamp_s=float(timestamp_s),
                images={
                    PRIMARY_CAMERA_KEYS[0]: self.wrist_rgb[index, 0],
                    PRIMARY_CAMERA_KEYS[1]: self.wrist_rgb[index, 1],
                },
                joint_position=self.joint_state[index],
                executed_action=self.action[index],
            )


class EpisodeBuffer:
    """In-memory builder for one fixed-rate dual-wrist episode."""

    def __init__(
        self,
        *,
        fps: float,
        task: str,
        episode_index: int,
        task_index: int = 0,
        metadata: Mapping[str, Any] | None = None,
        fps_tolerance: float = FPS_RELATIVE_TOLERANCE,
    ) -> None:
        if not isfinite(fps) or fps <= 0:
            raise DatasetError("fps must be finite and positive")
        if not 0 <= fps_tolerance < 1:
            raise DatasetError("fps_tolerance must be in [0, 1)")
        self.fps = float(fps)
        self.task = task
        self.episode_index = int(episode_index)
        self.task_index = int(task_index)
        self.metadata = dict(metadata or {})
        self.fps_tolerance = float(fps_tolerance)
        self._frames: list[SensorimotorFrame] = []
        self._resolution: tuple[int, int] | None = None

    def append(self, frame: SensorimotorFrame) -> None:
        if frame.executed_action is None:
            raise DatasetError("every dataset frame must include the executed/sent action")
        if self._frames and frame.timestamp_s <= self._frames[-1].timestamp_s:
            raise DatasetError("timestamps must be strictly increasing")

        left, right = frame.primary_images
        if left.shape != right.shape:
            raise DatasetError("left and right wrist images must share one resolution")
        resolution = (int(left.shape[0]), int(left.shape[1]))
        if self._resolution is None:
            self._resolution = resolution
        elif resolution != self._resolution:
            raise DatasetError(f"all wrist frames must share resolution {self._resolution}, got {resolution}")

        if self._frames:
            expected_dt_s = 1.0 / self.fps
            actual_dt_s = frame.timestamp_s - self._frames[-1].timestamp_s
            if abs(actual_dt_s - expected_dt_s) > expected_dt_s * self.fps_tolerance:
                raise DatasetError(
                    f"timestamp step {actual_dt_s:.6f}s is inconsistent with fps {self.fps:g}"
                )

        self._frames.append(frame)

    def extend(self, frames: Iterator[SensorimotorFrame] | tuple[SensorimotorFrame, ...]) -> None:
        for frame in frames:
            self.append(frame)

    def to_episode_data(self) -> EpisodeData:
        if not self._frames:
            raise DatasetError("episode must contain at least one frame")
        timestamps = np.array([frame.timestamp_s for frame in self._frames], dtype=np.float64)
        wrist_rgb = np.stack(
            [np.stack(frame.primary_images, axis=0) for frame in self._frames],
            axis=0,
        )
        joint_state = np.stack([frame.joint_position for frame in self._frames], axis=0).astype(np.float32)
        action_values = []
        for frame in self._frames:
            if frame.executed_action is None:
                raise DatasetError("every dataset frame must include the executed/sent action")
            action_values.append(frame.executed_action)
        action = np.stack(action_values, axis=0).astype(np.float32)
        return EpisodeData(
            timestamps_s=timestamps,
            wrist_rgb=wrist_rgb,
            joint_state=joint_state,
            action=action,
            fps=self.fps,
            task=self.task,
            episode_index=self.episode_index,
            task_index=self.task_index,
            metadata=self.metadata,
        )

    def save(
        self,
        directory: str | Path,
        *,
        stem: str | None = None,
    ) -> tuple[Path, Path]:
        return save_episode(
            self.to_episode_data(),
            directory,
            stem=stem,
        )


def save_episode(
    data: EpisodeData,
    directory: str | Path,
    *,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Validate and immutably publish one NPZ payload plus its JSON manifest.

    Same-directory hard links provide race-safe no-overwrite publication. If the
    second link cannot be published, the first link is rolled back when it still
    points at this writer's temporary inode. A new episode requires a new stem.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_stem = stem or f"episode_{data.episode_index:06d}"
    if (
        not safe_stem
        or Path(safe_stem).name != safe_stem
        or safe_stem in {".", ".."}
    ):
        raise DatasetError("episode stem must be one non-empty filename component")
    npz_path = directory / f"{safe_stem}.npz"
    manifest_path = directory / f"{safe_stem}.json"
    if npz_path.exists() or manifest_path.exists():
        raise DatasetArtifactExistsError(
            "episode artifact already exists; choose a new episode stem: "
            f"{npz_path}, {manifest_path}"
        )

    with NamedTemporaryFile(
        dir=directory,
        prefix=f".{safe_stem}.",
        suffix=".npz",
        delete=False,
    ) as tmp:
        tmp_npz_path = Path(tmp.name)
    try:
        with NamedTemporaryFile(
            dir=directory,
            prefix=f".{safe_stem}.",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp_manifest_path = Path(tmp.name)
    except BaseException:
        tmp_npz_path.unlink(missing_ok=True)
        raise
    try:
        with tmp_npz_path.open("wb") as npz_file:
            np.savez_compressed(
                npz_file,
                timestamp=data.timestamps_s,
                wrist_rgb=data.wrist_rgb,
                joint_state=data.joint_state,
                action=data.action,
            )
            npz_file.flush()
            os.fsync(npz_file.fileno())

        manifest = _build_manifest(data, npz_sha256=_file_sha256(tmp_npz_path))
        _write_json_file(tmp_manifest_path, manifest)

        loaded = load_episode(tmp_npz_path, tmp_manifest_path)
        if loaded.fingerprint != data.fingerprint:
            raise DatasetError("round-trip validation failed: fingerprint mismatch")

        _publish_no_overwrite_pair(
            tmp_npz_path=tmp_npz_path,
            tmp_manifest_path=tmp_manifest_path,
            npz_path=npz_path,
            manifest_path=manifest_path,
        )
    finally:
        tmp_npz_path.unlink(missing_ok=True)
        tmp_manifest_path.unlink(missing_ok=True)

    loaded = load_episode(npz_path, manifest_path)
    if loaded.fingerprint != data.fingerprint:
        raise DatasetError("published round-trip validation failed: fingerprint mismatch")
    return npz_path, manifest_path


def load_episode(npz_path: str | Path, manifest_path: str | Path | None = None) -> EpisodeData:
    """Load and validate an episode payload against its manifest."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else npz_path.with_suffix(".json")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError("unsupported episode schema version")
    if manifest.get("npz_sha256") != _file_sha256(npz_path):
        raise DatasetError("npz checksum mismatch")

    try:
        with np.load(npz_path) as payload:
            required = {TIMESTAMPS_KEY, WRIST_RGB_KEY, JOINT_STATE_KEY, ACTION_KEY}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise DatasetError(f"episode payload missing array(s): {missing}")
            data = EpisodeData(
                timestamps_s=payload[TIMESTAMPS_KEY],
                wrist_rgb=payload[WRIST_RGB_KEY],
                joint_state=payload[JOINT_STATE_KEY],
                action=payload[ACTION_KEY],
                fps=float(manifest["fps"]),
                task=str(manifest["task"]),
                episode_index=int(manifest["episode_index"]),
                task_index=int(manifest["task_index"]),
                metadata=dict(manifest.get("metadata", {})),
            )
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"failed to load episode payload: {exc}") from exc

    _validate_manifest_matches_data(manifest, data)
    return data


def physical_prompt_from_episode(
    data: EpisodeData,
    *,
    policy_hz: float,
) -> PhysicalPrompt:
    """Downsample one 3-12 second recorded episode into a policy-rate prompt.

    Source frames are selected by nearest timestamp without interpolating pixels,
    joint states, or executed actions. The first and final recorded frames are
    always retained so the validated prompt duration remains auditable.
    """

    if not isfinite(policy_hz) or policy_hz <= 0:
        raise DatasetError("policy_hz must be finite and positive")
    if policy_hz > data.fps + 1e-9:
        raise DatasetError(
            f"policy_hz {policy_hz:g} cannot exceed recorded fps {data.fps:g} without interpolation"
        )

    start_s = float(data.timestamps_s[0])
    end_s = float(data.timestamps_s[-1])
    step_s = 1.0 / float(policy_hz)
    target_count = int(np.floor((end_s - start_s) / step_s + 1e-9)) + 1
    target_timestamps = start_s + np.arange(target_count, dtype=np.float64) * step_s

    selected_indices: list[int] = [0]
    for target_s in target_timestamps[1:]:
        insertion = int(np.searchsorted(data.timestamps_s, target_s, side="left"))
        candidates = tuple(index for index in (insertion - 1, insertion) if 0 <= index < data.frame_count)
        nearest = min(candidates, key=lambda index: abs(float(data.timestamps_s[index]) - float(target_s)))
        if nearest != selected_indices[-1]:
            selected_indices.append(nearest)
    if selected_indices[-1] != data.frame_count - 1:
        selected_indices.append(data.frame_count - 1)

    frames = tuple(data.frames())
    try:
        return PhysicalPrompt(tuple(frames[index] for index in selected_indices))
    except ValueError as error:
        raise DatasetError(f"episode cannot be used as a physical prompt: {error}") from error


def lerobot_v061_feature_mapping(data: EpisodeData) -> dict[str, Any]:
    """Return LeRobot v0.6.1 feature-name arrays without importing lerobot."""

    frame_index = np.arange(data.frame_count, dtype=np.int64)
    return {
        LEROBOT_FEATURE_NAMES["left_wrist"]: data.wrist_rgb[:, 0],
        LEROBOT_FEATURE_NAMES["right_wrist"]: data.wrist_rgb[:, 1],
        LEROBOT_FEATURE_NAMES["state"]: data.joint_state,
        LEROBOT_FEATURE_NAMES["action"]: data.action,
        LEROBOT_FEATURE_NAMES["timestamp"]: data.timestamps_s,
        LEROBOT_FEATURE_NAMES["frame_index"]: frame_index,
        LEROBOT_FEATURE_NAMES["episode_index"]: np.full(data.frame_count, data.episode_index, dtype=np.int64),
        LEROBOT_FEATURE_NAMES["index"]: frame_index.copy(),
        LEROBOT_FEATURE_NAMES["task_index"]: np.full(data.frame_count, data.task_index, dtype=np.int64),
        "metadata": {
            "lerobot_version": LEROBOT_VERSION_PIN,
            "task": data.task,
            "camera_keys": PRIMARY_CAMERA_KEYS,
            "joint_keys": JOINT_KEYS,
            "action_keys": JOINT_KEYS,
        },
    }


def _validate_arrays(
    *,
    timestamps_s: NDArray[np.float64],
    wrist_rgb: NDArray[np.generic],
    joint_state: NDArray[np.float32],
    action: NDArray[np.float32],
    fps: float,
) -> None:
    if not isfinite(fps) or fps <= 0:
        raise DatasetError("fps must be finite and positive")
    if timestamps_s.ndim != 1 or timestamps_s.shape[0] < 1:
        raise DatasetError("timestamp must have shape [T] with T >= 1")
    if not np.isfinite(timestamps_s).all() or np.any(timestamps_s < 0):
        raise DatasetError("timestamp contains NaN, infinity, or negative values")
    if np.any(np.diff(timestamps_s) <= 0):
        raise DatasetError("timestamps must be strictly increasing")
    if wrist_rgb.dtype != np.uint8:
        raise DatasetError(f"wrist_rgb must use uint8 RGB pixels, got {wrist_rgb.dtype}")
    if wrist_rgb.ndim != 5 or wrist_rgb.shape[1] != len(PRIMARY_CAMERA_KEYS) or wrist_rgb.shape[-1] != 3:
        raise DatasetError(f"wrist_rgb must have shape [T, 2, H, W, 3], got {wrist_rgb.shape}")
    if wrist_rgb.shape[0] != timestamps_s.shape[0]:
        raise DatasetError(
            "wrist_rgb time dimension must match timestamp length, "
            f"got {wrist_rgb.shape[0]} and {timestamps_s.shape[0]}"
        )
    if wrist_rgb.shape[2] < 2 or wrist_rgb.shape[3] < 2:
        raise DatasetError(f"wrist_rgb resolution is too small: {wrist_rgb.shape[2:4]}")
    if joint_state.shape != (timestamps_s.shape[0], ACTION_DIM):
        raise DatasetError(f"joint_state must have shape [T, {ACTION_DIM}], got {joint_state.shape}")
    if action.shape != (timestamps_s.shape[0], ACTION_DIM):
        raise DatasetError(f"action must have shape [T, {ACTION_DIM}], got {action.shape}")
    if not np.isfinite(joint_state).all():
        raise DatasetError("joint_state contains NaN or infinity")
    if not np.isfinite(action).all():
        raise DatasetError("action contains NaN or infinity")
    if timestamps_s.shape[0] > 1:
        expected_dt_s = 1.0 / fps
        max_error_s = expected_dt_s * FPS_RELATIVE_TOLERANCE
        if np.any(np.abs(np.diff(timestamps_s) - expected_dt_s) > max_error_s):
            raise DatasetError(f"timestamps are inconsistent with fps {fps:g}")


def _build_manifest(data: EpisodeData, *, npz_sha256: str) -> dict[str, Any]:
    height, width = data.resolution
    return {
        "schema_version": SCHEMA_VERSION,
        "task": data.task,
        "episode_index": data.episode_index,
        "task_index": data.task_index,
        "fps": float(data.fps),
        "fps_tolerance": FPS_RELATIVE_TOLERANCE,
        "frame_count": data.frame_count,
        "resolution": {"height": height, "width": width},
        "camera_keys": list(PRIMARY_CAMERA_KEYS),
        "joint_keys": list(JOINT_KEYS),
        "action_keys": list(JOINT_KEYS),
        "array_keys": {
            "timestamps_s": TIMESTAMPS_KEY,
            "wrist_rgb": WRIST_RGB_KEY,
            "joint_state": JOINT_STATE_KEY,
            "action": ACTION_KEY,
        },
        "npz_sha256": npz_sha256,
        "fingerprint": data.fingerprint,
        "start_timestamp_s": float(data.timestamps_s[0]),
        "end_timestamp_s": float(data.timestamps_s[-1]),
        "duration_s": float(data.timestamps_s[-1] - data.timestamps_s[0]),
        "metadata": dict(data.metadata or {}),
    }


def _validate_manifest_matches_data(manifest: Mapping[str, Any], data: EpisodeData) -> None:
    height, width = data.resolution
    expected = {
        "frame_count": data.frame_count,
        "camera_keys": list(PRIMARY_CAMERA_KEYS),
        "joint_keys": list(JOINT_KEYS),
        "action_keys": list(JOINT_KEYS),
        "fingerprint": data.fingerprint,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise DatasetError(f"manifest {key} does not match episode payload")
    if manifest.get("resolution") != {"height": height, "width": width}:
        raise DatasetError("manifest resolution does not match episode payload")
    if float(manifest.get("fps", -1.0)) != float(data.fps):
        raise DatasetError("manifest fps does not match episode payload")


def _array_fingerprint(*arrays: NDArray[np.generic], extra: Mapping[str, Any]) -> str:
    digest = blake2b(digest_size=16)
    digest.update(json.dumps(extra, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json_safe_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(dict(value), sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{name} must be JSON serializable") from exc
    if not isinstance(decoded, dict):
        raise DatasetError(f"{name} must be a JSON object")
    return decoded


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_file(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _same_inode(left: Path, right: Path) -> bool:
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except FileNotFoundError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def _publish_no_overwrite_pair(
    *,
    tmp_npz_path: Path,
    tmp_manifest_path: Path,
    npz_path: Path,
    manifest_path: Path,
) -> None:
    npz_published = False
    try:
        os.link(tmp_npz_path, npz_path)
        npz_published = True
        os.link(tmp_manifest_path, manifest_path)
    except Exception as error:
        rollback_error: Exception | None = None
        if npz_published and _same_inode(tmp_npz_path, npz_path):
            try:
                npz_path.unlink()
            except Exception as cleanup_error:  # pragma: no cover - filesystem fault
                rollback_error = cleanup_error
        if rollback_error is not None:
            raise DatasetError(
                "episode publication failed and NPZ rollback also failed"
            ) from rollback_error
        if isinstance(error, FileExistsError):
            raise DatasetArtifactExistsError(
                "episode artifact already exists; choose a new episode stem: "
                f"{npz_path}, {manifest_path}"
            ) from error
        raise


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise DatasetError(f"failed to read episode manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetError("episode manifest must be a JSON object")
    return value


__all__ = [
    "ACTION_KEY",
    "DatasetArtifactExistsError",
    "DatasetError",
    "EpisodeBuffer",
    "EpisodeData",
    "FPS_RELATIVE_TOLERANCE",
    "JOINT_STATE_KEY",
    "LEROBOT_FEATURE_NAMES",
    "SCHEMA_VERSION",
    "TIMESTAMPS_KEY",
    "WRIST_RGB_KEY",
    "lerobot_v061_feature_mapping",
    "load_episode",
    "physical_prompt_from_episode",
    "save_episode",
]
