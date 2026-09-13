"""Deterministic fake bimanual robot and wrist cameras for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Mapping

import numpy as np

from so101_wam.constants import ACTION_DIM, JOINT_KEYS, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import ActionChunk, SensorimotorFrame


@dataclass(slots=True)
class FakeWristCameras:
    height: int = 8
    width: int = 8

    def capture(self, frame_index: int = 0) -> dict[str, np.ndarray]:
        if self.height < 2 or self.width < 2:
            raise ValueError("fake camera frames must be at least 2x2")
        images: dict[str, np.ndarray] = {}
        for camera_index, key in enumerate(PRIMARY_CAMERA_KEYS):
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image[..., 0] = (frame_index + camera_index * 37) % 256
            image[..., 1] = np.arange(self.height, dtype=np.uint8).reshape(
                self.height, 1
            )
            image[..., 2] = np.arange(self.width, dtype=np.uint8).reshape(1, self.width)
            images[key] = image
        return images


@dataclass(slots=True)
class FakeBimanualRobot:
    backend: ClassVar[str] = "fake"
    cameras: FakeWristCameras = field(default_factory=FakeWristCameras)
    joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32),
    )
    sent_actions: list[np.ndarray] = field(default_factory=list)
    frame_index: int = 0

    def __post_init__(self) -> None:
        self.joint_position = np.array(self.joint_position, dtype=np.float32, copy=True)
        if self.joint_position.shape != (ACTION_DIM,):
            raise ValueError(f"joint_position must have shape ({ACTION_DIM},)")
        if not np.isfinite(self.joint_position).all():
            raise ValueError("joint_position contains NaN or infinity")

    def get_observation(self, *, timestamp_s: float) -> SensorimotorFrame:
        images = self.cameras.capture(self.frame_index)
        frame = SensorimotorFrame(
            timestamp_s=timestamp_s,
            images=images,
            joint_position=self.joint_position,
            image_timestamps_s={key: timestamp_s for key in images},
        )
        self.frame_index += 1
        return frame

    def get_lerobot_observation(self) -> dict[str, object]:
        observation: dict[str, object] = {
            key: float(value)
            for key, value in zip(JOINT_KEYS, self.joint_position, strict=True)
        }
        observation.update(self.cameras.capture(self.frame_index))
        self.frame_index += 1
        return observation

    def send_action(
        self, action: Mapping[str, float] | ActionChunk | np.ndarray
    ) -> dict[str, float]:
        if isinstance(action, Mapping):
            target = np.array([action[key] for key in JOINT_KEYS], dtype=np.float32)
        elif isinstance(action, ActionChunk):
            if action.horizon != 1:
                raise ValueError(
                    "ActionChunk horizon must be 1; send one row per servo tick"
                )
            target = np.array(
                action.target_joint_position[0], dtype=np.float32, copy=True
            )
        else:
            target = np.array(action, dtype=np.float32, copy=True)
        if target.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},)")
        if not np.isfinite(target).all():
            raise ValueError("action contains NaN or infinity")

        self.joint_position = target
        self.sent_actions.append(target.copy())
        return {
            key: float(value) for key, value in zip(JOINT_KEYS, target, strict=True)
        }


__all__ = ["FakeBimanualRobot", "FakeWristCameras"]
