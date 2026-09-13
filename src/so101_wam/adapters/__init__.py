"""Robot adapter boundaries.

The package remains dependency-free at import time. Hardware integrations must
be explicitly enabled by their caller.
"""

from .fake import FakeBimanualRobot, FakeWristCameras
from .lerobot import (
    ActuationDisabledError,
    AtomicObservationUnavailable,
    LeRobotBiSOAdapter,
    action_to_lerobot,
    observation_from_lerobot,
    read_lerobot_observation_atomic,
)
from .mujoco import (
    CollisionContact,
    MujocoActuationDisabledError,
    MujocoAdapterError,
    MujocoBiSOAdapter,
    MujocoCollisionError,
    MujocoContact,
    MujocoDependencyError,
    SO101MujocoCoordinates,
)

__all__ = [
    "ActuationDisabledError",
    "AtomicObservationUnavailable",
    "FakeBimanualRobot",
    "FakeWristCameras",
    "LeRobotBiSOAdapter",
    "CollisionContact",
    "MujocoActuationDisabledError",
    "MujocoAdapterError",
    "MujocoBiSOAdapter",
    "MujocoCollisionError",
    "MujocoContact",
    "MujocoDependencyError",
    "SO101MujocoCoordinates",
    "action_to_lerobot",
    "observation_from_lerobot",
    "read_lerobot_observation_atomic",
]
