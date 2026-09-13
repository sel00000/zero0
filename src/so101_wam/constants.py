"""Canonical names shared by datasets, policies, and LeRobot adapters."""

ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

ARM_SIDES: tuple[str, str] = ("left", "right")

JOINT_KEYS: tuple[str, ...] = tuple(
    f"{side}_{joint}.pos" for side in ARM_SIDES for joint in ARM_JOINT_NAMES
)

PRIMARY_CAMERA_KEYS: tuple[str, str] = ("left_wrist", "right_wrist")
OPTIONAL_HEAD_CAMERA_KEY = "head_optional"

ACTION_DIM = len(JOINT_KEYS)
PRIMARY_CAMERA_COUNT = len(PRIMARY_CAMERA_KEYS)

LEROBOT_VERSION_PIN = "0.6.1"
LEROBOT_GIT_COMMIT = "7e241bd630a3719a56157a497ce5d08f244784f1"
