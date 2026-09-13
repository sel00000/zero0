from __future__ import annotations

from dataclasses import asdict
from hashlib import blake2b
import json
from pathlib import Path

import pytest

from so101_wam.config import (
    DEFAULT_MUJOCO_CONFIG_PATH,
    DEFAULT_ROBOT_FREE_CONFIG_PATH,
    ConfigError,
    ProjectConfig,
    SafetyConfig,
)
from so101_wam.constants import ACTION_DIM
from so101_wam.deployment import (
    canonical_json_sha256,
    project_config_fingerprint,
    project_config_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MUJOCO_OBSERVATION_LOWER = (
    -109.9998,
    -100.0,
    -96.8298,
    -94.9998,
    -157.2108,
    -0.000156261459173829,
    -109.9998,
    -100.0,
    -96.8298,
    -94.9998,
    -157.2108,
    -0.000156261459173829,
)
MUJOCO_OBSERVATION_UPPER = (
    109.9998,
    100.0,
    96.8298,
    94.9998,
    157.2108,
    100.0,
    109.9998,
    100.0,
    96.8298,
    94.9998,
    157.2108,
    100.0,
)


def _safety(**overrides: object) -> SafetyConfig:
    values = {
        "joint_lower": (-1.0,) * ACTION_DIM,
        "joint_upper": (1.0,) * ACTION_DIM,
        "max_delta_per_servo_tick": (0.1,) * ACTION_DIM,
    }
    values.update(overrides)
    return SafetyConfig(**values)


def _legacy_payload(config: ProjectConfig) -> dict[str, object]:
    payload = asdict(config)
    safety = payload["safety"]
    assert isinstance(safety, dict)
    safety.pop("observation_joint_lower", None)
    safety.pop("observation_joint_upper", None)
    return payload


@pytest.mark.parametrize(
    "overrides",
    [
        {"observation_joint_lower": (-1.0,) * ACTION_DIM},
        {"observation_joint_upper": (1.0,) * ACTION_DIM},
        {
            "observation_joint_lower": (-1.0,) * (ACTION_DIM - 1),
            "observation_joint_upper": (1.0,) * ACTION_DIM,
        },
        {
            "observation_joint_lower": (-1.0,) * ACTION_DIM,
            "observation_joint_upper": (1.0,) * (ACTION_DIM - 1),
        },
        {
            "observation_joint_lower": (-1.0,) * (ACTION_DIM - 1) + (float("nan"),),
            "observation_joint_upper": (1.0,) * ACTION_DIM,
        },
        {
            "observation_joint_lower": (-1.0,) * ACTION_DIM,
            "observation_joint_upper": (1.0,) * (ACTION_DIM - 1) + (float("inf"),),
        },
        {
            "observation_joint_lower": (0.0,) * ACTION_DIM,
            "observation_joint_upper": (0.0,) * ACTION_DIM,
        },
    ],
)
def test_observation_bounds_are_strictly_validated(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        _safety(**overrides)


def test_observation_bounds_normalize_direct_constructor_inputs() -> None:
    config = _safety(
        observation_joint_lower=[-1] * ACTION_DIM,
        observation_joint_upper=[1] * ACTION_DIM,
    )

    assert config.observation_joint_lower == (-1.0,) * ACTION_DIM
    assert config.observation_joint_upper == (1.0,) * ACTION_DIM


def test_observation_fields_do_not_shift_positional_safety_arguments() -> None:
    config = SafetyConfig(
        (-1.0,) * ACTION_DIM,
        (1.0,) * ACTION_DIM,
        (0.1,) * ACTION_DIM,
        0.2,
        0.03,
        0.4,
        True,
    )

    assert config.max_observation_age_s == 0.2
    assert config.max_camera_skew_s == 0.03
    assert config.watchdog_timeout_s == 0.4
    assert config.calibrated is True
    assert config.observation_joint_lower is None
    assert config.observation_joint_upper is None


def test_project_config_round_trips_paired_none_observation_bounds() -> None:
    config = ProjectConfig()

    round_tripped = ProjectConfig.from_mapping(asdict(config))

    assert round_tripped == config


def test_project_config_round_trips_explicit_observation_bounds() -> None:
    config = ProjectConfig(
        safety=_safety(
            observation_joint_lower=[-2] * ACTION_DIM,
            observation_joint_upper=[2] * ACTION_DIM,
        )
    )

    round_tripped = ProjectConfig.from_mapping(asdict(config))

    assert round_tripped == config


def test_legacy_none_observation_bounds_do_not_change_config_hashes() -> None:
    config = ProjectConfig(safety=_safety())
    payload = _legacy_payload(config)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert project_config_sha256(config) == canonical_json_sha256(payload)
    assert project_config_fingerprint(config) == blake2b(
        encoded,
        digest_size=16,
    ).hexdigest()


def test_explicit_observation_bounds_change_config_hashes() -> None:
    legacy = ProjectConfig(safety=_safety())
    explicit = ProjectConfig(
        safety=_safety(
            observation_joint_lower=(-2.0,) * ACTION_DIM,
            observation_joint_upper=(2.0,) * ACTION_DIM,
        )
    )

    assert project_config_sha256(explicit) != project_config_sha256(legacy)
    assert project_config_fingerprint(explicit) != project_config_fingerprint(legacy)


def test_legacy_configs_load_with_observation_bounds_unset() -> None:
    config = ProjectConfig.load(ROOT / "configs" / "fake.toml")

    assert config.safety.observation_joint_lower is None
    assert config.safety.observation_joint_upper is None


@pytest.mark.parametrize(
    ("path", "bundled_path"),
    [
        (ROOT / "configs" / "mujoco.toml", DEFAULT_MUJOCO_CONFIG_PATH),
        (ROOT / "configs" / "mujoco_robot_free.toml", DEFAULT_ROBOT_FREE_CONFIG_PATH),
    ],
)
def test_mujoco_configs_set_matching_observation_bounds(
    path: Path,
    bundled_path: Path,
) -> None:
    config = ProjectConfig.load(path)
    bundled = ProjectConfig.load(bundled_path)

    assert bundled_path.read_bytes() == path.read_bytes()
    assert config.safety.observation_joint_lower == MUJOCO_OBSERVATION_LOWER
    assert config.safety.observation_joint_upper == MUJOCO_OBSERVATION_UPPER
    assert bundled.safety.observation_joint_lower == config.safety.observation_joint_lower
    assert bundled.safety.observation_joint_upper == config.safety.observation_joint_upper
