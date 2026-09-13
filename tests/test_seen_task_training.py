from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_wam.checkpoint import load_compact_wam_bundle
from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.deployment import canonical_json_sha256
from so101_wam.hardware_cli import HardwareCLIError, _require_deployment_checkpoint
from so101_wam.model import ActionRangeConstraint
from so101_wam.seen_task_training import (
    SeenTaskTrainingError,
    _action_output_contract,
    train_seen_task_candidate,
)
from so101_wam.training import CompactWAMTrainingConfig
from so101_wam.training_data import (
    ACTION_TIMING_KEY,
    INITIAL_PREVIOUS_ACTION_KEY,
    OBSERVATION_THEN_COMMAND,
    load_episode_records,
)


def _config() -> CompactWAMTrainingConfig:
    return CompactWAMTrainingConfig(
        latent_dim=8,
        transformer_heads=2,
        future_steps=1,
        action_horizon=2,
        action_history_steps=1,
        ifp_steps=0,
        stage1_steps=1,
        stage2_steps=2,
        seed=13,
    )


def test_seen_task_trains(tmp_path: Path) -> None:
    records = _records(tmp_path / "episodes")
    checkpoint_path = tmp_path / "seen.pt"
    report_path = tmp_path / "seen.training.json"
    torch_state = torch.random.get_rng_state().clone()
    np_state = np.random.get_state()

    artifacts = train_seen_task_candidate(
        records,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        checkpoint_id="seen-task-001",
        config=_config(),
    )

    assert torch.equal(torch.random.get_rng_state(), torch_state)
    _assert_np_state(np.random.get_state(), np_state)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_core = {
        key: value
        for key, value in report.items()
        if key not in {"training_evidence_sha256", "artifacts"}
    }
    bundle = load_compact_wam_bundle(checkpoint_path)

    assert artifacts.checkpoint_id == "seen-task-001"
    assert artifacts.optimizer_steps == 3
    assert report["artifact_kind"] == "compact_wam_seen_task_diagnostic"
    assert report["trained"] is False
    assert report["offline_trained"] is True
    assert report["deployment_ready"] is False
    assert report["zero_shot_claimed"] is False
    assert "validation" not in report
    assert report["protocol"]["split"] == "same_task_reconstruction"
    assert report["protocol"]["real_output_authorized"] is False
    assert report["data"]["episode_count"] == 2
    assert report["data"]["source_kind"] == "mujoco_reference"
    assert report["data"]["action_timing"] == OBSERVATION_THEN_COMMAND
    assert report["optimization"]["optimizer_steps"] == 3
    assert report["reconstruction"]["pre"]["action_mse_normalized"] >= 0.0
    assert report["reconstruction"]["post"]["action_mse_normalized"] >= 0.0
    assert report["training_evidence_sha256"] == canonical_json_sha256(report_core)
    assert bundle.metadata["artifact_kind"] == "compact_wam_seen_task_diagnostic"
    assert bundle.metadata["trained"] is False
    assert bundle.metadata["deployment_ready"] is False
    assert bundle.metadata["action_history_steps"] == bundle.architecture[
        "action_history_steps"
    ]
    with pytest.raises(HardwareCLIError, match="offline"):
        _require_deployment_checkpoint(bundle.metadata)


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_seen_task_bounded_output(tmp_path: Path, mode: str) -> None:
    assert mode in {item.value for item in ActionRangeConstraint}
    constraint = ActionRangeConstraint(mode)
    records = _records(tmp_path / "episodes")
    checkpoint = tmp_path / "bounded.pt"
    report_path = tmp_path / "bounded.json"
    lower, upper = (0.0,) * ACTION_DIM, (100.0,) * ACTION_DIM
    train_seen_task_candidate(
        records, checkpoint_path=checkpoint, report_path=report_path,
        checkpoint_id="bounded", config=_config(),
        action_range_constraint=constraint,
        joint_lower=lower, joint_upper=upper,
    )
    bundle = load_compact_wam_bundle(checkpoint)
    report = json.loads(report_path.read_text())
    assert bundle.model.action_range_constraint is constraint
    assert bundle.metadata["action_range_constraint"] == mode
    assert report["schema_version"] == "so101_wam.seen_task_diagnostic.v2"
    assert report["action_output"]["joint_lower"] == list(lower)
    assert report["action_output"]["joint_upper"] == list(upper)
    assert report["action_output"]["mode"] == mode
    assert bundle.metadata["action_output_sha256"] == canonical_json_sha256(report["action_output"])
    assert "content_sha256" not in report["data"]
    assert report["data"]["content_fingerprint_algorithm"] == "blake2b-128"
    assert all(len(value) == 32 for value in report["data"]["content_fingerprints"])
    for phase in ("pre", "post"):
        metrics = report["reconstruction"][phase]
        assert metrics["action_out_of_bounds_count"] == 0
        assert metrics["action_mae_native_axis_11"] >= 0
    assert report["trained"] is False and report["deployment_ready"] is False


@pytest.mark.parametrize("mode", ["affine_tanh", "normalized_clamp"])
def test_output_space_metadata(mode: str) -> None:
    assert mode in {item.value for item in ActionRangeConstraint}
    contract = _action_output_contract(
        ActionRangeConstraint(mode), (0.0,) * ACTION_DIM, (100.0,) * ACTION_DIM,
    )
    assert contract["safety_supervisor"] == "unchanged"
    if mode == "affine_tanh":
        assert contract["head_space"] == "logits"
        assert contract["endpoint_limitation"] == "finite mathematical logits cannot reach exact endpoints"
        return

    assert contract["head_space"] == "normalized_actions"
    assert contract["endpoint_limitation"] == "only float32-representable native endpoints are exact"
    assert contract["gradient_limitation"] == "zero outside the projected interval; no straight-through estimator"


def test_seen_task_bad_labels(tmp_path: Path) -> None:
    from so101_wam.model import ActionRangeConstraint

    with pytest.raises(SeenTaskTrainingError, match="actions.*bounds"):
        train_seen_task_candidate(
            _records(tmp_path / "episodes"), checkpoint_path=tmp_path / "bad.pt",
            report_path=tmp_path / "bad.json", checkpoint_id="bad", config=_config(),
            action_range_constraint=ActionRangeConstraint.AFFINE_TANH,
            joint_lower=(0.0,) * ACTION_DIM, joint_upper=(1.0,) * ACTION_DIM,
        )
    assert not (tmp_path / "bad.pt").exists()
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("case", ["missing", "inverted", "nonfinite", "ignored", "string"])
def test_seen_output_contract(tmp_path: Path, case: str) -> None:
    from so101_wam.model import ActionRangeConstraint

    options: dict[str, object] = {
        "action_range_constraint": ActionRangeConstraint.AFFINE_TANH,
        "joint_lower": (0.0,) * ACTION_DIM,
        "joint_upper": (100.0,) * ACTION_DIM,
    }
    if case == "missing":
        options.pop("joint_upper")
    elif case == "inverted":
        options["joint_upper"] = (-1.0,) * ACTION_DIM
    elif case == "nonfinite":
        options["joint_upper"] = (float("inf"),) * ACTION_DIM
    elif case == "ignored":
        options["action_range_constraint"] = ActionRangeConstraint.UNBOUNDED
    else:
        options["action_range_constraint"] = "affine_tanh"
    with pytest.raises(SeenTaskTrainingError):
        train_seen_task_candidate(
            _records(tmp_path / "episodes"), checkpoint_path=tmp_path / "bad.pt",
            report_path=tmp_path / "bad.json", checkpoint_id="bad", config=_config(),
            **options,
        )
    assert not (tmp_path / "bad.pt").exists()


@pytest.mark.parametrize(
    ("metadata", "task_index", "count", "match"),
    [
        ({"source_kind": "synthetic_fixture"}, 1, 2, "source_kind"),
        ({"reference_success": False}, 1, 2, "reference_success"),
        ({ACTION_TIMING_KEY: "omit"}, 1, 2, "action_timing"),
        ({}, 2, 2, "same task"),
        ({}, 1, 1, "at least two"),
        ({}, 1, 0, "at least two"),
    ],
)
def test_seen_task_reject_inputs(
    tmp_path: Path,
    metadata: dict[str, object],
    task_index: int,
    count: int,
    match: str,
) -> None:
    records = _records(
        tmp_path / "episodes",
        second_metadata=metadata,
        second_task_index=task_index,
    )[:count]

    with pytest.raises(SeenTaskTrainingError, match=match):
        train_seen_task_candidate(
            records,
            checkpoint_path=tmp_path / "bad.pt",
            report_path=tmp_path / "bad.json",
            checkpoint_id="bad",
            config=_config(),
        )


def test_seen_task_no_overwrite(tmp_path: Path) -> None:
    records = _records(tmp_path / "episodes")
    checkpoint_path = tmp_path / "seen.pt"
    report_path = tmp_path / "seen.json"
    checkpoint_path.write_bytes(b"keep")

    with pytest.raises(SeenTaskTrainingError, match="immutable"):
        train_seen_task_candidate(
            records,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            checkpoint_id="seen-task-001",
            config=_config(),
        )

    assert checkpoint_path.read_bytes() == b"keep"
    assert not report_path.exists()


def test_seen_task_restores_warn(tmp_path: Path) -> None:
    records = _records(tmp_path / "episodes")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        train_seen_task_candidate(
            records,
            checkpoint_path=tmp_path / "seen.pt",
            report_path=tmp_path / "seen.json",
            checkpoint_id="seen-task-001",
            config=_config(),
        )

        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)


def test_seen_task_restores_warn_err(tmp_path: Path) -> None:
    records = _records(tmp_path / "episodes")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with pytest.raises(SeenTaskTrainingError, match="requires cpu"):
            train_seen_task_candidate(
                records,
                checkpoint_path=tmp_path / "seen.pt",
                report_path=tmp_path / "seen.json",
                checkpoint_id="seen-task-001",
                config=_config(),
                device="meta",
            )

        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)


def test_seen_task_reject_dupe(tmp_path: Path) -> None:
    records = _records(tmp_path / "episodes", second_offset=0.0)

    with pytest.raises(SeenTaskTrainingError, match="distinct contents"):
        train_seen_task_candidate(
            records,
            checkpoint_path=tmp_path / "dupe.pt",
            report_path=tmp_path / "dupe.json",
            checkpoint_id="dupe",
            config=_config(),
        )


def _records(
    root: Path,
    *,
    second_metadata: dict[str, object] | None = None,
    second_task_index: int = 1,
    second_offset: float = 10.0,
) -> tuple[object, ...]:
    _save_episode(root, episode_index=0, action_offset=0.0)
    _save_episode(
        root,
        episode_index=1,
        action_offset=second_offset,
        metadata=second_metadata,
        task_index=second_task_index,
    )
    return load_episode_records((root,))


def _save_episode(
    root: Path,
    *,
    episode_index: int,
    action_offset: float,
    metadata: dict[str, object] | None = None,
    task_index: int = 1,
) -> None:
    merged = {
        "source_kind": "mujoco_reference",
        "reference_success": True,
        ACTION_TIMING_KEY: OBSERVATION_THEN_COMMAND,
        INITIAL_PREVIOUS_ACTION_KEY: _axis(-1.0).tolist(),
        **(metadata or {}),
    }
    if merged.get(ACTION_TIMING_KEY) == "omit":
        del merged[ACTION_TIMING_KEY]
    buffer = EpisodeBuffer(
        fps=30.0,
        task="sim reach" if task_index == 1 else "sim place",
        task_index=task_index,
        episode_index=episode_index,
        metadata=merged,
    )
    for index in range(91):
        action = _axis(action_offset + index / 30.0)
        buffer.append(
            SensorimotorFrame(
                timestamp_s=index / 30.0,
                images={
                    "left_wrist": np.full((8, 8, 3), index, dtype=np.uint8),
                    "right_wrist": np.full((8, 8, 3), index + 1, dtype=np.uint8),
                },
                joint_position=action + 0.5,
                executed_action=action,
            )
        )
    buffer.save(root, stem=f"episode_{episode_index:06d}")


def _axis(base: float) -> np.ndarray:
    return np.array([base + axis for axis in range(ACTION_DIM)], dtype=np.float32)


def _assert_np_state(actual: object, expected: object) -> None:
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]
