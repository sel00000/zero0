from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_wam.checkpoint import load_compact_wam_bundle
from so101_wam.constants import ACTION_DIM
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer, save_episode
from so101_wam.deployment import canonical_json_sha256, file_sha256
from so101_wam.model import ActionDecoder, CompactWAM
from so101_wam.paired_data import (
    HUMAN_TASK_SPEC_FORMAT,
    LEROBOT_V3_COMPATIBLE,
    HumanRobotPairRecord,
    HumanTaskSpecArtifact,
    SemanticMatchStatus,
    paired_manifest_as_json,
    validate_pair_disjoint_split,
)
from so101_wam.paired_training import (
    PairedWindowMode,
    PairedTrainingError,
    paired_training_batch,
)
from so101_wam import paired_training
from so101_wam.training import (
    TRAINING_REPORT_SCHEMA,
    CompactWAMTrainingConfig,
    TrainingArtifactExistsError,
)
from so101_wam.training_data import load_episode_records
from so101_wam.training_data import CompactWAMTrainingBatch


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _pair(
    tmp_path: Path,
    *,
    pair_id: str = "pair-0001",
    task: str = "place block",
    task_index: int = 9,
    episode_index: int = 3,
    value_offset: int = 0,
) -> HumanRobotPairRecord:
    episode = EpisodeBuffer(
        fps=30.0,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
        metadata={"action_source": "imported_lerobot_v3_action"},
    )
    for frame_index in range(91):
        axes = np.array(
            [frame_index + axis + value_offset for axis in range(ACTION_DIM)],
            dtype=np.float32,
        )
        episode.append(
            SensorimotorFrame(
                timestamp_s=frame_index / 30.0,
                images={
                    "left_wrist": np.full(
                        (8, 8, 3), frame_index + value_offset, dtype=np.uint8
                    ),
                    "right_wrist": np.full(
                        (8, 8, 3),
                        frame_index + value_offset + 1,
                        dtype=np.uint8,
                    ),
                },
                joint_position=axes,
                executed_action=axes,
            )
        )
    robot_path, _ = episode.save(
        tmp_path / "episodes",
        stem=f"episode_{episode_index:06d}",
    )
    robot = load_episode_records((robot_path,))[0]

    human_video_path = tmp_path / "human" / f"{pair_id}.mp4"
    human_video_path.parent.mkdir(parents=True)
    human_video_path.write_bytes(f"source-video:{pair_id}".encode("utf-8"))
    task_spec_path = tmp_path / "human" / f"{pair_id}.task_spec.npz"
    np.savez_compressed(
        task_spec_path,
        timestamp=np.linspace(0.0, 3.0, 31, dtype=np.float64),
        rgb=np.stack(
            [
                np.full(
                    (8, 8, 3),
                    200 - frame_index + value_offset,
                    dtype=np.uint8,
                )
                for frame_index in range(31)
            ]
        ),
    )

    return HumanRobotPairRecord(
        pair_id=pair_id,
        task=task,
        task_index=task_index,
        semantic_match=SemanticMatchStatus.HUMAN_REVIEWED,
        source_format=LEROBOT_V3_COMPATIBLE,
        source_url="hf://datasets/example/repo",
        human_video_path=human_video_path,
        human_video_sha256=_file_sha256(human_video_path),
        human_view="third_person",
        human_task_spec=HumanTaskSpecArtifact(
            path=task_spec_path,
            sha256=_file_sha256(task_spec_path),
            format=HUMAN_TASK_SPEC_FORMAT,
        ),
        robot_episode=robot,
        robot_episode_sha256=_file_sha256(robot_path),
        provenance={
            "source_dataset": "example/repo",
            "source_layout": "data/,videos/",
            "license": "apache-2.0",
            "transformation": "human RGB extraction and local robot conversion",
        },
    )


def _batch(pair: HumanRobotPairRecord):
    return paired_training_batch(
        pair,
        neutral_axes=np.zeros(ACTION_DIM, dtype=np.float32),
        policy_hz=10.0,
        servo_hz=50.0,
        action_history_steps=4,
        future_steps=1,
        action_horizon=2,
    )


def test_paired_prior_command(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    data = replace(
        pair.robot_episode.data,
        metadata={
            "action_timing": "observation_then_command",
            "initial_previous_action": [-7.0] * ACTION_DIM,
        },
    )
    path, _ = save_episode(data, tmp_path / "timed")
    pair = replace(
        pair,
        robot_episode=load_episode_records((path,))[0],
        robot_episode_sha256=_file_sha256(path),
    )
    batch = _batch(pair)
    np.testing.assert_array_equal(batch.live_actions[0, :, 0], [-7.0, 2.0, 5.0, 8.0])
    assert batch.target_actions[0, 0, 0] == 9.0


def _training_config(*, seed: int = 7) -> CompactWAMTrainingConfig:
    return CompactWAMTrainingConfig(
        latent_dim=8,
        transformer_layers=1,
        transformer_heads=2,
        future_steps=1,
        action_horizon=2,
        action_history_steps=4,
        ifp_steps=0,
        stage1_steps=1,
        stage2_steps=2,
        seed=seed,
    )


def _anchor_action(batch: CompactWAMTrainingBatch) -> float:
    return float(batch.target_actions[0, 0, 0])


class _PromptSumModel:
    def __init__(self) -> None:
        self.axis_mean = torch.zeros(ACTION_DIM)
        self.axis_scale = torch.ones(ACTION_DIM)
        self.calls: list[dict[str, torch.Tensor]] = []

    def eval(self) -> None:
        return None

    def encoder(self, images: torch.Tensor, *, segment_id: int) -> torch.Tensor:
        del segment_id
        return torch.zeros(images.shape[0], images.shape[1], 3)

    def __call__(
        self,
        *,
        prompt_images: torch.Tensor,
        prompt_proprio: torch.Tensor,
        prompt_actions: torch.Tensor,
        live_images: torch.Tensor,
        live_proprio: torch.Tensor,
        live_actions: torch.Tensor,
        prompt_mask: torch.Tensor,
        compute_ifp: bool,
    ) -> dict[str, torch.Tensor]:
        del prompt_proprio, prompt_actions, compute_ifp
        self.calls.append(
            {
                "live_images": live_images.clone(),
                "live_proprio": live_proprio.clone(),
                "live_actions": live_actions.clone(),
                "prompt_mask": prompt_mask.clone(),
            }
        )
        value = prompt_images.flatten(start_dim=1).sum(dim=1).view(-1, 1, 1)
        return {
            "actions": value.expand(-1, 2, ACTION_DIM).clone(),
            "future_latents": value.expand(-1, 1, 3).clone(),
        }


def _eval_batch(
    *,
    pair_id: str,
    task_index: int,
    prompt_steps: int,
    prompt_value: float,
    image_size: int = 8,
) -> paired_training._PairedBatch:
    batch = CompactWAMTrainingBatch(
        prompt_images=torch.full(
            (1, prompt_steps, 2, 3, image_size, image_size),
            prompt_value,
        ),
        prompt_proprio=torch.zeros(1, prompt_steps, ACTION_DIM),
        prompt_actions=torch.zeros(1, prompt_steps, ACTION_DIM),
        live_images=torch.full(
            (1, 4, 2, 3, image_size, image_size),
            float(task_index),
        ),
        live_proprio=torch.full((1, 4, ACTION_DIM), float(task_index)),
        live_actions=torch.full((1, 4, ACTION_DIM), float(task_index)),
        prompt_mask=torch.ones(1, prompt_steps, dtype=torch.bool),
        target_future_images=torch.zeros(1, 1, 2, 3, image_size, image_size),
        target_actions=torch.zeros(1, 2, ACTION_DIM),
        target_ifp_images=None,
    )
    pair = SimpleNamespace(
        pair_id=pair_id,
        fingerprint=f"fingerprint:{pair_id}",
        task=f"task:{task_index}",
        task_index=task_index,
    )
    return paired_training._PairedBatch(pair=pair, batch=batch)


def _multi_pair_split(
    tmp_path: Path,
) -> tuple[tuple[HumanRobotPairRecord, ...], tuple[HumanRobotPairRecord, ...]]:
    train = (
        _pair(
            tmp_path / "train-a",
            pair_id="train-pair-a",
            episode_index=1,
        ),
        _pair(
            tmp_path / "train-b",
            pair_id="train-pair-b",
            task="press button",
            task_index=12,
            episode_index=2,
            value_offset=1,
        ),
    )
    validation = (
        _pair(
            tmp_path / "validation-a",
            pair_id="validation-pair-a",
            task="stack cup",
            task_index=10,
            episode_index=3,
            value_offset=2,
        ),
        _pair(
            tmp_path / "validation-b",
            pair_id="validation-pair-b",
            task="open drawer",
            task_index=11,
            episode_index=4,
            value_offset=3,
        ),
    )
    return train, validation


def _write_manifest(
    path: Path,
    pairs: tuple[HumanRobotPairRecord, ...],
    *,
    root: Path,
) -> None:
    path.write_text(
        json.dumps(paired_manifest_as_json(pairs, root=root)),
        encoding="utf-8",
    )


def test_paired_batch_backpropagates_from_action_free_prompt(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path)
    batch = _batch(pair)

    assert batch.prompt_images.shape == (1, 31, 2, 3, 8, 8)
    assert torch.equal(batch.prompt_images[:, :, 0], batch.prompt_images[:, :, 1])
    assert int(batch.prompt_images[0, 0, 0, 0, 0, 0]) == 200
    assert int(batch.live_images[0, 0, 0, 0, 0, 0]) == 0
    assert torch.count_nonzero(batch.prompt_proprio) == 0
    assert torch.count_nonzero(batch.prompt_actions) == 0

    torch.manual_seed(7)
    model = CompactWAM(
        latent_dim=8,
        transformer_heads=2,
        future_steps=1,
        action_horizon=2,
        action_history_steps=4,
        ifp_steps=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    previous_position_weights = (
        model.temporal_position_embedding.weight.detach().clone()
    )
    prompt_images = batch.prompt_images.float().div(255.0).requires_grad_()
    outputs = model(
        **{
            **batch.model_kwargs(),
            "prompt_images": prompt_images,
        },
        compute_ifp=False,
    )
    with torch.no_grad():
        target_future = model.encoder(batch.target_future_images, segment_id=1)
    losses = model.loss(
        outputs,
        target_future_latents=target_future,
        target_actions=batch.target_actions,
        target_ifp_latents=None,
    )
    losses["total"].backward()

    assert prompt_images.grad is not None
    assert torch.isfinite(prompt_images.grad).all()
    assert torch.count_nonzero(prompt_images.grad) > 0
    optimizer.step()
    assert not torch.equal(
        previous_position_weights,
        model.temporal_position_embedding.weight,
    )


def test_pair_windows_default(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    config = _training_config()

    batches = paired_training._paired_batches(
        (pair,),
        neutral_axes=np.zeros(ACTION_DIM, dtype=np.float32),
        config=config,
    )

    assert len(batches) == 1
    assert batches[0].anchor_policy_position == 3
    assert _anchor_action(batches[0].batch) == _anchor_action(_batch(pair))
    assert paired_training._schedule_identity(
        batches[0],
        window_mode=PairedWindowMode.EARLIEST,
    ) == pair.fingerprint


def test_pair_windows_all(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    config = _training_config()

    batches = paired_training._paired_batches(
        (pair,),
        neutral_axes=np.zeros(ACTION_DIM, dtype=np.float32),
        config=config,
        window_mode=PairedWindowMode.ALL_COMPLETE,
    )

    assert len(batches) == 27
    assert [item.anchor_policy_position for item in batches[:3]] == [3, 4, 5]
    assert batches[-1].anchor_policy_position == 29
    assert [_anchor_action(item.batch) for item in batches[:3]] == [
        pytest.approx(9.0),
        pytest.approx(12.0),
        pytest.approx(15.0),
    ]


def test_pair_windows_horizon(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    config = replace(_training_config(), action_horizon=21)

    batches = paired_training._paired_batches(
        (pair,),
        neutral_axes=np.zeros(ACTION_DIM, dtype=np.float32),
        config=config,
        window_mode=PairedWindowMode.ALL_COMPLETE,
    )

    assert batches[-1].anchor_policy_position == 26


def test_pair_window_mode_enum(tmp_path: Path) -> None:
    pair = _pair(tmp_path)

    with pytest.raises(PairedTrainingError, match="PairedWindowMode"):
        paired_training._paired_batches(
            (pair,),
            neutral_axes=np.zeros(ACTION_DIM, dtype=np.float32),
            config=_training_config(),
            window_mode="all_complete",
        )


def test_pair_window_schedule_fair() -> None:
    short = _eval_batch(
        pair_id="short",
        task_index=1,
        prompt_steps=3,
        prompt_value=1.0,
    )
    long = tuple(
        paired_training._PairedBatch(
            pair=_eval_batch(
                pair_id="long",
                task_index=1,
                prompt_steps=3,
                prompt_value=1.0,
            ).pair,
            batch=short.batch,
            anchor_policy_position=anchor,
        )
        for anchor in range(8)
    )
    batches = (
        paired_training._PairedBatch(
            pair=short.pair,
            batch=short.batch,
            anchor_policy_position=0,
        ),
        *long,
    )

    schedule = paired_training._draw_pair_schedule(
        batches,
        steps=16,
        rng=np.random.default_rng(7),
        window_mode=PairedWindowMode.ALL_COMPLETE,
    )

    counts: dict[str, int] = {}
    for item in schedule:
        counts[item.pair.pair_id] = counts.get(item.pair.pair_id, 0) + 1
    assert counts == {"long": 8, "short": 8}


@pytest.mark.parametrize("anchor", [-1, 2, 30, True, 3.5])
def test_pair_anchor_rejected(tmp_path: Path, anchor: object) -> None:
    pair = _pair(tmp_path)
    config = _training_config()

    with pytest.raises(PairedTrainingError):
        paired_training_batch(
            pair,
            neutral_axes=np.zeros(ACTION_DIM, dtype=np.float32),
            policy_hz=config.policy_hz,
            servo_hz=config.servo_hz,
            action_history_steps=config.action_history_steps,
            future_steps=config.future_steps,
            action_horizon=config.action_horizon,
            anchor_policy_position=anchor,
        )


def test_window_audit_identity() -> None:
    first = _eval_batch(
        pair_id="first", task_index=1, prompt_steps=3, prompt_value=1.0
    )
    later = replace(first, anchor_policy_position=1, anchor_time_s=0.1)
    mode = PairedWindowMode.ALL_COMPLETE

    first_id = paired_training._schedule_identity(first, window_mode=mode)
    later_id = paired_training._schedule_identity(later, window_mode=mode)
    assert canonical_json_sha256(first_id) != canonical_json_sha256(later_id)
    assert len(paired_training._window_draw_counts((first, later))) == 2
    assert paired_training._window_inventory_sha256(
        (first,), window_mode=mode
    ) != paired_training._window_inventory_sha256(
        (first,), window_mode=PairedWindowMode.EARLIEST
    )


def test_window_task_balance() -> None:
    first = _eval_batch(
        pair_id="first", task_index=1, prompt_steps=3, prompt_value=1.0
    )
    second = _eval_batch(
        pair_id="second", task_index=2, prompt_steps=3, prompt_value=1.0
    )
    batches = (
        first,
        *(replace(second, anchor_policy_position=i) for i in range(8)),
    )
    schedules = tuple(
        paired_training._draw_pair_schedule(
            batches,
            steps=32,
            rng=np.random.default_rng(7),
            window_mode=PairedWindowMode.ALL_COMPLETE,
        )
        for _ in range(2)
    )
    identities = [
        [paired_training._window_identity(item) for item in schedule]
        for schedule in schedules
    ]
    assert identities[0] == identities[1]
    assert sum(item.pair.task_index == 1 for item in schedules[0]) == 16
    assert {item.anchor_policy_position for item in schedules[0] if item.pair.task_index == 2} == set(range(8))


def test_paired_training_requires_reviewed_checksum_bound_prompt(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path)

    with pytest.raises(PairedTrainingError, match="human_reviewed"):
        _batch(
            replace(
                pair,
                semantic_match=SemanticMatchStatus.UNVERIFIED,
            )
        )

    with pytest.raises(PairedTrainingError, match="task-spec"):
        _batch(replace(pair, human_task_spec=None))


@pytest.mark.parametrize("artifact", ("human video", "robot episode"))
def test_paired_training_rechecks_source_checksum(
    tmp_path: Path,
    artifact: str,
) -> None:
    pair = _pair(tmp_path)
    path = (
        pair.human_video_path
        if artifact == "human video"
        else pair.robot_episode.path
    )
    path.write_bytes(b"changed-after-pair-import")

    with pytest.raises(
        PairedTrainingError,
        match=rf"{artifact} checksum changed",
    ):
        _batch(pair)


@pytest.mark.parametrize(
    "mode",
    [
        ActionDecoder.LEGACY_MEAN,
        ActionDecoder.ORDERED_CONCAT,
    ],
)
def test_paired_training_publishes_multi_pair_candidate_with_false_claims(
    tmp_path: Path,
    mode: ActionDecoder,
) -> None:
    from so101_wam.paired_training import train_paired_candidate

    train, validation = _multi_pair_split(tmp_path)
    checkpoint_path = tmp_path / "paired-candidate.pt"
    report_path = tmp_path / "paired-candidate.training.json"
    split = validate_pair_disjoint_split(train, validation)

    config = _training_config()
    if mode is not ActionDecoder.LEGACY_MEAN:
        config = replace(config, action_decoder=mode)

    artifacts = train_paired_candidate(
        train,
        validation,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        checkpoint_id="paired-candidate-001",
        config=config,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_core = {
        key: value
        for key, value in report.items()
        if key not in {"training_evidence_sha256", "artifacts"}
    }
    bundle = load_compact_wam_bundle(checkpoint_path)

    assert artifacts.train_pairs == 2
    assert artifacts.validation_pairs == 2
    assert artifacts.optimizer_steps == 3
    assert report["schema_version"] == TRAINING_REPORT_SCHEMA
    assert report["artifact_kind"] == "compact_wam_candidate"
    assert report["result"] == "pass"
    assert report["trained"] is False
    assert report["offline_trained"] is True
    assert report["deployment_ready"] is False
    assert report["real_world_success_claimed"] is False
    assert report["official_zero_wam_claimed"] is False
    assert report["semantic_heldout_success_claimed"] is False
    assert report["prompt_causality_claimed"] is False
    assert report["protocol"]["real_output_authorized"] is False
    assert report["protocol"]["prompt_robot_signals"] == "neutral_train_mean"
    assert report["data"]["train_pair_count"] == 2
    assert report["data"]["validation_pair_count"] == 2
    assert sorted(report["data"]["sampling_audit"]["task_draw_counts"].values()) == [
        1,
        2,
    ]
    assert report["data"]["train_pair_digest"] == split.train_digest
    assert report["data"]["validation_pair_digest"] == split.validation_digest
    assert report["data"]["validation_tasks"] == [
        {"task_index": 10, "task": "stack cup"},
        {"task_index": 11, "task": "open drawer"},
    ]
    assert {
        item["pair_fingerprint"] for item in report["data"]["validation_pairs"]
    } == {pair.fingerprint for pair in validation}
    assert {
        item["human_task_spec_sha256"] for item in report["data"]["validation_pairs"]
    } == {pair.human_task_spec.sha256 for pair in validation if pair.human_task_spec}
    assert report["validation"]["pair_digest"] == split.validation_digest
    assert report["validation"]["evaluation_scope"] == (
        "task_disjoint_human_video_offline_proxy"
    )
    assert report["validation"]["prompt_causality_claimed"] is False
    assert report["validation"]["semantic_task_success_evaluated"] is False
    assert report["validation"]["mismatched_prompt_action_mean_abs_delta"] > 0.0
    assert report["validation"][
        "length_controlled_mismatched_prompt_action_mean_abs_delta"
    ] >= 0.0
    assert report["validation"]["mismatched_prompt_length_control"] == (
        paired_training._PROMPT_LEN_CONTROL
    )
    assert {
        item["prompt_length_control"]
        for item in report["validation"]["mismatched_pairing"]
    } == {paired_training._PROMPT_LEN_CONTROL}
    assert report["validation"]["null_prompt_action_mean_abs_delta"] > 0.0
    assert report["training_evidence_sha256"] == canonical_json_sha256(report_core)
    assert report["artifacts"]["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert report["model"]["action_decoder"] == mode.value

    assert bundle.metadata["offline_trained"] is True
    assert bundle.metadata["trained"] is False
    assert bundle.metadata["deployment_ready"] is False
    assert bundle.metadata["paired_human_video"] is True
    assert (
        bundle.metadata["training_evidence_sha256"]
        == report["training_evidence_sha256"]
    )
    assert bundle.metadata["action_decoder"] == mode.value


def test_paired_training_rejects_unreviewed_pair_before_outputs(
    tmp_path: Path,
) -> None:
    from so101_wam.paired_training import train_paired_candidate

    train, validation = _multi_pair_split(tmp_path)
    checkpoint_path = tmp_path / "paired-candidate.pt"
    report_path = tmp_path / "paired-candidate.training.json"
    validation = (
        replace(
            validation[0],
            semantic_match=SemanticMatchStatus.UNVERIFIED,
        ),
        validation[1],
    )

    with pytest.raises(PairedTrainingError, match="human_reviewed"):
        train_paired_candidate(
            train,
            validation,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            checkpoint_id="paired-candidate-001",
            config=_training_config(),
        )

    assert not checkpoint_path.exists()
    assert not report_path.exists()


def test_paired_all_report(tmp_path: Path) -> None:
    from so101_wam.paired_training import train_paired_candidate

    train, validation = _multi_pair_split(tmp_path)
    checkpoint_path = tmp_path / "paired-all.pt"
    report_path = tmp_path / "paired-all.training.json"

    train_paired_candidate(
        train,
        validation,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        checkpoint_id="paired-all-001",
        config=replace(_training_config(), stage1_steps=1, stage2_steps=1),
        window_mode=PairedWindowMode.ALL_COMPLETE,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = load_compact_wam_bundle(checkpoint_path)
    audit = report["data"]["sampling_audit"]

    assert report["protocol"]["paired_window_mode"] == "all_complete"
    assert report["protocol"]["paired_batch_scope"] == (
        "all_complete_anchors_per_pair"
    )
    assert report["data"]["train_window_count"] == 54
    assert report["data"]["validation_window_count"] == 54
    assert len(audit["window_draw_counts"]) == 2
    assert all(
        key.rsplit(":", maxsplit=1)[1].isdigit()
        for key in audit["window_draw_counts"]
    )
    assert report["validation"]["evaluation_window_weighting"] == (
        "one_complete_window_one_vote"
    )
    assert {
        "validation_anchor_policy_position",
        "validation_anchor_time_s",
        "mismatched_anchor_policy_position",
        "mismatched_anchor_time_s",
    } <= set(report["validation"]["mismatched_pairing"][0])
    assert bundle.metadata["paired_window_mode"] == "all_complete"
    assert (
        bundle.metadata["sampling_schedule_sha256"]
        == report["training_schedule_sha256"]
    )


def test_paired_training_requires_multi_pair_heldout_controls(
    tmp_path: Path,
) -> None:
    from so101_wam.paired_training import train_paired_candidate

    train, validation = _multi_pair_split(tmp_path)

    with pytest.raises(PairedTrainingError, match="at least two train pairs"):
        train_paired_candidate(
            train[:1],
            validation,
            checkpoint_path=tmp_path / "train-short.pt",
            report_path=tmp_path / "train-short.json",
            checkpoint_id="train-short",
            config=_training_config(),
        )

    with pytest.raises(PairedTrainingError, match="validation tasks"):
        train_paired_candidate(
            train,
            validation[:1],
            checkpoint_path=tmp_path / "validation-short.pt",
            report_path=tmp_path / "validation-short.json",
            checkpoint_id="validation-short",
            config=_training_config(),
        )


def test_paired_training_no_overwrite_precedes_batch_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_wam import paired_training

    train, validation = _multi_pair_split(tmp_path)
    checkpoint_path = tmp_path / "paired-candidate.pt"
    report_path = tmp_path / "paired-candidate.training.json"
    checkpoint_path.write_bytes(b"keep")

    def unexpected_batches(*args: object, **kwargs: object) -> object:
        raise AssertionError("batch materialization must not start")

    monkeypatch.setattr(paired_training, "_paired_batches", unexpected_batches)

    with pytest.raises(TrainingArtifactExistsError, match="immutable"):
        paired_training.train_paired_candidate(
            train,
            validation,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            checkpoint_id="paired-candidate-001",
            config=_training_config(),
        )

    assert checkpoint_path.read_bytes() == b"keep"
    assert not report_path.exists()


@pytest.mark.parametrize("mode", [None, "all_complete"])
def test_paired_cli_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str | None,
) -> None:
    from so101_wam.paired_train_cli import main

    train, validation = _multi_pair_split(tmp_path)
    train_manifest = tmp_path / "train-pairs.json"
    validation_manifest = tmp_path / "validation-pairs.json"
    output = tmp_path / "paired-cli.pt"
    report = tmp_path / "paired-cli.training.json"
    _write_manifest(train_manifest, train, root=tmp_path)
    _write_manifest(validation_manifest, validation, root=tmp_path)

    assert (
        main(
            [
                "--train-manifest",
                str(train_manifest),
                "--validation-manifest",
                str(validation_manifest),
                "--output",
                str(output),
                "--report",
                str(report),
                "--checkpoint-id",
                "paired-cli-001",
                "--latent-dim",
                "8",
                "--transformer-heads",
                "2",
                "--future-steps",
                "1",
                "--action-horizon",
                "2",
                "--stage1-steps",
                "1",
                "--stage2-steps",
                "2",
                *([] if mode is None else ["--window-mode", mode]),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert result["checkpoint_id"] == "paired-cli-001"
    assert result["optimizer_steps"] == 3
    assert result["trained"] is False
    assert result["deployment_ready"] is False
    assert output.is_file()
    assert evidence["offline_trained"] is True
    assert evidence["prompt_causality_claimed"] is False
    assert evidence["protocol"]["paired_window_mode"] == (mode or "earliest")


def test_paired_eval_len_ctrl() -> None:
    model = _PromptSumModel()
    short = _eval_batch(
        pair_id="short",
        task_index=1,
        prompt_steps=3,
        prompt_value=1.0,
        image_size=8,
    )
    long = _eval_batch(
        pair_id="long",
        task_index=2,
        prompt_steps=5,
        prompt_value=1.0,
        image_size=8,
    )

    metrics, pairs = paired_training._evaluate_paired(
        model,
        (short, long),
        device=torch.device("cpu"),
    )

    assert metrics["mismatched_prompt_action_mean_abs_delta"] > 0.0
    assert (
        metrics["length_controlled_mismatched_prompt_action_mean_abs_delta"]
        == 0.0
    )
    assert (
        metrics["length_controlled_mismatched_prompt_future_mean_abs_delta"]
        == 0.0
    )
    assert pairs[0]["prompt_length_control"] == (
        "nearest_uniform_index_resample_to_matched_prompt_length"
    )
    assert pairs[0]["matched_prompt_steps"] == 3
    assert pairs[0]["mismatched_prompt_steps"] == 5
    assert pairs[0]["controlled_mismatched_prompt_steps"] == 3
    assert torch.equal(model.calls[3]["live_images"], model.calls[0]["live_images"])
    assert torch.equal(model.calls[3]["live_proprio"], model.calls[0]["live_proprio"])
    assert torch.equal(model.calls[3]["live_actions"], model.calls[0]["live_actions"])
    assert torch.equal(model.calls[3]["prompt_mask"], model.calls[0]["prompt_mask"])


def test_paired_eval_ctrl_sens() -> None:
    model = _PromptSumModel()
    short = _eval_batch(
        pair_id="short",
        task_index=1,
        prompt_steps=3,
        prompt_value=1.0,
    )
    long = _eval_batch(
        pair_id="long",
        task_index=2,
        prompt_steps=5,
        prompt_value=2.0,
    )

    metrics, _ = paired_training._evaluate_paired(
        model,
        (short, long),
        device=torch.device("cpu"),
    )

    assert metrics["length_controlled_mismatched_prompt_action_mean_abs_delta"] > 0.0
    assert metrics["length_controlled_mismatched_prompt_future_mean_abs_delta"] > 0.0


def test_paired_eval_ctrl_equal() -> None:
    model = _PromptSumModel()
    first = _eval_batch(
        pair_id="first",
        task_index=1,
        prompt_steps=3,
        prompt_value=1.0,
    )
    second = _eval_batch(
        pair_id="second",
        task_index=2,
        prompt_steps=3,
        prompt_value=1.0,
    )
    second_mask = torch.tensor([[True, True, False]])
    second = paired_training._PairedBatch(
        pair=second.pair,
        batch=replace(second.batch, prompt_mask=second_mask),
    )

    metrics, pairs = paired_training._evaluate_paired(
        model,
        (first, second),
        device=torch.device("cpu"),
    )

    assert metrics["mismatched_prompt_action_mean_abs_delta"] == 0.0
    assert (
        metrics["length_controlled_mismatched_prompt_action_mean_abs_delta"]
        == metrics["mismatched_prompt_action_mean_abs_delta"]
    )
    assert pairs[0]["matched_prompt_steps"] == 3
    assert pairs[0]["mismatched_prompt_steps"] == 3
    assert pairs[0]["controlled_mismatched_prompt_steps"] == 3
    assert torch.equal(model.calls[3]["prompt_mask"], model.calls[0]["prompt_mask"])
    assert not torch.equal(model.calls[3]["prompt_mask"], second_mask)


def test_resample_prompt_indices() -> None:
    values = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    image_like = values.view(1, 5, 1, 1, 1, 1)
    prop_like = values.expand(1, 5, ACTION_DIM)
    action_like = (values + 10).expand(1, 5, ACTION_DIM)

    image = paired_training._resample_prompt(image_like, target_steps=3)
    prop = paired_training._resample_prompt(prop_like, target_steps=3)
    action = paired_training._resample_prompt(action_like, target_steps=3)

    assert image.flatten().tolist() == [0.0, 2.0, 4.0]
    assert prop[0, :, 0].tolist() == [0.0, 2.0, 4.0]
    assert action[0, :, 0].tolist() == [10.0, 12.0, 14.0]
    assert paired_training._resample_prompt(prop_like, target_steps=5) is prop_like


def test_compact_len_ctrl_equal() -> None:
    torch.manual_seed(7)
    model = CompactWAM(
        latent_dim=8,
        transformer_heads=2,
        future_steps=1,
        action_horizon=2,
        action_history_steps=4,
        ifp_steps=0,
    )
    short = _eval_batch(
        pair_id="short",
        task_index=1,
        prompt_steps=3,
        prompt_value=1.0,
    )
    long = _eval_batch(
        pair_id="long",
        task_index=2,
        prompt_steps=5,
        prompt_value=1.0,
    )

    metrics, _ = paired_training._evaluate_paired(
        model,
        (short, long),
        device=torch.device("cpu"),
    )

    assert metrics["mismatched_prompt_action_mean_abs_delta"] > 0.0
    assert (
        metrics["length_controlled_mismatched_prompt_action_mean_abs_delta"]
        == 0.0
    )
