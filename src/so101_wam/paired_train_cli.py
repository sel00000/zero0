"""CLI for reviewed human-video paired candidate training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .checkpoint import CheckpointError
from .model import ModelContractError
from .paired_data import PairedDataError, load_human_robot_pairs
from .paired_training import (
    PairedTrainingError,
    PairedWindowMode,
    paired_training_artifacts_as_json,
    train_paired_candidate,
)
from .training import (
    CompactWAMTrainingConfig,
    SamplingStrategy,
    TrainingError,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a task-disjoint offline candidate from reviewed human-video "
            "pairs. The result cannot enable real robot output."
        )
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--policy-hz", type=float, default=10.0)
    parser.add_argument("--servo-hz", type=float, default=50.0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--action-history-steps", type=int, default=4)
    parser.add_argument("--max-context-steps", type=int, default=300)
    parser.add_argument(
        "--window-mode",
        choices=tuple(item.value for item in PairedWindowMode),
        default=PairedWindowMode.EARLIEST.value,
    )
    parser.add_argument("--future-latent-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    report_path = args.report or args.output.with_suffix(".training.json")
    try:
        config = CompactWAMTrainingConfig(
            policy_hz=args.policy_hz,
            servo_hz=args.servo_hz,
            latent_dim=args.latent_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            future_steps=args.future_steps,
            action_horizon=args.action_horizon,
            action_history_steps=args.action_history_steps,
            ifp_steps=0,
            max_context_steps=args.max_context_steps,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            sampling_strategy=SamplingStrategy.TASK_BALANCED,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            future_latent_weight=args.future_latent_weight,
            action_weight=args.action_weight,
            ifp_weight=0.0,
            seed=args.seed,
        )
        artifacts = train_paired_candidate(
            load_human_robot_pairs(args.train_manifest),
            load_human_robot_pairs(args.validation_manifest),
            checkpoint_path=args.output,
            report_path=report_path,
            checkpoint_id=args.checkpoint_id,
            config=config,
            device=args.device,
            window_mode=PairedWindowMode(args.window_mode),
        )
    except (
        CheckpointError,
        ModelContractError,
        PairedDataError,
        PairedTrainingError,
        TrainingError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            dict(paired_training_artifacts_as_json(artifacts)),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
