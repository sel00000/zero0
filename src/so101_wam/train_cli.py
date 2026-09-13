"""CLI for task-disjoint offline CompactWAM candidate training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .checkpoint import CheckpointError
from .model import ActionDecoder, ModelContractError
from .training import (
    CompactWAMTrainingConfig,
    IFPArchitecture,
    SamplingStrategy,
    TrainingError,
    train_offline_candidate,
    training_artifacts_as_json,
)
from .training_data import TrainingDataError, load_episode_records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a task-disjoint offline CompactWAM candidate. The resulting "
            "checkpoint stores trained=false and cannot enable real output."
        )
    )
    parser.add_argument("--train-episodes", nargs="+", required=True, type=Path)
    parser.add_argument("--validation-episodes", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=100)
    parser.add_argument(
        "--sampling-strategy",
        choices=tuple(item.value for item in SamplingStrategy),
        default=SamplingStrategy.TASK_BALANCED.value,
    )
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
    parser.add_argument(
        "--action-decoder",
        choices=tuple(item.value for item in ActionDecoder),
        default=ActionDecoder.LEGACY_MEAN.value,
    )
    parser.add_argument("--ifp-steps", type=int, default=2)
    parser.add_argument("--ifp-stride", type=int, default=2)
    parser.add_argument(
        "--ifp-architecture",
        choices=tuple(item.value for item in IFPArchitecture),
        default=IFPArchitecture.COMPACT_LINEAR.value,
    )
    parser.add_argument("--ifp-window-steps", type=int)
    parser.add_argument("--max-context-steps", type=int, default=300)
    parser.add_argument("--future-latent-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--ifp-weight", type=float, default=0.25)
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
            action_decoder=ActionDecoder(args.action_decoder),
            ifp_steps=args.ifp_steps,
            ifp_stride=args.ifp_stride,
            ifp_architecture=IFPArchitecture(args.ifp_architecture),
            ifp_window_steps=args.ifp_window_steps,
            max_context_steps=args.max_context_steps,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            sampling_strategy=SamplingStrategy(args.sampling_strategy),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            future_latent_weight=args.future_latent_weight,
            action_weight=args.action_weight,
            ifp_weight=args.ifp_weight,
            seed=args.seed,
        )
        artifacts = train_offline_candidate(
            load_episode_records(args.train_episodes),
            load_episode_records(args.validation_episodes),
            checkpoint_path=args.output,
            report_path=report_path,
            checkpoint_id=args.checkpoint_id,
            config=config,
            device=args.device,
        )
    except (
        CheckpointError,
        ModelContractError,
        TrainingDataError,
        TrainingError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            dict(training_artifacts_as_json(artifacts)),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
