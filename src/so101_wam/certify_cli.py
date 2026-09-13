"""CLI for evidence-bound, immutable G10 deployment artifact issuance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .checkpoint import CheckpointError
from .config import ConfigError, ProjectConfig
from .dataset import DatasetError
from .deployment import DeploymentCertificationError
from .deployment_issuer import (
    DeploymentIssuanceError,
    issue_deployment_artifacts,
    issued_artifacts_as_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact G6-G9 evidence and immutably issue a G10 dry-rollout "
            "checkpoint/certification pair. No evidence values can be supplied "
            "directly; every authorization field is derived from source artifacts."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--mujoco-config", required=True, type=Path)
    parser.add_argument("--mujoco-report", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="prompt manifest; defaults to the prompt path with a .json suffix",
    )
    parser.add_argument("--manual-signoff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--certification", required=True, type=Path)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=1,
        help="maximum policy steps authorized for the G10 dry rollout",
    )
    args = parser.parse_args(argv)

    try:
        artifacts = issue_deployment_artifacts(
            ProjectConfig.load(args.config),
            candidate_checkpoint_path=args.candidate,
            training_report_path=args.training_report,
            preflight_report_path=args.preflight_report,
            mujoco_config_path=args.mujoco_config,
            mujoco_report_path=args.mujoco_report,
            prompt_npz_path=args.prompt,
            prompt_manifest_path=args.manifest,
            manual_signoff_path=args.manual_signoff,
            output_checkpoint_path=args.output,
            output_certification_path=args.certification,
            checkpoint_id=args.checkpoint_id,
            max_policy_steps=args.max_policy_steps,
        )
    except (
        CheckpointError,
        ConfigError,
        DatasetError,
        DeploymentCertificationError,
        DeploymentIssuanceError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            dict(issued_artifacts_as_json(artifacts)),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
