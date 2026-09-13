"""Frozen, seen-task simulator learning diagnostic with a reference/control gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DEFAULT_ROBOT_FREE_CONFIG_PATH, ProjectConfig
from .deployment import canonical_json_sha256, file_sha256, project_config_sha256
from .mujoco_semantic_benchmark import SemanticTask
from .model import ActionRangeConstraint
from .robot_free_cli import _write_json_no_overwrite
from .seen_task_training import _action_output_contract, train_seen_task_candidate
from .semantic_pilot import _fixed_runtime, _runtime, _source_hashes
from .sim_reference import SimTrialConfig, TrialKind, run_sim_trial
from .training import CompactWAMTrainingConfig
from .training_data import load_episode_records


STUDY_SCHEMA = "so101_wam.reference_learning.v2"
REPORT_NAME = "reference-learning-report.json"
MODEL_SEED = 7
INITIAL_CONDITIONS = (7, 13)
POLICY_STEPS = 56
POLICY_HZ = 10.0
SERVO_HZ = 50.0
ACTION_HORIZON = 10
RAMP_TICKS = 120
REFERENCE_TARGET = (9.3000001907, 66.0, -62.6666679382, -31.3333339691, 150.0, 20.0)


class ReferenceLearningError(ValueError):
    """The bounded diagnostic cannot preserve its evidence contract."""


def run_reference_learning(
    output_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_ROBOT_FREE_CONFIG_PATH,
    action_range_constraint: ActionRangeConstraint = ActionRangeConstraint.UNBOUNDED,
) -> dict[str, Any]:
    """Collect controls first, then attempt one fixed-budget learning run."""
    root = Path(output_dir).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ReferenceLearningError("output directory must be empty")
    config = ProjectConfig.load(config_path)
    output_contract = _action_output_contract(
        action_range_constraint,
        config.safety.joint_lower if action_range_constraint is not ActionRangeConstraint.UNBOUNDED else None,
        config.safety.joint_upper if action_range_constraint is not ActionRangeConstraint.UNBOUNDED else None,
    )
    if config.runtime.backend != "mujoco":
        raise ReferenceLearningError("study requires runtime.backend='mujoco'")
    if not config.runtime.actuation_enabled:
        raise ReferenceLearningError("study requires actuation_enabled=true")
    if not config.mujoco.forbid_collisions:
        raise ReferenceLearningError("study requires forbid_collisions=true")
    if config.runtime.action_horizon != ACTION_HORIZON:
        raise ReferenceLearningError(f"study requires action_horizon={ACTION_HORIZON}")
    config = replace(config, runtime=replace(config.runtime, camera_hz=SERVO_HZ))
    if config.runtime.policy_hz != POLICY_HZ or config.runtime.servo_hz != SERVO_HZ:
        raise ReferenceLearningError("frozen study requires 10 Hz policy / 50 Hz servo")
    root.mkdir(parents=True, exist_ok=True)
    with _fixed_runtime():
        return _run_study(root, config, output_contract)


def _development_task() -> SemanticTask:
    return SemanticTask(
        task_id="reference-nudge-block",
        label="sim-reference-nudge-block",
        policy_steps=POLICY_STEPS,
        seeds=INITIAL_CONDITIONS,
        object_body="task_block",
        initial_object_positions=(
            (7, (0.2162682466, 0.1176982024, 0.475)),
            (13, (0.2162682466, 0.1181982024, 0.475)),
        ),
        target_object_position=(0.210, 0.105, 0.470),
        position_tolerance_m=0.004,
    )


def _run_study(
    root: Path, config: ProjectConfig, output_contract: dict[str, Any],
) -> dict[str, Any]:
    task = _development_task()
    duration_s = POLICY_STEPS / config.runtime.policy_hz
    target = np.asarray((*REFERENCE_TARGET, *([0.0] * 6)), dtype=np.float32)
    times = (0.0, (RAMP_TICKS - 1) / config.runtime.servo_hz, duration_s)
    targets = np.stack((target / RAMP_TICKS, target, target))
    training = CompactWAMTrainingConfig(
        policy_hz=POLICY_HZ, servo_hz=SERVO_HZ, action_horizon=ACTION_HORIZON,
        action_history_steps=1, ifp_steps=0, stage1_steps=200,
        stage2_steps=800, learning_rate=1e-3, seed=MODEL_SEED,
    )
    rng = np.random.default_rng(MODEL_SEED)
    slots = [
        (seed, kind)
        for seed in INITIAL_CONDITIONS
        for kind in rng.permutation([TrialKind.REFERENCE.value, TrialKind.HOLD.value])
    ]
    frozen_source = _source_hashes()
    protocol = {
        "schema_version": STUDY_SCHEMA,
        "scope": "developmental_seen_task_diagnostic",
        "task": asdict(task),
        "limitations": [
            "reachable reset and trajectory selected using development probes",
            "approximately 1 cm low-margin nudge, not robust manipulation",
            "two reset conditions are technical repeats, not independent tasks",
            "same-task demonstrations are used for both learning and prompting",
        ],
        "config": asdict(config),
        "config_sha256": project_config_sha256(config),
        "source_sha256": frozen_source,
        "runtime": _runtime(),
        "duration_s": duration_s,
        "waypoints": {"times_s": times, "targets": targets.tolist()},
        "training": asdict(training),
        "action_output": output_contract,
        "reference_control_order": [[seed, str(kind)] for seed, kind in slots],
        "reference_gate": "both references succeed and both holds fail the fixed goal",
        "learned_trials": list(INITIAL_CONDITIONS),
        "generalization": "separate future protocol; not attempted in this diagnostic",
    }
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    _write_json_no_overwrite(root / "protocol.json", protocol)
    report: dict[str, Any] = {
        "schema_version": STUDY_SCHEMA,
        "protocol_sha256": canonical_json_sha256(protocol),
        "protocol_file_sha256": file_sha256(root / "protocol.json"),
        "result": "reference_gate_failed",
        "trained": False,
        "deployment_ready": False,
        "zero_shot_claimed": False,
        "real_world_success_claimed": False,
        "trials": [],
        "learning": {"status": "not_attempted", "reason": "reference gate not passed"},
        "generalization": {
            "status": "not_attempted",
            "reason": "bounded seen-task diagnostic; needs a separate untouched protocol",
        },
    }
    try:
        for index, (seed, kind_name) in enumerate(slots):
            _require_source(frozen_source)
            kind = TrialKind(kind_name)
            trial = SimTrialConfig(
                config=config, task=task, seed=seed, output_dir=root / f"control-{index}",
                kind=kind, duration_s=duration_s, episode_index=index,
                times_s=times if kind is TrialKind.REFERENCE else None,
                targets=targets if kind is TrialKind.REFERENCE else None,
            )
            _run_slot(trial, report["trials"])
            _require_source(frozen_source)
        if _reference_gate(report["trials"]):
            report["learning"] = {"status": "running", "phase": "input_validation"}
            _learn_and_evaluate(
                root, config, task, training, frozen_source, report["trials"], report["learning"],
                output_contract,
            )
            report["result"] = (
                "seen_task_success" if report["learning"]["success_count"] == len(task.seeds)
                else "seen_task_not_reproduced"
            )
    except Exception as error:
        # Retain completed slots when an input, training, or evidence check fails.
        report["result"] = "execution_failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        if report["learning"]["status"] == "running":
            report["learning"]["status"] = "failed"
            report["learning"]["reason"] = str(error)
    report["source_unchanged"] = _source_hashes() == frozen_source
    report["artifact_sha256"] = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    report["learned_trials_not_attempted"] = [
        seed for seed in task.seeds
        if not any(trial["kind"] == TrialKind.LEARNED and trial["seed"] == seed
                   for trial in report["trials"])
    ]
    report = json.loads(json.dumps(report, allow_nan=False))
    _write_json_no_overwrite(root / REPORT_NAME, report)
    return report


def _run_slot(
    trial: SimTrialConfig, trials: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        outcome = run_sim_trial(trial)
    except Exception as error:
        outcome = {
            "status": "execution_failure", "object_success": False,
            "failure_reason": "trial_exception",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        trials.append({"kind": trial.kind.value, "seed": trial.seed, "outcome": outcome})
        raise
    trials.append({"kind": trial.kind.value, "seed": trial.seed, "outcome": outcome})
    return outcome


def _reference_gate(trials: Sequence[dict[str, Any]]) -> bool:
    matched_outcomes = len(trials) == 4 and all(
        trial["outcome"]["status"] == "scored"
        and trial["outcome"]["object_success"] is (trial["kind"] == TrialKind.REFERENCE)
        for trial in trials
    )
    if not matched_outcomes:
        return False
    for trial in trials:
        outcome = trial["outcome"]
        if trial["kind"] == TrialKind.REFERENCE:
            if outcome.get("reference_success") is not True:
                raise ReferenceLearningError("reference_success must be true for successful reference")
            if not isinstance(outcome.get("episode"), dict) or not outcome["episode"]:
                raise ReferenceLearningError("successful reference requires an episode payload")
        elif outcome.get("reference_success") is True:
            raise ReferenceLearningError("hold cannot have positive reference_success")
    return True


def _require_source(expected: dict[str, str]) -> None:
    if _source_hashes() != expected:
        raise ReferenceLearningError("source changed after protocol freeze")


def _episode_inputs(root: Path, trials: Sequence[dict[str, Any]]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for trial in trials:
        if trial["kind"] != TrialKind.REFERENCE:
            continue
        episode = trial["outcome"]["episode"]
        for kind in ("npz", "manifest"):
            path = Path(episode[f"{kind}_path"]).resolve()
            if not path.is_relative_to(root):
                raise ReferenceLearningError("reference input must be inside this run")
            inputs[str(path)] = episode[f"{kind}_sha256"]
    _require_inputs(inputs)
    return inputs


def _require_inputs(expected: dict[str, str]) -> None:
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ReferenceLearningError(f"input changed after collection: {path}")


def _learn_and_evaluate(
    root: Path, config: ProjectConfig, task: SemanticTask,
    training: CompactWAMTrainingConfig, frozen_source: dict[str, str],
    trials: list[dict[str, Any]], learning: dict[str, Any],
    output_contract: dict[str, Any],
) -> None:
    inputs = _episode_inputs(root, trials)
    _write_json_no_overwrite(root / "training-protocol.json", {
        "inputs": inputs, "training": asdict(training), "source_sha256": frozen_source,
        "action_output": output_contract,
    })
    episode_paths = [path for path in inputs if path.endswith(".npz")]
    records = load_episode_records(episode_paths)
    _require_inputs(inputs)
    _require_source(frozen_source)
    learning["phase"] = "training"
    output_mode = ActionRangeConstraint(output_contract["mode"])
    bounded = output_mode is not ActionRangeConstraint.UNBOUNDED
    checkpoint_id = (
        f"sim-reference-seen-task-{output_mode.value}-seed-7" if bounded
        else "sim-reference-seen-task-seed-7"
    )
    artifacts = train_seen_task_candidate(
        records, checkpoint_path=root / "candidate.pt", report_path=root / "training-report.json",
        checkpoint_id=checkpoint_id, config=training, device="cpu",
        action_range_constraint=output_mode,
        joint_lower=config.safety.joint_lower if bounded else None,
        joint_upper=config.safety.joint_upper if bounded else None,
    )
    _require_inputs(inputs)
    _require_source(frozen_source)
    learning["artifacts"] = asdict(artifacts)
    learning["phase"] = "closed_loop"
    candidate_inputs = {
        **inputs, artifacts.checkpoint_path: file_sha256(artifacts.checkpoint_path),
        artifacts.report_path: file_sha256(artifacts.report_path),
    }
    success_count = 0
    for index, seed in enumerate(task.seeds):
        _require_inputs(candidate_inputs)
        _require_source(frozen_source)
        trial = SimTrialConfig(
            config=config, task=task, seed=seed, output_dir=root / f"learned-{index}",
            kind=TrialKind.LEARNED, duration_s=task.policy_steps / config.runtime.policy_hz,
            episode_index=index + 4, checkpoint_path=Path(artifacts.checkpoint_path),
            prompt_path=Path(episode_paths[0]),
        )
        outcome = _run_slot(trial, trials)
        success_count += int(outcome["status"] == "scored" and outcome["object_success"] is True)
        _require_inputs(candidate_inputs)
        _require_source(frozen_source)
    learning.update({
        "status": "completed", "phase": "complete",
        "success_count": success_count, "scheduled_trials": len(task.seeds),
        "evaluation_scope": "same-task closed-loop development diagnostic",
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_ROBOT_FREE_CONFIG_PATH)
    parser.add_argument(
        "--action-range-constraint", choices=[mode.value for mode in ActionRangeConstraint],
        default=ActionRangeConstraint.UNBOUNDED.value,
    )
    args = parser.parse_args(argv)
    report = run_reference_learning(
        args.output_dir, config_path=args.config,
        action_range_constraint=ActionRangeConstraint(args.action_range_constraint),
    )
    print(f"{report['result']}: {args.output_dir / REPORT_NAME}")
    return int(report["result"] == "execution_failure")


if __name__ == "__main__":
    raise SystemExit(main())
