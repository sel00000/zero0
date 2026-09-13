from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from so101_wam.checkpoint import load_compact_wam_bundle
from so101_wam.constants import ACTION_DIM, PRIMARY_CAMERA_COUNT
from so101_wam.ifp_ablation import (
    ClosedLoopMode,
    ClosedLoopResult,
    IFPAblationError,
    MujocoSemanticSuiteProxy,
    MujocoTerminalProxy,
    _semantic_protocol_hash,
    main as ifp_main,
    run_ifp_ablation,
)
from so101_wam.config import DEFAULT_ROBOT_FREE_CONFIG_PATH, ProjectConfig
from so101_wam.deployment import canonical_json_sha256, file_sha256, project_config_sha256
from so101_wam.mujoco_semantic_suite import SEMANTIC_SUITE_REPORT_SCHEMA
from so101_wam.model import CompactWAM, FusedIFP
from so101_wam.robot_free_cli import _write_synthetic_episodes
from so101_wam.robot_free_semantic_cli import (
    _write_mapping_manifest,
    _write_semantic_episodes,
    _write_semantic_manifests,
    _write_suite_manifest,
)
from so101_wam.training import (
    CompactWAMTrainingConfig,
    IFPArchitecture,
    TrainingArtifactExistsError,
    train_offline_candidate,
)
from so101_wam.training_data import EpisodeRecord, load_episode_records


MUJOCO_MODEL_IDENTITY = {
    "schema_version": "so101_wam.mujoco_compiled_model.v1",
    "engine_version": "3.12.0",
    "compiled_model_sha256": "f" * 64,
    "compiled_model_bytes": 128,
}


def _batch(
    *,
    batch: int = 2,
    prompt_steps: int = 3,
    live_steps: int = 2,
    size: int = 8,
) -> tuple[torch.Tensor, ...]:
    prompt = torch.randint(
        0,
        256,
        (batch, prompt_steps, PRIMARY_CAMERA_COUNT, 3, size, size),
        dtype=torch.uint8,
    )
    live = torch.randint(
        0,
        256,
        (batch, live_steps, PRIMARY_CAMERA_COUNT, 3, size, size),
        dtype=torch.uint8,
    )
    prompt_proprio = torch.randn(batch, prompt_steps, ACTION_DIM)
    prompt_actions = torch.randn(batch, prompt_steps, ACTION_DIM)
    live_proprio = torch.randn(batch, live_steps, ACTION_DIM)
    live_actions = torch.randn(batch, live_steps, ACTION_DIM)
    prompt_mask = torch.ones(batch, prompt_steps, dtype=torch.bool)
    return (
        prompt,
        prompt_proprio,
        prompt_actions,
        live,
        live_proprio,
        live_actions,
        prompt_mask,
    )


def _config(*, ifp_steps: int) -> CompactWAMTrainingConfig:
    return CompactWAMTrainingConfig(
        latent_dim=8,
        transformer_layers=2,
        transformer_heads=2,
        future_steps=1,
        action_horizon=2,
        action_history_steps=1,
        ifp_steps=ifp_steps,
        ifp_stride=2,
        ifp_architecture=IFPArchitecture.FUSED_MODULES,
        ifp_window_steps=4,
        stage1_steps=0,
        stage2_steps=1,
        seed=23,
    )


def _records(
    tmp_path: Path,
) -> tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]]:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    _write_synthetic_episodes(
        train_dir=train_dir,
        validation_dir=validation_dir,
        seed=23,
    )
    return load_episode_records([train_dir]), load_episode_records([validation_dir])


def test_fused_ifp_uses_layer_trace_and_stays_out_of_inference_state() -> None:
    torch.manual_seed(5)
    model = CompactWAM(
        latent_dim=8,
        transformer_layers=2,
        transformer_heads=2,
        future_steps=1,
        action_horizon=2,
        action_history_steps=1,
        ifp_steps=0,
    )
    inputs = _batch()
    ordinary = model(*inputs[:-1], prompt_mask=inputs[-1], compute_ifp=False)
    traced, layer_features = model.forward_with_context_features(
        *inputs[:-1],
        prompt_mask=inputs[-1],
    )

    assert layer_features.shape == (2, 2, 8)
    assert torch.equal(ordinary["future_latents"], traced["future_latents"])
    assert torch.equal(ordinary["actions"], traced["actions"])

    auxiliary = FusedIFP(model, ifp_steps=4)
    rng_state = torch.random.get_rng_state().clone()
    predicted = auxiliary(layer_features)
    assert predicted.shape == (2, 4, PRIMARY_CAMERA_COUNT, 8)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    last_layer = model.temporal.layers[-1].state_dict()
    for module in auxiliary.future_modules:
        for key, value in last_layer.items():
            assert torch.equal(module.state_dict()[key], value)

    predicted.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.temporal.parameters())
    assert any(parameter.grad is not None for parameter in auxiliary.parameters())

    before = model.infer_action(*inputs[:-1], prompt_mask=inputs[-1])
    with torch.no_grad():
        for parameter in auxiliary.parameters():
            parameter.add_(1.0)
    after = model.infer_action(*inputs[:-1], prompt_mask=inputs[-1])

    assert torch.equal(before, after)
    assert all("ifp" not in key for key in model.state_dict())


def test_closed_loop_result_rejects_non_hex_artifact_hash() -> None:
    with pytest.raises(IFPAblationError, match="artifact metadata"):
        ClosedLoopResult(
            scope="synthetic_terminal_joint_proxy",
            protocol_sha256="a" * 64,
            trial_count=1,
            scored_trial_count=1,
            execution_failure_count=0,
            success_count=0,
            success_rate=0.0,
            terminal_error_mean=1.0,
            terminal_error_basis="max_abs_joint_position_error",
            failure_counts=(("terminal_joint_target_tolerance", 1),),
            artifact="trial.json",
            artifact_sha256="z" * 64,
        )


def test_cli_rejects_horizon_mismatch_before_loading_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "ablation"
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "mujoco_robot_free.toml"
    )

    with pytest.raises(SystemExit) as raised:
        ifp_main(
            [
                "--train-episodes",
                str(tmp_path / "missing-train"),
                "--validation-episodes",
                str(tmp_path / "missing-validation"),
                "--output-dir",
                str(output_dir),
                "--checkpoint-id-prefix",
                "mismatch",
                "--mujoco-config",
                str(config_path),
                "--terminal-tolerance",
                "1.0",
                "--action-horizon",
                "9",
            ]
        )

    assert raised.value.code == 2
    assert "action_horizon must match MuJoCo runtime" in capsys.readouterr().err
    assert not output_dir.exists()


def test_fused_ifp_training_exports_an_inference_only_checkpoint(
    tmp_path: Path,
) -> None:
    train_records, validation_records = _records(tmp_path)
    checkpoint_path = tmp_path / "candidate.pt"
    report_path = tmp_path / "candidate.training.json"

    train_offline_candidate(
        train_records,
        validation_records,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        checkpoint_id="ifp-k4-test",
        config=_config(ifp_steps=4),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = load_compact_wam_bundle(checkpoint_path)
    assert report["protocol"]["ifp_architecture"] == "fused_modules"
    assert report["protocol"]["ifp_loss_weights"] == [0.5, 0.25, 0.15, 0.15]
    assert report["protocol"]["ifp_module_removed_from_inference"] is True
    assert report["model"]["ifp_steps"] == 0
    assert report["optimization"]["ifp_steps"] == 4
    assert report["optimization"]["ifp_window_steps"] == 4
    assert report["optimization"]["stage2_final_ifp_loss"] >= 0.0
    assert len(report["initial_model_state_sha256"]) == 64
    assert len(report["training_schedule_sha256"]) == 64
    assert report["validation"]["null_prompt_action_mean_abs_delta"] >= 0.0
    assert report["validation"]["temporal_prompt_action_mean_abs_delta"] >= 0.0
    assert bundle.model.ifp_steps == 0
    assert all("ifp" not in key for key in bundle.model.state_dict())
    assert bundle.metadata["training_ifp_steps"] == 4
    assert bundle.metadata["ifp_architecture"] == "fused_modules"
    assert bundle.metadata["inference_ifp_module_present"] is False


@pytest.mark.parametrize("protocol_mode", ["same", "changed"])
def test_ifp_ablation_enforces_comparability_and_records_negative_results(
    tmp_path: Path,
    protocol_mode: str,
) -> None:
    train_records, validation_records = _records(tmp_path)
    output_dir = tmp_path / "ablation"
    report_path = tmp_path / "ifp_ablation.json"

    def closed_loop(checkpoint_path: Path, *, ifp_steps: int) -> ClosedLoopResult:
        assert checkpoint_path.is_file()
        success_count = int(ifp_steps == 4)
        return ClosedLoopResult(
            scope="synthetic_terminal_joint_proxy",
            protocol_sha256=(
                "b" * 64 if protocol_mode == "changed" and ifp_steps == 4
                else "a" * 64
            ),
            trial_count=1,
            scored_trial_count=1,
            execution_failure_count=0,
            success_count=success_count,
            success_rate=float(success_count),
            terminal_error_mean=float(4 - ifp_steps),
            terminal_error_basis="max_abs_joint_position_error",
            failure_counts=(
                ()
                if success_count
                else (("terminal_joint_target_tolerance", 1),)
            ),
        )

    if protocol_mode == "changed":
        with pytest.raises(IFPAblationError, match="same_closed_loop_protocol"):
            run_ifp_ablation(
                train_records,
                validation_records,
                output_dir=output_dir,
                report_path=report_path,
                checkpoint_id_prefix="ifp-protocol-drift",
                base_config=_config(ifp_steps=0),
                closed_loop_evaluator=closed_loop,
            )
        assert not report_path.exists()
        return

    report = run_ifp_ablation(
        train_records,
        validation_records,
        output_dir=output_dir,
        report_path=report_path,
        checkpoint_id_prefix="ifp-ablation-test",
        base_config=_config(ifp_steps=0),
        closed_loop_evaluator=closed_loop,
    )

    assert report["result"] == "complete"
    assert report["robot_used"] is False
    assert report["scientific_claim"] == "not_evaluated"
    assert [variant["ifp_steps"] for variant in report["variants"]] == [0, 2, 4]
    assert report["comparability"] == {
        "same_closed_loop_protocol": True,
        "same_initial_model_state": True,
        "same_optimizer_budget": True,
        "same_seed": True,
        "same_train_split": True,
        "same_training_schedule": True,
        "same_validation_split": True,
        "same_window_count": True,
    }
    assert [
        variant["closed_loop"]["success_count"] for variant in report["variants"]
    ] == [0, 0, 1]
    assert all(
        variant["inference"]["ifp_module_present"] is False
        for variant in report["variants"]
    )
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert len(tuple(output_dir.glob("*.pt"))) == 3
    assert len(tuple(output_dir.glob("*.training.json"))) == 3
    assert np.isfinite(
        [
            variant["validation"]["action_mse_normalized"]
            for variant in report["variants"]
        ]
    ).all()

    with pytest.raises(TrainingArtifactExistsError, match="must be empty"):
        run_ifp_ablation(
            train_records,
            validation_records,
            output_dir=output_dir,
            report_path=tmp_path / "second_report.json",
            checkpoint_id_prefix="ifp-ablation-repeat",
            base_config=_config(ifp_steps=0),
            closed_loop_evaluator=closed_loop,
        )


def test_ifp_ablation_rejects_closed_loop_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    train_records, validation_records = _records(tmp_path)
    output_dir = tmp_path / "ablation"
    report_path = tmp_path / "ifp_ablation.json"

    def closed_loop(checkpoint_path: Path, *, ifp_steps: int) -> ClosedLoopResult:
        artifact_path = checkpoint_path.with_suffix(".mujoco.json")
        artifact_path.write_text("{}\n", encoding="utf-8")
        return ClosedLoopResult(
            scope="synthetic_terminal_joint_proxy",
            protocol_sha256="a" * 64,
            trial_count=1,
            scored_trial_count=1,
            execution_failure_count=0,
            success_count=0,
            success_rate=0.0,
            terminal_error_mean=float(ifp_steps),
            terminal_error_basis="max_abs_joint_position_error",
            failure_counts=(("terminal_joint_target_tolerance", 1),),
            artifact=artifact_path.name,
            artifact_sha256="0" * 64,
        )

    with pytest.raises(IFPAblationError, match="artifact checksum mismatch"):
        run_ifp_ablation(
            train_records,
            validation_records,
            output_dir=output_dir,
            report_path=report_path,
            checkpoint_id_prefix="ifp-artifact-mismatch",
            base_config=_config(ifp_steps=0),
            closed_loop_evaluator=closed_loop,
        )

    assert not report_path.exists()


def test_mujoco_terminal_proxy_scores_and_publishes_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "k4.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    prompt_path = tmp_path / "prompt.npz"
    prompt_path.write_bytes(b"prompt")
    prompt_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    target = np.arange(ACTION_DIM, dtype=np.float64)
    final = target.copy()
    final[3] += 0.25
    model_identity = dict(MUJOCO_MODEL_IDENTITY)

    def fake_session(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "result": "pass",
            "policy": "compact_wam",
            "mujoco_model_identity": dict(model_identity),
            "terminal_joint_position": final.tolist(),
        }

    monkeypatch.setattr(
        "so101_wam.ifp_ablation.run_mujoco_checkpoint_session",
        fake_session,
    )
    evaluator = MujocoTerminalProxy(
        config=ProjectConfig.load(
            Path(__file__).resolve().parents[1] / "configs" / "mujoco_robot_free.toml"
        ),
        prompt_path=prompt_path,
        target_joint_position=tuple(float(value) for value in target),
        tolerance=0.5,
        policy_steps=2,
        device="cpu",
    )

    result = evaluator(checkpoint_path, ifp_steps=4)

    assert result.success_count == 1
    assert result.success_rate == 1.0
    assert result.scored_trial_count == 1
    assert result.execution_failure_count == 0
    assert result.failure_counts == ()
    assert result.terminal_error_basis == "max_abs_joint_position_error"
    assert result.terminal_error_mean == 0.25
    assert len(result.protocol_sha256) == 64
    assert result.artifact == "k4.mujoco.json"
    assert result.artifact_sha256 is not None
    report = json.loads((tmp_path / "k4.mujoco.json").read_text(encoding="utf-8"))
    assert report["terminal_proxy"]["success"] is True
    assert report["terminal_proxy"]["ifp_steps"] == 4
    assert result.protocol_sha256 == canonical_json_sha256(report["evaluation_protocol"])

    changed = replace(evaluator, tolerance=0.1)(
        tmp_path / "k0.pt", ifp_steps=0
    )
    assert changed.protocol_sha256 != result.protocol_sha256

    model_identity["engine_version"] = "3.12.1"
    changed_engine = evaluator(tmp_path / "k2.pt", ifp_steps=2)
    assert changed_engine.protocol_sha256 != result.protocol_sha256


def _suite_identity(config: object, paths: dict[str, object]) -> dict[str, object]:
    assert isinstance(config, ProjectConfig)
    return {
        "schema_version": SEMANTIC_SUITE_REPORT_SCHEMA,
        "independent_mapping_verified": False,
        "mujoco_model_identity": dict(MUJOCO_MODEL_IDENTITY),
        "mujoco_config_sha256": project_config_sha256(config),
        "suite_sha256": file_sha256(Path(str(paths["suite_path"]))),
        "mapping_sha256": file_sha256(Path(str(paths["mapping_path"]))),
        "checkpoint_sha256": file_sha256(Path(str(paths["checkpoint_path"]))),
        "training_report_sha256": file_sha256(Path(str(paths["training_report_path"]))),
        "cases": [
            {
                "case_id": case_id,
                "semantic_manifest_sha256": str(index) * 64,
                "prompt_npz_sha256": str(index + 1) * 64,
                "prompt_manifest_sha256": str(index + 2) * 64,
                "object_task_signature_sha256": str(index + 3) * 64,
                "object_physical_profile_sha256": str(index + 4) * 64,
            }
            for index, case_id in enumerate(("left", "right"), start=1)
        ],
    }


@pytest.mark.parametrize(
    "field",
    [
        "suite_sha256", "mapping_sha256", "mujoco_config_sha256",
        "semantic_manifest_sha256", "prompt_npz_sha256", "prompt_manifest_sha256",
        "object_task_signature_sha256", "object_physical_profile_sha256", "device",
    ],
)
def test_semantic_protocol_binds_inputs(tmp_path: Path, field: str) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    config = ProjectConfig.load(DEFAULT_ROBOT_FREE_CONFIG_PATH)
    report = _suite_identity(config, {
        key: source for key in (
            "suite_path", "mapping_path", "checkpoint_path", "training_report_path"
        )
    })
    original = _semantic_protocol_hash(report, device="cpu")

    # Results, exported models and case serialization order are not protocol inputs.
    changed_results = deepcopy(report)
    changed_results["checkpoint_sha256"] = "c" * 64
    changed_results["training_report_sha256"] = "d" * 64
    changed_results["summary"] = {"success_rate": 0.0}
    changed_results["cases"].reverse()
    assert _semantic_protocol_hash(changed_results, device="cpu") == original

    device = "cpu"
    if field == "device":
        device = "cuda"
    elif field in report:
        report[field] = "e" * 64
    else:
        report["cases"][0][field] = "e" * 64
    assert _semantic_protocol_hash(report, device=device) != original


@pytest.mark.parametrize(
    ("field", "changed"),
    [("compiled_model_sha256", "b" * 64), ("engine_version", "3.12.1")],
)
def test_ifp_protocol_binds_model(field: str, changed: str) -> None:
    report = {
        "suite_sha256": "a" * 64,
        "mapping_sha256": "b" * 64,
        "mujoco_config_sha256": "c" * 64,
        "mujoco_model_identity": dict(MUJOCO_MODEL_IDENTITY),
        "cases": [
            {
                "case_id": case,
                **{key: "d" * 64 for key in (
                    "semantic_manifest_sha256", "prompt_npz_sha256",
                    "prompt_manifest_sha256", "object_task_signature_sha256",
                    "object_physical_profile_sha256",
                )},
            }
            for case in ("block", "cylinder")
        ],
    }
    before = _semantic_protocol_hash(report, device="cpu")
    report["mujoco_model_identity"][field] = changed

    assert _semantic_protocol_hash(report, device="cpu") != before


@pytest.mark.parametrize("report_mode", ["same", "diverged", "boolean_on_disk"])
def test_mujoco_semantic_suite_proxy_scores_and_publishes_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_mode: str,
) -> None:
    checkpoint_path = tmp_path / "k4.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    training_report_path = tmp_path / "k4.training.json"
    training_report_path.write_text("{}\n", encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text("{}\n", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_suite(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        artifact_dir = Path(str(kwargs["artifact_dir"]))
        artifact_dir.mkdir()
        report = {
            **_suite_identity(args[0], kwargs),
            "result": "complete",
            "robot_used": False,
            "semantic_mujoco_object_state_evaluated": True,
            "semantic_heldout_success_claimed": False,
            "prompt_causality_claimed": False,
            "real_world_success_claimed": False,
            "official_zero_wam_claimed": False,
            "summary": {
                "total_trial_count": 6,
                "scored_trial_count": 5,
                "execution_failure_count": 1,
                "success_count": 2,
                "success_rate": 2 / 6,
                "object_position_error_mean_m": 0.25,
                "failure_counts": {
                    "object_body_position_tolerance": 3,
                    "stale_action": 1,
                },
            },
        }
        report_path = Path(str(kwargs["report_path"]))
        persisted = deepcopy(report)
        if report_mode == "boolean_on_disk":
            persisted["summary"]["execution_failure_count"] = True
        report_path.write_text(json.dumps(persisted), encoding="utf-8")
        if report_mode == "diverged":
            report["summary"]["object_position_error_mean_m"] = 0.1
        return report

    monkeypatch.setattr(
        "so101_wam.ifp_ablation.run_mujoco_semantic_suite",
        fake_suite,
    )
    evaluator = MujocoSemanticSuiteProxy(
        config=ProjectConfig.load(
            Path(__file__).resolve().parents[1] / "configs" / "mujoco_robot_free.toml"
        ),
        suite_path=suite_path,
        mapping_path=mapping_path,
        device="cpu",
    )

    if report_mode != "same":
        message = "persisted" if report_mode == "diverged" else "execution_failure_count"
        with pytest.raises(IFPAblationError, match=message):
            evaluator(checkpoint_path, ifp_steps=4)
        return

    result = evaluator(checkpoint_path, ifp_steps=4)

    assert calls == [
        {
            "suite_path": suite_path,
            "mapping_path": mapping_path,
            "checkpoint_path": checkpoint_path,
            "training_report_path": training_report_path,
            "artifact_dir": tmp_path / "k4.semantic-suite-artifacts",
            "report_path": tmp_path / "k4.semantic-suite.json",
            "device": "cpu",
        }
    ]
    assert result.scope == "mujoco_semantic_object_state_suite"
    assert result.trial_count == 6
    assert result.scored_trial_count == 5
    assert result.execution_failure_count == 1
    assert result.success_count == 2
    assert result.success_rate == pytest.approx(2 / 6)
    assert result.terminal_error_mean == pytest.approx(0.25)
    assert result.terminal_error_basis == "object_position_error_m_scored_trials"
    assert dict(result.failure_counts) == {
        "object_body_position_tolerance": 3,
        "stale_action": 1,
    }
    assert result.artifact == "k4.semantic-suite.json"
    assert result.artifact_sha256 == file_sha256(
        tmp_path / "k4.semantic-suite.json"
    )


def test_semantic_proxy_keeps_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "k2.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint_path.with_suffix(".training.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text("{}\n", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text("{}\n", encoding="utf-8")

    def fake_suite(*args: object, **kwargs: object) -> dict[str, object]:
        Path(str(kwargs["artifact_dir"])).mkdir()
        report = {
            **_suite_identity(args[0], kwargs),
            "result": "complete",
            "robot_used": False,
            "semantic_mujoco_object_state_evaluated": False,
            "semantic_heldout_success_claimed": False,
            "prompt_causality_claimed": False,
            "real_world_success_claimed": False,
            "official_zero_wam_claimed": False,
            "summary": {
                "total_trial_count": 3,
                "scored_trial_count": 0,
                "execution_failure_count": 3,
                "success_count": 0,
                "success_rate": 0.0,
                "object_position_error_mean_m": None,
                "failure_counts": {"stale_action": 3},
            },
        }
        Path(str(kwargs["report_path"])).write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        "so101_wam.ifp_ablation.run_mujoco_semantic_suite",
        fake_suite,
    )
    evaluator = MujocoSemanticSuiteProxy(
        config=ProjectConfig.load(
            Path(__file__).resolve().parents[1] / "configs" / "mujoco_robot_free.toml"
        ),
        suite_path=suite_path,
        mapping_path=mapping_path,
    )

    result = evaluator(checkpoint_path, ifp_steps=2)

    assert result.trial_count == result.execution_failure_count == 3
    assert result.scored_trial_count == result.success_count == 0
    assert result.terminal_error_mean is None
    assert dict(result.failure_counts) == {"stale_action": 3}
    assert result.artifact_sha256 == file_sha256(
        checkpoint_path.with_suffix(".semantic-suite.json")
    )


@pytest.mark.parametrize("field", ["success_rate", "terminal_error_mean"])
def test_ifp_rejects_boolean_metrics(field: str) -> None:
    result = ClosedLoopResult(
        scope="synthetic_terminal_joint_proxy",
        protocol_sha256="a" * 64,
        trial_count=1,
        scored_trial_count=1,
        execution_failure_count=0,
        success_count=1,
        success_rate=1.0,
        terminal_error_mean=1.0,
        terminal_error_basis="max_abs_joint_position_error",
        failure_counts=(),
    )

    with pytest.raises(IFPAblationError):
        replace(result, **{field: True})


def test_ifp_cli_selects_semantic_suite_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text("{}\n", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "so101_wam.ifp_ablation.load_episode_records",
        lambda paths: (object(),),
    )

    def fake_ablation(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        captured.update(kwargs)
        return {"result": "complete"}

    monkeypatch.setattr(
        "so101_wam.ifp_ablation.run_ifp_ablation",
        fake_ablation,
    )
    result = ifp_main(
        [
            "--train-episodes",
            str(tmp_path / "train"),
            "--validation-episodes",
            str(tmp_path / "validation"),
            "--output-dir",
            str(tmp_path / "output"),
            "--checkpoint-id-prefix",
            "semantic-suite",
            "--closed-loop-mode",
            ClosedLoopMode.SEMANTIC_SUITE_PROXY.value,
            "--semantic-suite",
            str(suite_path),
            "--semantic-mapping",
            str(mapping_path),
        ]
    )

    assert result == 0
    evaluator = captured["closed_loop_evaluator"]
    assert isinstance(evaluator, MujocoSemanticSuiteProxy)
    assert evaluator.suite_path == suite_path
    assert evaluator.mapping_path == mapping_path
    assert json.loads(capsys.readouterr().out) == {"result": "complete"}


def test_ifp_real_semantic_suite(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    config = ProjectConfig.load(DEFAULT_ROBOT_FREE_CONFIG_PATH)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    prompts = _write_semantic_episodes(
        train_dir=train_dir, validation_dir=validation_dir, seed=23
    )
    semantic_paths = _write_semantic_manifests(tmp_path)
    suite_path = tmp_path / "suite.json"
    mapping_path = tmp_path / "mapping.json"
    _write_suite_manifest(
        tmp_path, suite_path=suite_path, semantic_paths=semantic_paths,
        prompts=prompts, seed=23,
    )
    _write_mapping_manifest(mapping_path, semantic_paths=semantic_paths, seed=23)
    output_dir = tmp_path / "ablation"
    report_path = tmp_path / "ifp_ablation.json"

    report = run_ifp_ablation(
        load_episode_records([train_dir]),
        load_episode_records([validation_dir]),
        output_dir=output_dir,
        report_path=report_path,
        checkpoint_id_prefix="semantic-integration",
        base_config=replace(
            _config(ifp_steps=0), action_horizon=config.runtime.action_horizon
        ),
        closed_loop_evaluator=MujocoSemanticSuiteProxy(
            config=config, suite_path=suite_path, mapping_path=mapping_path,
        ),
    )

    assert report["result"] == "complete"
    assert report["schema_version"] == "so101_wam.ifp_ablation.v4"
    assert all(report["comparability"].values())
    assert report["scientific_claim"] == "not_evaluated"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    protocols = set()
    for variant in report["variants"]:
        closed = variant["closed_loop"]
        protocols.add(closed["protocol_sha256"])
        artifact = output_dir / closed["artifact"]
        suite = json.loads(artifact.read_text(encoding="utf-8"))
        summary = suite["summary"]
        assert closed["artifact_sha256"] == file_sha256(artifact)
        assert closed["trial_count"] == summary["total_trial_count"] == 6
        assert closed["scored_trial_count"] == summary["scored_trial_count"]
        assert closed["execution_failure_count"] == summary["execution_failure_count"]
        assert closed["failure_counts"] == summary["failure_counts"]
        assert closed["terminal_error_mean"] == summary["object_position_error_mean_m"]
        assert suite["checkpoint_sha256"] == variant["checkpoint_sha256"]
        assert suite["training_report_sha256"] == variant["training_report_sha256"]
        assert suite["semantic_heldout_success_claimed"] is False
    assert len(protocols) == 1
