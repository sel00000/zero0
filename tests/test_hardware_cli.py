from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_wam.adapters.fake import FakeBimanualRobot
from so101_wam.checkpoint import save_compact_wam_checkpoint
from so101_wam.config import LeRobotConfig, ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.constants import ACTION_DIM, ARM_JOINT_NAMES
from so101_wam.contracts import SensorimotorFrame
from so101_wam.dataset import EpisodeBuffer
from so101_wam.deployment import (
    DEPLOYMENT_AUTHORIZATION_SCOPE,
    DEPLOYMENT_CERTIFICATION_KIND,
    DEPLOYMENT_CERTIFICATION_SCHEMA,
    DEPLOYMENT_SOURCE_EVIDENCE_KEYS,
    DeploymentCertificationError,
    deployment_evidence_sha256,
    file_sha256,
    project_config_sha256,
)
from so101_wam.deployment_issuer import DeploymentIssuanceError
from so101_wam.hardware_cli import (
    HardwareCLIError,
    _require_deployment_checkpoint,
    resolve_output_mode,
    run_hardware_session,
)
from so101_wam.model import ActionDecoder, CompactWAM
from so101_wam.training import train_offline_candidate
from so101_wam.training_data import EpisodeRecord


class FakeClock:
    def __init__(self) -> None:
        self.now_s = 50.0

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s


class AliveThread:
    def is_alive(self) -> bool:
        return True


class PinnedBus:
    def __init__(self, fake: FakeBimanualRobot, *, offset: int) -> None:
        self.fake = fake
        self.offset = offset

    def sync_read(self, data_name: str, *, num_retry: int = 0) -> dict[str, float]:
        assert data_name == "Present_Position"
        assert num_retry == 0
        values = self.fake.joint_position[self.offset : self.offset + 6]
        return {
            joint: float(value)
            for joint, value in zip(ARM_JOINT_NAMES, values, strict=True)
        }


class PinnedFrameLock:
    def __init__(self, camera) -> None:
        self.camera = camera

    def __enter__(self):
        self.camera.refresh()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


class PinnedCamera:
    def __init__(
        self,
        fake: FakeBimanualRobot,
        *,
        key: str,
        timestamp_offset_s: float,
    ) -> None:
        self.fake = fake
        self.key = key
        self.timestamp_offset_s = timestamp_offset_s
        self.latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.latest_timestamp = perf_counter()
        self.thread = AliveThread()
        self.frame_lock = PinnedFrameLock(self)
        self.is_connected = True

    def refresh(self) -> None:
        self.latest_frame = self.fake.get_lerobot_observation()[self.key]
        self.latest_timestamp = perf_counter() + self.timestamp_offset_s

    def disconnect(self) -> None:
        self.is_connected = False
        self.thread = None


class DummyBiSOFollower:
    def __init__(self) -> None:
        self.fake = FakeBimanualRobot()
        self.is_connected = False
        self.is_calibrated = True
        self.disconnect_calls = 0
        self.left_arm = SimpleNamespace(
            bus=PinnedBus(self.fake, offset=0),
            cameras={
                "wrist": PinnedCamera(
                    self.fake,
                    key="left_wrist",
                    timestamp_offset_s=-0.005,
                )
            },
            config=SimpleNamespace(num_read_retries=0),
        )
        self.right_arm = SimpleNamespace(
            bus=PinnedBus(self.fake, offset=6),
            cameras={
                "wrist": PinnedCamera(
                    self.fake,
                    key="right_wrist",
                    timestamp_offset_s=0.0,
                )
            },
            config=SimpleNamespace(num_read_retries=0),
        )

    def connect(self, *, calibrate: bool = True) -> None:
        assert calibrate is False
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    def get_observation(self) -> dict[str, object]:
        return self.fake.get_lerobot_observation()

    def get_atomic_observation(
        self,
    ) -> tuple[dict[str, object], dict[str, float]]:
        now_s = perf_counter()
        return self.fake.get_lerobot_observation(), {
            "left_wrist": now_s - 0.005,
            "right_wrist": now_s,
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self.fake.send_action(action)


def _config(*, actuation: bool) -> ProjectConfig:
    lerobot = LeRobotConfig(
        left_port="left",
        right_port="right",
        left_wrist_camera=0,
        right_wrist_camera=1,
        calibration_dir="calibration",
        hardware_id="bench-a",
        calibration_id="cal-a",
        home_joint_position=(0.0,) * ACTION_DIM,
        home_joint_tolerance=(0.1,) * ACTION_DIM,
    )
    return ProjectConfig(
        runtime=RuntimeConfig(backend="lerobot", actuation_enabled=actuation),
        safety=SafetyConfig(
            joint_lower=(-10.0,) * ACTION_DIM,
            joint_upper=(10.0,) * ACTION_DIM,
            max_delta_per_servo_tick=(1.0,) * ACTION_DIM,
            calibrated=actuation,
        ),
        lerobot=lerobot,
    )


def _save_prompt(directory: Path, *, offset: int = 0) -> tuple[Path, Path]:
    episode = EpisodeBuffer(fps=30.0, task="pick", episode_index=0)
    for index in range(91):
        value = (index + offset) / 90.0
        episode.append(
            SensorimotorFrame(
                timestamp_s=index / 30.0,
                images={
                    "left_wrist": np.full(
                        (8, 8, 3), (index + offset) % 255, dtype=np.uint8
                    ),
                    "right_wrist": np.full(
                        (8, 8, 3), (index + offset + 1) % 255, dtype=np.uint8
                    ),
                },
                joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
                executed_action=np.full(ACTION_DIM, value, dtype=np.float32),
            )
        )
    return episode.save(directory)


def _deployment_certification_core(
    config: ProjectConfig,
    *,
    checkpoint_id: str,
    training_evidence_sha256: str,
    max_policy_steps: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": DEPLOYMENT_CERTIFICATION_SCHEMA,
        "artifact_kind": DEPLOYMENT_CERTIFICATION_KIND,
        "evidence_level": "real",
        "result": "pass",
        "authorization_scope": DEPLOYMENT_AUTHORIZATION_SCOPE,
        "checkpoint_id": checkpoint_id,
        "training_evidence_sha256": training_evidence_sha256,
        "hardware_id": config.lerobot.hardware_id,
        "calibration_id": config.lerobot.calibration_id,
        "config_sha256": project_config_sha256(config),
        "max_policy_steps": max_policy_steps,
        "gate_results": {gate: "pass" for gate in ("G6", "G7", "G8", "G9")},
        "gate_evidence_sha256": {
            gate: str(index) * 64
            for index, gate in enumerate(("G6", "G7", "G8", "G9"), start=1)
        },
        "source_evidence_sha256": {
            source: "a" * 64 for source in DEPLOYMENT_SOURCE_EVIDENCE_KEYS
        },
    }


def test_real_output_requires_config_opt_in_and_exact_hardware_ack() -> None:
    with pytest.raises(HardwareCLIError, match="actuation_enabled=true"):
        resolve_output_mode(
            _config(actuation=False),
            enable_real_output=True,
            acknowledged_hardware_id="bench-a",
        )
    with pytest.raises(HardwareCLIError, match="exactly match"):
        resolve_output_mode(
            _config(actuation=True),
            enable_real_output=True,
            acknowledged_hardware_id="wrong",
        )

    shadow = resolve_output_mode(
        _config(actuation=True),
        enable_real_output=False,
        acknowledged_hardware_id="",
    )
    assert shadow.runtime.actuation_enabled is False


def test_hardware_session_loads_checkpoint_prompt_and_runs_shadow(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.manual_seed(3)
    save_compact_wam_checkpoint(
        CompactWAM(
            latent_dim=8,
            transformer_heads=2,
            future_steps=1,
            action_horizon=10,
            action_history_steps=1,
        ),
        checkpoint,
    )

    prompt_path, manifest_path = _save_prompt(tmp_path)
    raw = DummyBiSOFollower()
    monkeypatch.setattr(
        "so101_wam.hardware_cli.create_bi_so_follower", lambda config: raw
    )
    clock = FakeClock()

    summary = run_hardware_session(
        _config(actuation=False),
        checkpoint_path=checkpoint,
        prompt_path=prompt_path,
        manifest_path=manifest_path,
        policy_steps=1,
        clock=clock,
        sleeper=clock.sleep,
    )

    assert summary.servo_steps == 5
    assert summary.sent_actions == 0
    assert summary.shadow_steps == 5
    assert raw.disconnect_calls == 1


def test_hardware_session_real_output_passes_all_gates_and_sends(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "trained_model.pt"
    config = _config(actuation=True)
    checkpoint_id = "deployment-test-1"
    training_evidence_sha256 = "a" * 64
    prompt_path, manifest_path = _save_prompt(tmp_path)
    certification_core = _deployment_certification_core(
        config,
        checkpoint_id=checkpoint_id,
        training_evidence_sha256=training_evidence_sha256,
    )
    certification_core["source_evidence_sha256"]["prompt_npz"] = file_sha256(  # type: ignore[index]
        prompt_path
    )
    certification_core["source_evidence_sha256"]["prompt_manifest"] = (  # type: ignore[index]
        file_sha256(manifest_path)
    )
    torch.manual_seed(4)
    deployment_model = CompactWAM(
        latent_dim=8,
        transformer_heads=2,
        future_steps=1,
        action_horizon=10,
        action_history_steps=1,
    )
    save_compact_wam_checkpoint(
        deployment_model,
        checkpoint,
        metadata={
            "artifact_kind": "compact_wam_deployment",
            "offline_trained": False,
            "trained": True,
            "deployment_ready": True,
            "checkpoint_id": checkpoint_id,
            "training_evidence_sha256": training_evidence_sha256,
            "deployment_evidence_sha256": deployment_evidence_sha256(
                certification_core
            ),
        },
    )
    certification_path = tmp_path / "deployment.certification.json"
    certification_path.write_text(
        json.dumps(
            {
                **certification_core,
                "checkpoint_sha256": file_sha256(checkpoint),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    raw = DummyBiSOFollower()
    monkeypatch.setattr(
        "so101_wam.hardware_cli.create_bi_so_follower", lambda config: raw
    )
    clock = FakeClock()

    with pytest.raises(HardwareCLIError, match="revalidation paths"):
        run_hardware_session(
            config,
            checkpoint_path=checkpoint,
            prompt_path=prompt_path,
            manifest_path=manifest_path,
            deployment_certification_path=certification_path,
            policy_steps=1,
            enable_real_output=True,
            acknowledged_hardware_id="bench-a",
            clock=clock,
            sleeper=clock.sleep,
        )
    assert raw.disconnect_calls == 0

    certified_sources = certification_core["source_evidence_sha256"]
    monkeypatch.setattr(
        "so101_wam.hardware_cli.validate_deployment_source_evidence",
        lambda *args, **kwargs: SimpleNamespace(
            source_evidence_sha256=certified_sources,
            candidate=SimpleNamespace(model=deployment_model),
        ),
    )

    summary = run_hardware_session(
        config,
        checkpoint_path=checkpoint,
        prompt_path=prompt_path,
        manifest_path=manifest_path,
        deployment_certification_path=certification_path,
        source_candidate_checkpoint_path=checkpoint,
        source_training_report_path=certification_path,
        source_preflight_report_path=certification_path,
        source_mujoco_config_path=certification_path,
        source_mujoco_report_path=certification_path,
        source_manual_signoff_path=certification_path,
        policy_steps=1,
        enable_real_output=True,
        acknowledged_hardware_id="bench-a",
        clock=clock,
        sleeper=clock.sleep,
    )

    assert summary.servo_steps == summary.sent_actions == 5
    assert summary.shadow_steps == 0
    assert len(raw.fake.sent_actions) == 5
    assert raw.disconnect_calls == 1

    torch.manual_seed(5)
    substituted_candidate_model = CompactWAM(
        latent_dim=8,
        transformer_heads=2,
        future_steps=1,
        action_horizon=10,
        action_history_steps=1,
    )
    monkeypatch.setattr(
        "so101_wam.hardware_cli.validate_deployment_source_evidence",
        lambda *args, **kwargs: SimpleNamespace(
            source_evidence_sha256=certified_sources,
            candidate=SimpleNamespace(model=substituted_candidate_model),
        ),
    )
    monkeypatch.setattr(
        "so101_wam.hardware_cli.create_bi_so_follower",
        lambda config: pytest.fail(
            "robot must not be constructed for candidate/deployment weight mismatch"
        ),
    )
    with pytest.raises(DeploymentIssuanceError, match="candidate model weights"):
        run_hardware_session(
            config,
            checkpoint_path=checkpoint,
            prompt_path=prompt_path,
            manifest_path=manifest_path,
            deployment_certification_path=certification_path,
            source_candidate_checkpoint_path=checkpoint,
            source_training_report_path=certification_path,
            source_preflight_report_path=certification_path,
            source_mujoco_config_path=certification_path,
            source_mujoco_report_path=certification_path,
            source_manual_signoff_path=certification_path,
            policy_steps=1,
            enable_real_output=True,
            acknowledged_hardware_id="bench-a",
            clock=FakeClock(),
            sleeper=lambda _: None,
        )

    substituted_prompt, substituted_manifest = _save_prompt(
        tmp_path / "substituted",
        offset=7,
    )
    substituted_sources = dict(certified_sources)  # type: ignore[arg-type]
    substituted_sources["prompt_npz"] = file_sha256(substituted_prompt)
    substituted_sources["prompt_manifest"] = file_sha256(substituted_manifest)
    monkeypatch.setattr(
        "so101_wam.hardware_cli.validate_deployment_source_evidence",
        lambda *args, **kwargs: SimpleNamespace(
            source_evidence_sha256=substituted_sources,
            candidate=SimpleNamespace(model=deployment_model),
        ),
    )
    monkeypatch.setattr(
        "so101_wam.hardware_cli.create_bi_so_follower",
        lambda config: pytest.fail("robot must not be constructed for prompt substitution"),
    )
    with pytest.raises(DeploymentCertificationError, match="source evidence"):
        run_hardware_session(
            config,
            checkpoint_path=checkpoint,
            prompt_path=substituted_prompt,
            manifest_path=substituted_manifest,
            deployment_certification_path=certification_path,
            source_candidate_checkpoint_path=checkpoint,
            source_training_report_path=certification_path,
            source_preflight_report_path=certification_path,
            source_mujoco_config_path=certification_path,
            source_mujoco_report_path=certification_path,
            source_manual_signoff_path=certification_path,
            policy_steps=1,
            enable_real_output=True,
            acknowledged_hardware_id="bench-a",
            clock=clock,
            sleeper=clock.sleep,
        )


def test_promoted_candidate_weights_require_external_certification(tmp_path) -> None:
    checkpoint = tmp_path / "forged_deployment.pt"
    config = _config(actuation=True)
    core = _deployment_certification_core(
        config,
        checkpoint_id="forged-candidate",
        training_evidence_sha256="a" * 64,
    )
    save_compact_wam_checkpoint(
        CompactWAM(),
        checkpoint,
        metadata={
            "artifact_kind": "compact_wam_deployment",
            "offline_trained": False,
            "trained": True,
            "deployment_ready": True,
            "checkpoint_id": "forged-candidate",
            "training_evidence_sha256": "a" * 64,
            "deployment_evidence_sha256": deployment_evidence_sha256(core),
        },
    )

    with pytest.raises(HardwareCLIError, match="external deployment certification"):
        run_hardware_session(
            config,
            checkpoint_path=checkpoint,
            prompt_path=tmp_path / "unused.npz",
            policy_steps=1,
            enable_real_output=True,
            acknowledged_hardware_id="bench-a",
        )


def test_real_output_rejects_checkpoint_without_training_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_compact_wam_checkpoint(CompactWAM(), checkpoint)

    with pytest.raises(HardwareCLIError, match="artifact_kind=compact_wam_deployment"):
        # Artifact checks happen before robot construction or connection.
        run_hardware_session(
            _config(actuation=True),
            checkpoint_path=checkpoint,
            prompt_path=tmp_path / "unused.npz",
            policy_steps=1,
            enable_real_output=True,
            acknowledged_hardware_id="bench-a",
        )


@pytest.mark.parametrize("mode", list(ActionDecoder))
def test_offline_gate_prebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_records: tuple[tuple[EpisodeRecord, ...], tuple[EpisodeRecord, ...]],
    decoder_config,
    mode: ActionDecoder,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    report = tmp_path / "training.json"
    train_offline_candidate(
        *decoder_records,
        checkpoint_path=checkpoint,
        report_path=report,
        checkpoint_id="guard-mode",
        config=replace(decoder_config, action_decoder=mode),
    )
    calls: list[object] = []

    def fail_robot(config: object) -> object:
        calls.append(config)
        raise AssertionError("robot must not be constructed")

    monkeypatch.setattr("so101_wam.hardware_cli.create_bi_so_follower", fail_robot)

    with pytest.raises(HardwareCLIError, match="rejects offline"):
        run_hardware_session(
            _config(actuation=True),
            checkpoint_path=checkpoint,
            prompt_path=tmp_path / "unused.npz",
            policy_steps=1,
            enable_real_output=True,
            acknowledged_hardware_id="bench-a",
        )

    assert calls == []


def test_real_output_rejects_offline_candidate_even_if_trained_flag_is_forged() -> None:
    with pytest.raises(HardwareCLIError, match="rejects offline"):
        _require_deployment_checkpoint(
            {
                "trained": True,
                "deployment_ready": True,
                "offline_trained": True,
                "artifact_kind": "compact_wam_candidate",
                "evidence_level": "offline",
                "checkpoint_id": "candidate",
                "training_evidence_sha256": "b" * 64,
                "deployment_evidence_sha256": "c" * 64,
            }
        )


def test_real_output_requires_deployment_artifact_and_evidence() -> None:
    metadata: dict[str, str | int | float | bool | None] = {
        "artifact_kind": "compact_wam_deployment",
        "offline_trained": False,
        "trained": True,
        "deployment_ready": True,
        "checkpoint_id": "deployment-test-1",
        "training_evidence_sha256": "a" * 64,
    }

    with pytest.raises(HardwareCLIError, match="deployment_evidence_sha256"):
        _require_deployment_checkpoint(metadata)
