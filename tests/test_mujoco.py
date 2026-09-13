from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import so101_wam.adapters.mujoco as mujoco_adapter_mod
from so101_wam.adapters.mujoco import (
    CollisionContact,
    MUJOCO_CAMERA_NAMES,
    MUJOCO_JOINT_NAMES,
    MujocoAdapterError,
    MujocoBiSOAdapter,
    MujocoCollisionError,
    MujocoContact,
)
from so101_wam.config import ProjectConfig
from so101_wam.constants import ACTION_DIM, JOINT_KEYS, PRIMARY_CAMERA_KEYS
from so101_wam.contracts import ActionChunk
from so101_wam.mujoco_cli import (
    MujocoCLIError,
    _MujocoRolloutObserver,
    _checkpoint_identity,
    _result,
    main as mujoco_main,
    run_mujoco_smoke,
    write_mujoco_report,
)
from so101_wam.mujoco_identity import (
    MUJOCO_IDENTITY_SCHEMA_VERSION,
    MujocoIdentityError,
    MujocoModelIdentity,
)
from so101_wam.rollout import ExecutabilityTrace


ROOT = Path(__file__).resolve().parents[1]
MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
requires_mujoco = pytest.mark.skipif(
    not MUJOCO_AVAILABLE, reason="optional mujoco dependency is absent"
)


def _config() -> ProjectConfig:
    return ProjectConfig.load(ROOT / "configs" / "mujoco.toml")


@requires_mujoco
def test_servo_observation_sync() -> None:
    config = ProjectConfig.load(ROOT / "configs" / "mujoco_robot_free.toml")
    adapter = MujocoBiSOAdapter(config=config.mujoco, actuation_enabled=True)
    adapter.connect()
    try:
        adapter.send_action(np.asarray(config.mujoco.home_joint_position))
        binding = adapter._object_bindings["task_block"]
        assert adapter._data is not None
        integrated = adapter._data.qpos[binding.qpos_address : binding.qpos_address + 3]
        np.testing.assert_allclose(
            adapter.object_body_position("task_block"), integrated, rtol=0.0, atol=1e-12
        )
        assert adapter.physics_step_count == 4
    finally:
        adapter.disconnect()


@dataclass(frozen=True)
class _Summary:
    servo_steps: int = 2
    sent_actions: int = 2
    final_state: str = "rollout_ready"


class _Adapter:
    physics_steps_per_servo_tick = 4
    physics_step_count = 8

    def model_identity(self) -> dict[str, object]:
        return MujocoModelIdentity(
            engine_version="3.12.0",
            compiled_model_sha256="a" * 64,
            compiled_model_bytes=128,
        ).to_payload()


def _minimal_mujoco_xml(
    *,
    actuator_kp: float = 1.0,
    camera_y: float = -1.0,
    link_length: float = 0.1,
) -> str:
    return f"""
<mujoco model="identity">
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="joint" type="hinge" axis="0 0 1"/>
      <geom name="link" type="capsule" size="0.01 {link_length}"/>
      <camera name="wrist" pos="0 {camera_y} 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="joint" joint="joint" kp="{actuator_kp}"/>
  </actuator>
</mujoco>
"""


def test_mujoco_config_is_sim_only_and_uses_the_two_wrist_contract() -> None:
    config = _config()

    assert config.runtime.backend == "mujoco"
    assert config.runtime.actuation_enabled is True
    assert config.runtime.primary_cameras == PRIMARY_CAMERA_KEYS
    assert config.runtime.use_head_camera is False
    assert config.mujoco.model_path == ""
    assert config.mujoco.forbid_collisions is True
    assert config.safety.calibrated is False
    assert len(config.mujoco.home_joint_position) == ACTION_DIM


def test_mujoco_g8_result_fields_bind_checkpoint_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate-bytes")

    checkpoint_sha256, checkpoint_id = _checkpoint_identity(
        checkpoint,
        {"checkpoint_id": " candidate-test "},
    )
    result = _result(
        config=_config(),
        adapter=_Adapter(),  # type: ignore[arg-type]
        summary=_Summary(),
        policy_kind="compact_wam",
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
    )

    assert result["schema_version"] == 1
    assert result["gate"] == "G8-simulation"
    assert result["result"] == "pass"
    assert result["policy"] == "compact_wam"
    assert result["checkpoint_sha256"] == checkpoint_sha256
    assert result["checkpoint_id"] == "candidate-test"


def test_mujoco_observer_attaches_post_servo_contacts() -> None:
    trace = ExecutabilityTrace(
        "observer:seed:7",
        joint_lower=(-10.0,) * ACTION_DIM,
        joint_upper=(10.0,) * ACTION_DIM,
    )
    action = ActionChunk(
        target_joint_position=np.zeros((1, ACTION_DIM), dtype=np.float32),
        dt_s=0.02,
        created_at_s=1.0,
    )
    contact = MujocoContact(
        geom1="left_gripper",
        geom2="task_block",
        body1="left_gripper",
        body2="task_block",
        category1="left_arm",
        category2="object",
        distance=-0.001,
        forbidden=False,
    )
    adapter = SimpleNamespace(contacts=lambda: (contact,))
    observer = _MujocoRolloutObserver(trace, adapter)  # type: ignore[arg-type]
    observer.on_policy_step(
        policy_index=0,
        step=SimpleNamespace(action=action),
    )
    observer.on_servo_step(
        policy_index=0,
        servo_index=0,
        step=SimpleNamespace(
            action=action,
            observation=SimpleNamespace(
                joint_position=np.zeros(ACTION_DIM, dtype=np.float32)
            ),
            executed_action=np.zeros(ACTION_DIM, dtype=np.float32),
            sent=True,
            shadow=False,
            safety=SimpleNamespace(accepted=True, clipped=False, reasons=()),
        ),
    )

    payload = trace.snapshot()

    assert payload["contact_progression_count"] == 1
    assert payload["contact_observation_count"] == 1
    assert payload["object_contact_observation_count"] == 1
    assert payload["forbidden_contact_observation_count"] == 0


def test_mujoco_smoke_g8_result_has_null_checkpoint_identity() -> None:
    result = _result(
        config=_config(),
        adapter=_Adapter(),  # type: ignore[arg-type]
        summary=_Summary(),
        policy_kind="wrist_roll_smoke",
    )

    assert result["schema_version"] == 1
    assert result["gate"] == "G8-simulation"
    assert result["result"] == "pass"
    assert result["checkpoint_sha256"] is None
    assert result["checkpoint_id"] is None


def test_checkpoint_identity_requires_candidate_id(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate-bytes")

    with pytest.raises(MujocoCLIError, match="checkpoint_id"):
        _checkpoint_identity(checkpoint, {})


def test_mujoco_report_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "gate": "G8-simulation",
        "result": "pass",
        "checkpoint_sha256": None,
        "checkpoint_id": None,
    }
    report_path = tmp_path / "g8.json"

    written = write_mujoco_report(report_path, report)

    assert written == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    with pytest.raises(MujocoCLIError, match="already exists"):
        write_mujoco_report(report_path, report)


def test_model_identity_payload() -> None:
    identity = MujocoModelIdentity(
        engine_version="3.12.0",
        compiled_model_sha256="a" * 64,
        compiled_model_bytes=128,
    )

    payload = identity.to_payload()

    assert payload == {
        "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
        "engine_version": "3.12.0",
        "compiled_model_sha256": "a" * 64,
        "compiled_model_bytes": 128,
    }
    assert MujocoModelIdentity.from_payload(payload) == identity


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": "",
            "compiled_model_sha256": "a" * 64,
            "compiled_model_bytes": 128,
        },
        {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": "   ",
            "compiled_model_sha256": "a" * 64,
            "compiled_model_bytes": 128,
        },
        {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": "3.12.0",
            "compiled_model_sha256": "A" * 64,
            "compiled_model_bytes": 128,
        },
        {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": "3.12.0",
            "compiled_model_sha256": "g" * 64,
            "compiled_model_bytes": 128,
        },
        {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": "3.12.0",
            "compiled_model_sha256": "a" * 64,
            "compiled_model_bytes": True,
        },
        {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": "3.12.0",
            "compiled_model_sha256": "a" * 64,
            "compiled_model_bytes": 0,
        },
        {
            "schema_version": "wrong",
            "engine_version": "3.12.0",
            "compiled_model_sha256": "a" * 64,
            "compiled_model_bytes": 128,
        },
        {
            "schema_version": MUJOCO_IDENTITY_SCHEMA_VERSION,
            "engine_version": "3.12.0",
            "compiled_model_sha256": "a" * 64,
            "compiled_model_bytes": 128,
            "extra": "field",
        },
    ],
)
def test_model_identity_invalid(
    payload: dict[str, object],
) -> None:
    with pytest.raises(MujocoIdentityError):
        MujocoModelIdentity.from_payload(payload)


@requires_mujoco
def test_compiled_identity_drift() -> None:
    mujoco = importlib.import_module("mujoco")
    base = mujoco.MjModel.from_xml_string(_minimal_mujoco_xml())
    changed_actuator = mujoco.MjModel.from_xml_string(
        _minimal_mujoco_xml(actuator_kp=2.0)
    )
    changed_robot = mujoco.MjModel.from_xml_string(
        _minimal_mujoco_xml(link_length=0.2)
    )
    changed_camera = mujoco.MjModel.from_xml_string(
        _minimal_mujoco_xml(camera_y=-1.1)
    )
    repeated = mujoco.MjModel.from_xml_string(_minimal_mujoco_xml())

    base_identity = mujoco_adapter_mod._compiled_model_identity(mujoco, base)

    assert MujocoModelIdentity.from_payload(base_identity.to_payload()) == base_identity
    assert base_identity.compiled_model_bytes > 0
    assert base_identity == mujoco_adapter_mod._compiled_model_identity(
        mujoco,
        repeated,
    )
    assert base_identity.compiled_model_sha256 != (
        mujoco_adapter_mod._compiled_model_identity(
            mujoco,
            changed_actuator,
        ).compiled_model_sha256
    )
    assert base_identity.compiled_model_sha256 != (
        mujoco_adapter_mod._compiled_model_identity(
            mujoco,
            changed_robot,
        ).compiled_model_sha256
    )
    assert base_identity.compiled_model_sha256 != (
        mujoco_adapter_mod._compiled_model_identity(
            mujoco,
            changed_camera,
        ).compiled_model_sha256
    )


@requires_mujoco
def test_adapter_model_identity(
    tmp_path: Path,
) -> None:
    config = _config()
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
    )
    with pytest.raises(MujocoAdapterError, match="model identity"):
        adapter.model_identity()

    adapter.connect(calibrate=False)
    first = adapter.model_identity()
    try:
        adapter.send_action(np.zeros(ACTION_DIM, dtype=np.float32))
        second = adapter.model_identity()
    finally:
        adapter.disconnect()
    after_disconnect = adapter.model_identity()

    assert MujocoModelIdentity.from_payload(first).to_payload() == first
    assert second == first
    assert after_disconnect == first
    first["engine_version"] = "mutated"
    assert adapter.model_identity()["engine_version"] != "mutated"

    adapter.config = replace(config.mujoco, model_path=str(tmp_path / "missing.xml"))
    with pytest.raises(MujocoAdapterError, match="does not exist"):
        adapter.connect(calibrate=False)
    with pytest.raises(MujocoAdapterError, match="model identity"):
        adapter.model_identity()


def test_mujoco_cli_writes_report_and_keeps_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "g8.json"
    result = {
        "schema_version": 1,
        "gate": "G8-simulation",
        "result": "pass",
        "policy": "wrist_roll_smoke",
        "checkpoint_sha256": None,
        "checkpoint_id": None,
    }

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        return result

    monkeypatch.setattr("so101_wam.mujoco_cli.run_mujoco_smoke", fake_run)
    monkeypatch.chdir(tmp_path)

    assert (
        mujoco_main(
            [
                "--steps",
                "1",
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == result
    assert json.loads(capsys.readouterr().out) == result


def test_mujoco_cli_report_fails_before_running_when_target_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "g8.json"
    report_path.write_text("{}", encoding="utf-8")

    def unexpected_run(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("simulation should not start when report exists")

    monkeypatch.setattr("so101_wam.mujoco_cli.run_mujoco_smoke", unexpected_run)

    with pytest.raises(SystemExit) as captured:
        mujoco_main(
            [
                "--config",
                str(ROOT / "configs" / "mujoco.toml"),
                "--steps",
                "1",
                "--report",
                str(report_path),
            ]
        )

    assert captured.value.code == 2


def test_mujoco_adapter_module_does_not_import_optional_dependency() -> None:
    sys.modules.pop("mujoco", None)
    module = importlib.import_module("so101_wam.adapters.mujoco")

    assert module.MujocoBiSOAdapter.backend == "mujoco"
    assert "mujoco" not in sys.modules


def test_task_block_position_rejects_non_finite_initial_state() -> None:
    config = _config()

    with pytest.raises(MujocoAdapterError, match="task block position"):
        MujocoBiSOAdapter(
            config=config.mujoco,
            servo_hz=config.runtime.servo_hz,
            initial_task_block_position=(0.34, 0.0, float("nan")),
        )


def test_legacy_task_block_position_rejects_other_object_body() -> None:
    config = _config()

    with pytest.raises(
        MujocoAdapterError,
        match="initial_task_block_position requires semantic_object_body='task_block'",
    ):
        MujocoBiSOAdapter(
            config=config.mujoco,
            servo_hz=config.runtime.servo_hz,
            semantic_object_body="task_cylinder",
            initial_task_block_position=(0.34, 0.0, 0.475),
        )


def test_task_block_is_optional_for_nonsemantic_custom_scene() -> None:
    class ObjectKinds:
        mjOBJ_BODY = 1
        mjOBJ_JOINT = 2

    class MujocoStub:
        mjtObj = ObjectKinds

        @staticmethod
        def mj_name2id(model: object, kind: int, name: str) -> int:
            del model, kind, name
            return -1

    assert (
        MujocoBiSOAdapter._bind_task_block(
            MujocoStub,
            object(),
        )
        is None
    )

    with pytest.raises(MujocoAdapterError, match="missing the task block"):
        MujocoBiSOAdapter._require_task_block(None)


@requires_mujoco
def test_task_block_position_is_seedable_through_adapter_boundary() -> None:
    config = _config()
    initial_position = (0.32, -0.02, 0.475)
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
        initial_task_block_position=initial_position,
    )
    adapter.connect(calibrate=False)
    try:
        measured = adapter.task_block_position()
    finally:
        adapter.disconnect()

    assert isinstance(measured, tuple)
    assert measured == pytest.approx(initial_position)


@requires_mujoco
def test_semantic_objects_are_seedable_and_profiled_by_body_name() -> None:
    config = _config()
    block_position = (0.32, -0.02, 0.475)
    cylinder_position = (0.36, 0.09, 0.485)

    block_adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
        semantic_object_body="task_block",
        initial_object_position=block_position,
    )
    cylinder_adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
        semantic_object_body="task_cylinder",
        initial_object_position=cylinder_position,
    )

    block_adapter.connect(calibrate=False)
    cylinder_adapter.connect(calibrate=False)
    try:
        block_profile = block_adapter.object_physical_profile("task_block")
        cylinder_profile = cylinder_adapter.object_physical_profile("task_cylinder")

        assert block_adapter.object_body_position("task_block") == pytest.approx(
            block_position
        )
        assert block_adapter.task_block_position() == pytest.approx(block_position)
        assert cylinder_adapter.object_body_position("task_cylinder") == pytest.approx(
            cylinder_position
        )
        assert block_profile["schema_version"] == 1
        assert cylinder_profile["schema_version"] == 1
        assert block_profile["geom"]["type"] == "box"
        assert cylinder_profile["geom"]["type"] == "cylinder"
        assert "task_block" not in json.dumps(block_profile, sort_keys=True)
        assert "task_cylinder" not in json.dumps(cylinder_profile, sort_keys=True)
        assert block_adapter.object_physical_profile_sha256(
            "task_block"
        ) != cylinder_adapter.object_physical_profile_sha256("task_cylinder")
    finally:
        block_adapter.disconnect()
        cylinder_adapter.disconnect()


@requires_mujoco
def test_official_dual_scene_renders_exact_two_wrist_observations() -> None:
    config = _config()
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=80, camera_height=60),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
    )
    adapter.connect(calibrate=False)
    try:
        observation = adapter.get_observation(timestamp_s=1.0)

        assert adapter.is_connected is True
        assert adapter.is_calibrated is True
        assert adapter.joint_keys == JOINT_KEYS
        assert adapter.camera_keys == PRIMARY_CAMERA_KEYS
        assert len(MUJOCO_JOINT_NAMES) == ACTION_DIM
        assert MUJOCO_CAMERA_NAMES == ("left_wrist_cam", "right_wrist_cam")
        assert tuple(observation.images) == PRIMARY_CAMERA_KEYS
        assert observation.joint_position.shape == (ACTION_DIM,)
        np.testing.assert_allclose(
            observation.joint_position, np.zeros(ACTION_DIM), atol=1e-6
        )
        for image in observation.images.values():
            assert image.shape == (60, 80, 3)
            assert image.dtype == np.uint8
            assert int(image.max()) > int(image.min())
        assert not np.array_equal(
            observation.images["left_wrist"],
            observation.images["right_wrist"],
        )
        assert adapter.forbidden_contacts() == ()
    finally:
        adapter.disconnect()


@requires_mujoco
def test_mujoco_adapter_exposes_all_contact_evidence() -> None:
    config = _config()
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
    )
    adapter.connect(calibrate=False)
    try:
        adapter.send_action(np.zeros(ACTION_DIM, dtype=np.float32))
        adapter.send_action(np.zeros(ACTION_DIM, dtype=np.float32))
        contacts = adapter.contacts()

        assert contacts
        assert any(
            "object" in {contact.category1, contact.category2}
            for contact in contacts
        )
        assert all(contact.forbidden is False for contact in contacts)
        assert adapter.forbidden_contacts() == ()
    finally:
        adapter.disconnect()


@requires_mujoco
def test_optional_head_camera_is_diagnostic_only() -> None:
    config = _config()
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=False,
        include_head_camera=True,
    )
    adapter.connect(calibrate=False)
    try:
        observation = adapter.get_observation(timestamp_s=1.0)
    finally:
        adapter.disconnect()

    assert tuple(observation.images) == (*PRIMARY_CAMERA_KEYS, "head_optional")
    assert observation.primary_images[0] is observation.images["left_wrist"]
    assert observation.primary_images[1] is observation.images["right_wrist"]


@requires_mujoco
def test_canonical_units_round_trip_and_servo_tick_steps_physics_four_times() -> None:
    config = _config()
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=50.0,
        actuation_enabled=True,
    )
    adapter.connect(calibrate=False)
    try:
        canonical = np.array(
            [10, -20, 30, -40, 50, 75, -10, 20, -30, 40, -50, 25],
            dtype=np.float32,
        )
        native = adapter.coordinates.to_native(canonical)
        round_trip = adapter.coordinates.from_native(native)
        np.testing.assert_allclose(round_trip, canonical, atol=1e-5)

        safe_target = np.zeros(ACTION_DIM, dtype=np.float32)
        safe_target[4] = 1.0
        safe_target[10] = -1.0
        returned = adapter.send_action(safe_target)

        np.testing.assert_allclose(returned, safe_target)
        assert adapter.physics_steps_per_servo_tick == 4
        assert adapter.physics_step_count == 4
        assert adapter.model_time_s == pytest.approx(0.02)
    finally:
        adapter.disconnect()


@pytest.mark.parametrize(
    ("category1", "category2", "body1", "body2", "expected"),
    [
        ("left_arm", "right_arm", "left_wrist", "right_wrist", True),
        ("left_arm", "torso", "left_wrist", "torso", True),
        ("right_arm", "table", "right_gripper", "table", True),
        ("left_arm", "floor", "left_gripper", "world", True),
        ("left_arm", "object", "left_gripper", "task_block", False),
        ("left_arm", "left_arm", "left_shoulder", "left_wrist", True),
        ("left_arm", "left_arm", "left_gripper", "left_moving_jaw_so101_v1", False),
    ],
)
def test_collision_pair_policy(
    category1: str,
    category2: str,
    body1: str,
    body2: str,
    expected: bool,
) -> None:
    assert (
        MujocoBiSOAdapter._is_forbidden_pair(category1, category2, body1, body2)
        is expected
    )


@requires_mujoco
def test_collision_preflight_rejects_protected_contacts_without_advancing_state() -> (
    None
):
    config = _config()
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
    )
    adapter.connect(calibrate=False)
    targets = (
        (
            np.array(
                [
                    82.7486,
                    -8.6370,
                    -74.3854,
                    90.6090,
                    -13.8658,
                    48.5286,
                    -92.7842,
                    61.7681,
                    -67.3935,
                    -71.2582,
                    -70.2933,
                    42.8293,
                ],
                dtype=np.float32,
            ),
            {"left_arm", "right_arm"},
        ),
        (
            np.array(
                [
                    -33.9852,
                    -92.7291,
                    -12.5859,
                    -62.4475,
                    -23.0395,
                    60.3039,
                    -61.3609,
                    68.3282,
                    37.3018,
                    -5.5774,
                    -31.5411,
                    50.4152,
                ],
                dtype=np.float32,
            ),
            {"right_arm", "torso"},
        ),
    )
    try:
        for colliding_target, expected_categories in targets:
            with pytest.raises(
                MujocoCollisionError, match="target preflight"
            ) as captured:
                adapter.send_action(colliding_target)

            assert captured.value.contacts
            assert any(
                {contact.category1, contact.category2} == expected_categories
                for contact in captured.value.contacts
            )
        assert adapter.physics_step_count == 0
        assert adapter.model_time_s == pytest.approx(0.0)
        observation = adapter.get_observation(timestamp_s=1.0)
        np.testing.assert_allclose(
            observation.joint_position, np.zeros(ACTION_DIM), atol=1e-6
        )
    finally:
        adapter.disconnect()


@requires_mujoco
def test_transition_collision_restores_the_complete_integration_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    adapter = MujocoBiSOAdapter(
        config=replace(config.mujoco, camera_width=64, camera_height=48),
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
    )
    adapter.connect(calibrate=False)
    try:
        assert adapter._data is not None
        before_qpos = adapter._data.qpos.copy()
        before_qvel = adapter._data.qvel.copy()
        before_ctrl = adapter._data.ctrl.copy()
        before_warmstart = adapter._data.qacc_warmstart.copy()
        before_time = adapter.model_time_s
        before_steps = adapter.physics_step_count
        injected_contact = CollisionContact(
            geom1="left_injected",
            geom2="right_injected",
            body1="left_wrist",
            body2="right_wrist",
            category1="left_arm",
            category2="right_arm",
            distance=-1e-4,
        )
        checks = 0

        def inject_transition_contact() -> tuple[CollisionContact, ...]:
            nonlocal checks
            checks += 1
            return (injected_contact,)

        monkeypatch.setattr(adapter, "forbidden_contacts", inject_transition_contact)
        safe_target = np.zeros(ACTION_DIM, dtype=np.float32)
        safe_target[4] = 1.0
        safe_target[10] = -1.0

        with pytest.raises(MujocoCollisionError, match="servo transition"):
            adapter.send_action(safe_target)

        assert checks == 1
        np.testing.assert_array_equal(adapter._data.qpos, before_qpos)
        np.testing.assert_array_equal(adapter._data.qvel, before_qvel)
        np.testing.assert_array_equal(adapter._data.ctrl, before_ctrl)
        np.testing.assert_array_equal(adapter._data.qacc_warmstart, before_warmstart)
        assert adapter.model_time_s == pytest.approx(before_time)
        assert adapter.physics_step_count == before_steps
    finally:
        adapter.disconnect()


@requires_mujoco
def test_headless_smoke_runs_existing_10hz_50hz_runtime_contract() -> None:
    config = _config()
    config = replace(
        config,
        runtime=replace(config.runtime, action_horizon=2),
        mujoco=replace(config.mujoco, camera_width=64, camera_height=48),
    )

    result = run_mujoco_smoke(config, policy_steps=1)

    assert result["mode"] == "mujoco"
    assert result["evidence_level"] == "simulation"
    assert result["physics_steps_per_servo_tick"] == 4
    assert result["physics_steps"] == 8
    assert result["rollout"]["servo_steps"] == 2
    assert result["rollout"]["sent_actions"] == 2
    assert result["rollout"]["final_state"] == "rollout_ready"
    assert MujocoModelIdentity.from_payload(result["mujoco_model_identity"])
