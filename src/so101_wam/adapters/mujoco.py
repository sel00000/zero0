"""MuJoCo dual-SO-101 adapter with two wrist cameras and collision rollback."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, ClassVar

import numpy as np

from so101_wam.config import MujocoConfig
from so101_wam.constants import (
    ACTION_DIM,
    ARM_JOINT_NAMES,
    ARM_SIDES,
    JOINT_KEYS,
    PRIMARY_CAMERA_KEYS,
)
from so101_wam.contracts import ActionChunk, ContractError, SensorimotorFrame
from so101_wam.mujoco_identity import MujocoModelIdentity


DEFAULT_MJCF_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "dual_so101_wam.xml"
)
MUJOCO_CAMERA_NAMES: tuple[str, str] = ("left_wrist_cam", "right_wrist_cam")
MUJOCO_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{side}_{joint}" for side in ARM_SIDES for joint in ARM_JOINT_NAMES
)
TASK_BLOCK_BODY_NAME = "task_block"
PROFILE_SCHEMA_VERSION = 1
_GRIPPER_INDICES = tuple(
    index for index, name in enumerate(MUJOCO_JOINT_NAMES) if name.endswith("_gripper")
)
_BODY_JOINT_INDICES = tuple(
    index for index in range(ACTION_DIM) if index not in _GRIPPER_INDICES
)


class MujocoAdapterError(RuntimeError):
    """Raised when the simulator cannot preserve its public adapter contract."""


class MujocoDependencyError(MujocoAdapterError):
    """Raised when the optional official MuJoCo package is unavailable."""


class MujocoActuationDisabledError(PermissionError):
    """Raised when simulated actuation was not explicitly enabled."""


class MujocoCollisionError(MujocoAdapterError):
    """Raised before committing a target that creates a forbidden contact."""

    def __init__(self, phase: str, contacts: tuple[CollisionContact, ...]) -> None:
        self.phase = phase
        self.contacts = contacts
        detail = "; ".join(contact.description for contact in contacts)
        super().__init__(f"forbidden MuJoCo contact during {phase}: {detail}")


@dataclass(frozen=True, slots=True)
class CollisionContact:
    geom1: str
    geom2: str
    body1: str
    body2: str
    category1: str
    category2: str
    distance: float

    @property
    def description(self) -> str:
        return (
            f"{self.category1}:{self.body1}/{self.geom1} <-> "
            f"{self.category2}:{self.body2}/{self.geom2} (dist={self.distance:.6g})"
        )


@dataclass(frozen=True, slots=True)
class MujocoContact:
    """One immutable MuJoCo contact, including its collision-gate decision."""

    geom1: str
    geom2: str
    body1: str
    body2: str
    category1: str
    category2: str
    distance: float
    forbidden: bool


@dataclass(frozen=True, slots=True)
class SO101MujocoCoordinates:
    """LeRobot-compatible degrees/percent <-> native MuJoCo radians."""

    native_lower: np.ndarray
    native_upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.array(self.native_lower, dtype=np.float64, copy=True)
        upper = np.array(self.native_upper, dtype=np.float64, copy=True)
        if lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,):
            raise MujocoAdapterError("MuJoCo coordinate limits must contain 12 values")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise MujocoAdapterError("MuJoCo coordinate limits contain NaN or infinity")
        if not np.all(lower < upper):
            raise MujocoAdapterError(
                "each MuJoCo coordinate lower limit must be below its upper limit"
            )
        lower.flags.writeable = False
        upper.flags.writeable = False
        object.__setattr__(self, "native_lower", lower)
        object.__setattr__(self, "native_upper", upper)

    @property
    def canonical_lower(self) -> np.ndarray:
        result = np.rad2deg(self.native_lower).astype(np.float64, copy=True)
        result[list(_GRIPPER_INDICES)] = 0.0
        result.flags.writeable = False
        return result

    @property
    def canonical_upper(self) -> np.ndarray:
        result = np.rad2deg(self.native_upper).astype(np.float64, copy=True)
        result[list(_GRIPPER_INDICES)] = 100.0
        result.flags.writeable = False
        return result

    def to_native(self, value: object, *, name: str = "joint target") -> np.ndarray:
        canonical = _joint_vector(value, name=name, dtype=np.float64)
        lower = self.canonical_lower
        upper = self.canonical_upper
        tolerance = 1e-6
        outside = np.flatnonzero(
            (canonical < lower - tolerance) | (canonical > upper + tolerance)
        )
        if outside.size:
            index = int(outside[0])
            raise ContractError(
                f"{name} is outside MuJoCo limits at {JOINT_KEYS[index]}: "
                f"{canonical[index]:.6g} not in [{lower[index]:.6g}, {upper[index]:.6g}]"
            )

        native = np.deg2rad(canonical)
        for index in _GRIPPER_INDICES:
            span = self.native_upper[index] - self.native_lower[index]
            native[index] = self.native_lower[index] + canonical[index] * span / 100.0
        return native

    def from_native(
        self, value: object, *, name: str = "MuJoCo joint position"
    ) -> np.ndarray:
        native = _joint_vector(value, name=name, dtype=np.float64)
        canonical = np.rad2deg(native)
        for index in _GRIPPER_INDICES:
            span = self.native_upper[index] - self.native_lower[index]
            canonical[index] = (native[index] - self.native_lower[index]) * 100.0 / span
        result = canonical.astype(np.float32)
        result.flags.writeable = False
        return result


@dataclass(frozen=True, slots=True)
class _ModelBindings:
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class _ObjectBinding:
    body_name: str
    body_id: int
    geom_id: int
    qpos_address: int


def _joint_vector(value: object, *, name: str, dtype: Any = np.float32) -> np.ndarray:
    if isinstance(value, ActionChunk):
        if value.horizon != 1:
            raise ContractError(
                "ActionChunk horizon must be 1; send one row per servo tick"
            )
        value = value.target_joint_position[0]
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != (ACTION_DIM,):
        raise ContractError(
            f"{name} must have shape ({ACTION_DIM},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains NaN or infinity")
    return array


def _position_tuple(value: object, *, name: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MujocoAdapterError(f"{name} must contain three numeric values") from error
    if array.shape != (3,):
        raise MujocoAdapterError(f"{name} must contain three values")
    if not np.isfinite(array).all():
        raise MujocoAdapterError(f"{name} must be finite")
    return (float(array[0]), float(array[1]), float(array[2]))


def _configure_gl_backend(backend: str) -> None:
    if "mujoco" not in sys.modules:
        os.environ.setdefault("MUJOCO_GL", backend)


def _import_mujoco(gl_backend: str) -> Any:
    _configure_gl_backend(gl_backend)
    try:
        return importlib.import_module("mujoco")
    except ImportError as error:
        raise MujocoDependencyError(
            "MuJoCo support requires the optional dependency; install with `pip install -e '.[sim]'`"
        ) from error


def _compiled_model_identity(mujoco: Any, model: Any) -> MujocoModelIdentity:
    size = int(mujoco.mj_sizeModel(model))
    if size < 1:
        raise MujocoAdapterError("MuJoCo compiled model is empty")
    buffer = np.empty(size, dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return MujocoModelIdentity(
        engine_version=str(mujoco.mj_versionString()),
        compiled_model_sha256=hashlib.sha256(buffer).hexdigest(),
        compiled_model_bytes=size,
    )


class MujocoBiSOAdapter:
    """Lifecycle-managed MuJoCo embodiment for the existing WAM runtime.

    Public joint values intentionally match LeRobot 0.6.1 defaults: the five
    body joints per arm are degrees and each gripper is normalized to 0..100.
    MuJoCo remains native-radian internally.
    """

    backend: ClassVar[str] = "mujoco"
    joint_keys: ClassVar[tuple[str, ...]] = JOINT_KEYS
    camera_keys: ClassVar[tuple[str, str]] = PRIMARY_CAMERA_KEYS

    def __init__(
        self,
        *,
        config: MujocoConfig | None = None,
        servo_hz: float = 50.0,
        actuation_enabled: bool = False,
        include_head_camera: bool = False,
        semantic_object_body: str | None = None,
        initial_object_position: tuple[float, float, float] | None = None,
        initial_task_block_position: tuple[float, float, float] | None = None,
    ) -> None:
        self.config = config or MujocoConfig()
        if not np.isfinite(servo_hz) or servo_hz <= 0:
            raise MujocoAdapterError("servo_hz must be finite and positive")
        self.servo_hz = float(servo_hz)
        self.actuation_enabled = bool(actuation_enabled)
        self.include_head_camera = bool(include_head_camera)
        if (
            initial_object_position is not None
            and initial_task_block_position is not None
        ):
            raise MujocoAdapterError(
                "initial_object_position and initial_task_block_position are mutually exclusive"
            )
        if (
            initial_task_block_position is not None
            and semantic_object_body not in {None, TASK_BLOCK_BODY_NAME}
        ):
            raise MujocoAdapterError(
                "initial_task_block_position requires "
                "semantic_object_body='task_block'"
            )
        self._semantic_object_body = semantic_object_body or (
            TASK_BLOCK_BODY_NAME if initial_task_block_position is not None else None
        )
        if initial_task_block_position is None:
            object_position = initial_object_position
        else:
            object_position = _position_tuple(
                initial_task_block_position,
                name="initial task block position",
            )

        self._initial_object_position = (
            None
            if object_position is None
            else _position_tuple(object_position, name="initial object position")
        )
        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._probe_data: Any | None = None
        self._renderer: Any | None = None
        self._bindings: _ModelBindings | None = None
        self._object_bindings: dict[str, _ObjectBinding] = {}
        self._selected_object_binding: _ObjectBinding | None = None
        self._coordinates: SO101MujocoCoordinates | None = None
        self._model_identity: MujocoModelIdentity | None = None
        self._physics_steps_per_servo_tick = 0
        self._physics_step_count = 0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._connected

    @property
    def physics_steps_per_servo_tick(self) -> int:
        return self._physics_steps_per_servo_tick

    @property
    def physics_step_count(self) -> int:
        return self._physics_step_count

    @property
    def model_time_s(self) -> float:
        self._require_connected()
        assert self._data is not None
        return float(self._data.time)

    @property
    def coordinates(self) -> SO101MujocoCoordinates:
        self._require_connected()
        assert self._coordinates is not None
        return self._coordinates

    def model_identity(self) -> dict[str, object]:
        if self._model_identity is None:
            raise MujocoAdapterError("MuJoCo model identity has not been captured")
        return dict(self._model_identity.to_payload())

    def task_block_position(self) -> tuple[float, float, float]:
        """Return the simulated task object's world position in metres."""

        return self.object_body_position(TASK_BLOCK_BODY_NAME)

    def object_body_position(self, body_name: str) -> tuple[float, float, float]:
        """Return a semantic object's world position in metres."""

        self._require_connected()
        assert self._data is not None
        binding = self._object_binding(body_name)
        value = self._data.xpos[binding.body_id]
        return _position_tuple(value, name=f"{body_name} position")

    def object_physical_profile(self, body_name: str) -> dict[str, object]:
        """Return a canonical name-free physical profile for a semantic object."""

        self._require_model_loaded()
        assert self._model is not None
        binding = self._object_binding(body_name)
        geom_type = self._geom_type_name(int(self._model.geom_type[binding.geom_id]))
        profile: dict[str, object] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "body": {
                "mass": _finite_float(self._model.body_mass[binding.body_id]),
                "inertia": _finite_list(self._model.body_inertia[binding.body_id]),
                "inertial_pos": _finite_list(self._model.body_ipos[binding.body_id]),
                "inertial_quat": _finite_list(self._model.body_iquat[binding.body_id]),
            },
            "geom": {
                "type": geom_type,
                "size": _finite_list(self._model.geom_size[binding.geom_id]),
                "pos": _finite_list(self._model.geom_pos[binding.geom_id]),
                "quat": _finite_list(self._model.geom_quat[binding.geom_id]),
                "friction": _finite_list(self._model.geom_friction[binding.geom_id]),
                "margin": _finite_float(self._model.geom_margin[binding.geom_id]),
                "gap": _finite_float(self._model.geom_gap[binding.geom_id]),
                "solref": _finite_list(self._model.geom_solref[binding.geom_id]),
                "solimp": _finite_list(self._model.geom_solimp[binding.geom_id]),
            },
        }
        _canonical_profile_bytes(profile)
        return profile

    def object_physical_profile_sha256(self, body_name: str) -> str:
        profile = self.object_physical_profile(body_name)
        return hashlib.sha256(_canonical_profile_bytes(profile)).hexdigest()

    def _model_path(self) -> Path:
        path = (
            Path(self.config.model_path).expanduser()
            if self.config.model_path
            else DEFAULT_MJCF_PATH
        )
        return path.resolve()

    def connect(self, *, calibrate: bool = True) -> None:
        del calibrate
        if self._connected:
            raise MujocoAdapterError("MuJoCo adapter is already connected")
        self._model_identity = None
        path = self._model_path()
        if not path.is_file():
            raise MujocoAdapterError(f"MuJoCo model does not exist: {path}")

        mujoco = _import_mujoco(self.config.gl_backend)
        renderer: Any | None = None
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
            model_identity = _compiled_model_identity(mujoco, model)
            data = mujoco.MjData(model)
            probe_data = mujoco.MjData(model)
            bindings, coordinates = self._bind_model(mujoco, model)
            object_bindings = self._bind_semantic_objects(mujoco, model)
            selected_object_binding = self._selected_object(
                object_bindings,
                self._semantic_object_body,
            )
            if self._initial_object_position is not None:
                selected_object_binding = self._require_selected_object(
                    selected_object_binding
                )
            if (
                self.include_head_camera
                and int(
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_CAMERA, "head_optional"
                    )
                )
                < 0
            ):
                raise MujocoAdapterError(
                    "MuJoCo model is missing optional camera 'head_optional'"
                )
            ratio = (1.0 / self.servo_hz) / float(model.opt.timestep)
            steps = int(round(ratio))
            if steps < 1 or not np.isclose(ratio, steps, rtol=0.0, atol=1e-9):
                raise MujocoAdapterError(
                    "servo period must be an integer multiple of MuJoCo timestep: "
                    f"servo_hz={self.servo_hz:g}, timestep={float(model.opt.timestep):g}"
                )

            self._mujoco = mujoco
            self._model = model
            self._data = data
            self._probe_data = probe_data
            self._bindings = bindings
            self._object_bindings = object_bindings
            self._selected_object_binding = selected_object_binding
            self._coordinates = coordinates
            self._physics_steps_per_servo_tick = steps
            self._reset_state()
            if self.config.forbid_collisions:
                contacts = self.forbidden_contacts()
                if contacts:
                    raise MujocoCollisionError("home reset", contacts)
            renderer = mujoco.Renderer(
                model,
                height=self.config.camera_height,
                width=self.config.camera_width,
            )
            self._renderer = renderer
            self._model_identity = model_identity
            self._connected = True
        except Exception:
            if renderer is not None:
                with suppress(Exception):
                    renderer.close()
            self._clear_runtime_objects()
            raise

    def disconnect(self) -> None:
        try:
            if self._renderer is not None:
                self._renderer.close()
        finally:
            self._clear_runtime_objects()

    def _clear_runtime_objects(self) -> None:
        self._renderer = None
        self._probe_data = None
        self._data = None
        self._model = None
        self._bindings = None
        self._object_bindings = {}
        self._selected_object_binding = None
        self._coordinates = None
        self._mujoco = None
        self._connected = False

    def _reset_state(self) -> None:
        assert self._mujoco is not None
        assert self._model is not None
        assert self._data is not None
        assert self._bindings is not None
        assert self._coordinates is not None
        self._mujoco.mj_resetData(self._model, self._data)
        native_home = self._coordinates.to_native(
            self.config.home_joint_position,
            name="mujoco.home_joint_position",
        )
        self._data.qpos[self._bindings.qpos_addresses] = native_home
        self._data.qvel[self._bindings.dof_addresses] = 0.0
        self._data.ctrl[self._bindings.actuator_ids] = native_home
        if self._initial_object_position is not None:
            assert self._selected_object_binding is not None
            address = self._selected_object_binding.qpos_address
            self._data.qpos[address : address + 3] = self._initial_object_position
        self._mujoco.mj_forward(self._model, self._data)
        self._physics_step_count = 0

    @staticmethod
    def _bind_model(
        mujoco: Any, model: Any
    ) -> tuple[_ModelBindings, SO101MujocoCoordinates]:
        joint_ids: list[int] = []
        qpos_addresses: list[int] = []
        dof_addresses: list[int] = []
        actuator_ids: list[int] = []
        native_lower: list[float] = []
        native_upper: list[float] = []

        for name in MUJOCO_JOINT_NAMES:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            actuator_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            )
            if joint_id < 0 or actuator_id < 0:
                raise MujocoAdapterError(
                    f"MuJoCo model is missing joint/actuator {name!r}"
                )
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise MujocoAdapterError(f"MuJoCo joint {name!r} must be a hinge")
            if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise MujocoAdapterError(
                    f"MuJoCo actuator {name!r} is not bound to its same-named joint"
                )
            if not bool(model.jnt_limited[joint_id]) or not bool(
                model.actuator_ctrllimited[actuator_id]
            ):
                raise MujocoAdapterError(
                    f"MuJoCo joint/actuator {name!r} must have hard ranges"
                )

            joint_range = model.jnt_range[joint_id]
            ctrl_range = model.actuator_ctrlrange[actuator_id]
            low = max(float(joint_range[0]), float(ctrl_range[0]))
            high = min(float(joint_range[1]), float(ctrl_range[1]))
            if low >= high:
                raise MujocoAdapterError(
                    f"MuJoCo joint/actuator {name!r} ranges do not overlap"
                )
            joint_ids.append(joint_id)
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            dof_addresses.append(int(model.jnt_dofadr[joint_id]))
            actuator_ids.append(actuator_id)
            native_lower.append(low)
            native_upper.append(high)

        for camera_name in MUJOCO_CAMERA_NAMES:
            if (
                int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
                < 0
            ):
                raise MujocoAdapterError(
                    f"MuJoCo model is missing camera {camera_name!r}"
                )

        bindings = _ModelBindings(
            joint_ids=np.asarray(joint_ids, dtype=np.int32),
            qpos_addresses=np.asarray(qpos_addresses, dtype=np.int32),
            dof_addresses=np.asarray(dof_addresses, dtype=np.int32),
            actuator_ids=np.asarray(actuator_ids, dtype=np.int32),
        )
        return bindings, SO101MujocoCoordinates(
            native_lower=np.asarray(native_lower, dtype=np.float64),
            native_upper=np.asarray(native_upper, dtype=np.float64),
        )

    @staticmethod
    def _bind_semantic_objects(
        mujoco: Any,
        model: Any,
    ) -> dict[str, _ObjectBinding]:
        bindings: dict[str, _ObjectBinding] = {}
        for body_id in range(int(model.nbody)):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None or not str(body_name).startswith("task_"):
                continue
            binding = MujocoBiSOAdapter._bind_semantic_object(
                mujoco,
                model,
                str(body_name),
            )
            bindings[binding.body_name] = binding
        return bindings

    @staticmethod
    def _bind_semantic_object(
        mujoco: Any,
        model: Any,
        body_name: str,
    ) -> _ObjectBinding:
        body_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
        )
        if body_id < 0:
            raise MujocoAdapterError(f"MuJoCo model is missing object body {body_name!r}")
        if int(model.body_jntnum[body_id]) != 1:
            raise MujocoAdapterError(
                f"MuJoCo object body {body_name!r} must have exactly one joint"
            )
        joint_id = int(model.body_jntadr[body_id])
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise MujocoAdapterError(
                f"MuJoCo object body {body_name!r} joint must be free"
            )
        if int(model.body_geomnum[body_id]) != 1:
            raise MujocoAdapterError(
                f"MuJoCo object body {body_name!r} must have exactly one direct geom"
            )
        return _ObjectBinding(
            body_name=body_name,
            body_id=body_id,
            geom_id=int(model.body_geomadr[body_id]),
            qpos_address=int(model.jnt_qposadr[joint_id]),
        )

    @staticmethod
    def _bind_task_block(
        mujoco: Any,
        model: Any,
    ) -> _ObjectBinding | None:
        body_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                TASK_BLOCK_BODY_NAME,
            )
        )
        if body_id < 0:
            return None
        return MujocoBiSOAdapter._bind_semantic_object(
            mujoco,
            model,
            TASK_BLOCK_BODY_NAME,
        )

    @staticmethod
    def _selected_object(
        bindings: dict[str, _ObjectBinding],
        body_name: str | None,
    ) -> _ObjectBinding | None:
        if body_name is None:
            return bindings.get(TASK_BLOCK_BODY_NAME)
        if body_name not in bindings:
            raise MujocoAdapterError(
                f"MuJoCo model is missing semantic object body {body_name!r}"
            )
        return bindings[body_name]

    @staticmethod
    def _require_selected_object(
        binding: _ObjectBinding | None,
    ) -> _ObjectBinding:
        if binding is None:
            raise MujocoAdapterError(
                "MuJoCo model is missing the selected semantic object free joint"
            )
        return binding

    @staticmethod
    def _require_task_block(
        binding: _ObjectBinding | None,
    ) -> _ObjectBinding:
        if binding is None:
            raise MujocoAdapterError(
                "MuJoCo model is missing the task block free joint"
            )
        return binding

    def _object_binding(self, body_name: str) -> _ObjectBinding:
        if body_name not in self._object_bindings:
            raise MujocoAdapterError(
                f"MuJoCo model is missing semantic object body {body_name!r}"
            )
        return self._object_bindings[body_name]

    def _require_connected(self) -> None:
        if not self._connected:
            raise MujocoAdapterError("MuJoCo adapter must be connected first")

    def _render(self, camera_name: str) -> np.ndarray:
        assert self._renderer is not None
        assert self._data is not None
        self._renderer.update_scene(self._data, camera=camera_name)
        image = np.asarray(self._renderer.render(), dtype=np.uint8)
        expected = (self.config.camera_height, self.config.camera_width, 3)
        if image.shape != expected:
            raise MujocoAdapterError(
                f"MuJoCo camera {camera_name!r} returned {image.shape}, expected {expected}"
            )
        return np.array(image, dtype=np.uint8, copy=True)

    def get_observation(self, *, timestamp_s: float) -> SensorimotorFrame:
        self._require_connected()
        assert self._model is not None
        assert self._data is not None
        assert self._bindings is not None
        assert self._coordinates is not None
        if self.config.forbid_collisions:
            contacts = self.forbidden_contacts()
            if contacts:
                raise MujocoCollisionError("observation", contacts)

        # Check physical limits before float32 conversion can round a breach away.
        native = _joint_vector(
            self._data.qpos[self._bindings.qpos_addresses],
            name="MuJoCo joint position", dtype=np.float64,
        )
        joint_ranges = self._model.jnt_range[self._bindings.joint_ids]
        outside = np.flatnonzero(
            (native < joint_ranges[:, 0]) | (native > joint_ranges[:, 1])
        )
        if outside.size:
            index = int(outside[0])
            raise MujocoAdapterError(
                f"MuJoCo observation is outside physical joint limits at {JOINT_KEYS[index]}"
            )

        images = {
            key: self._render(camera_name)
            for key, camera_name in zip(
                PRIMARY_CAMERA_KEYS, MUJOCO_CAMERA_NAMES, strict=True
            )
        }
        if self.include_head_camera:
            images["head_optional"] = self._render("head_optional")
        return SensorimotorFrame(
            timestamp_s=timestamp_s,
            images=images,
            joint_position=self._coordinates.from_native(native),
            image_timestamps_s={key: timestamp_s for key in images},
        )

    def send_action(self, action: ActionChunk | np.ndarray) -> np.ndarray:
        if not self.actuation_enabled:
            raise MujocoActuationDisabledError("simulated MuJoCo actuation is disabled")
        self._require_connected()
        assert self._mujoco is not None
        assert self._model is not None
        assert self._data is not None
        assert self._bindings is not None
        assert self._coordinates is not None

        canonical = _joint_vector(action, name="MuJoCo action")
        native = self._coordinates.to_native(canonical, name="MuJoCo action")
        if self.config.forbid_collisions:
            target_contacts = self._contacts_at_target(native)
            if target_contacts:
                raise MujocoCollisionError("target preflight", target_contacts)

        state = self._get_integration_state(self._data)
        self._data.ctrl[self._bindings.actuator_ids] = native
        for _ in range(self._physics_steps_per_servo_tick):
            self._mujoco.mj_step(self._model, self._data)
            if self.config.forbid_collisions:
                contacts = self.forbidden_contacts()
                if contacts:
                    self._set_integration_state(self._data, state)
                    raise MujocoCollisionError("servo transition", contacts)
        # Align rendered poses and contacts with the last integrated qpos.
        self._mujoco.mj_forward(self._model, self._data)
        if self.config.forbid_collisions:
            contacts = self.forbidden_contacts()
            if contacts:
                self._set_integration_state(self._data, state)
                raise MujocoCollisionError("servo transition", contacts)
        self._physics_step_count += self._physics_steps_per_servo_tick
        result = canonical.astype(np.float32, copy=True)
        result.flags.writeable = False
        return result

    def _contacts_at_target(self, native: np.ndarray) -> tuple[CollisionContact, ...]:
        assert self._mujoco is not None
        assert self._model is not None
        assert self._data is not None
        assert self._probe_data is not None
        assert self._bindings is not None
        state = self._get_integration_state(self._data)
        self._set_integration_state(self._probe_data, state)
        self._probe_data.qpos[self._bindings.qpos_addresses] = native
        self._probe_data.qvel[self._bindings.dof_addresses] = 0.0
        self._probe_data.ctrl[self._bindings.actuator_ids] = native
        self._mujoco.mj_forward(self._model, self._probe_data)
        return self._forbidden_contacts(self._probe_data)

    def _get_integration_state(self, data: Any) -> np.ndarray:
        assert self._mujoco is not None
        assert self._model is not None
        state_spec = self._mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(
            self._mujoco.mj_stateSize(self._model, state_spec), dtype=np.float64
        )
        self._mujoco.mj_getState(self._model, data, state, state_spec)
        return state

    def _set_integration_state(self, data: Any, state: np.ndarray) -> None:
        assert self._mujoco is not None
        assert self._model is not None
        state_spec = self._mujoco.mjtState.mjSTATE_INTEGRATION
        self._mujoco.mj_setState(self._model, data, state, state_spec)
        self._mujoco.mj_forward(self._model, data)

    def forbidden_contacts(self) -> tuple[CollisionContact, ...]:
        self._require_model_loaded()
        assert self._data is not None
        return self._forbidden_contacts(self._data)

    def contacts(self) -> tuple[MujocoContact, ...]:
        """Return all current contacts without exposing mutable MuJoCo state."""

        self._require_model_loaded()
        assert self._data is not None
        return self._contacts(self._data)

    def _require_model_loaded(self) -> None:
        if self._model is None or self._data is None or self._mujoco is None:
            raise MujocoAdapterError("MuJoCo model is not loaded")

    def _forbidden_contacts(self, data: Any) -> tuple[CollisionContact, ...]:
        return tuple(
            CollisionContact(
                geom1=contact.geom1,
                geom2=contact.geom2,
                body1=contact.body1,
                body2=contact.body2,
                category1=contact.category1,
                category2=contact.category2,
                distance=contact.distance,
            )
            for contact in self._contacts(data)
            if contact.forbidden
        )

    def _contacts(self, data: Any) -> tuple[MujocoContact, ...]:
        assert self._mujoco is not None
        assert self._model is not None
        contacts: list[MujocoContact] = []
        for index in range(int(data.ncon)):
            raw = data.contact[index]
            geom1_id = int(raw.geom1)
            geom2_id = int(raw.geom2)
            body1_id = int(self._model.geom_bodyid[geom1_id])
            body2_id = int(self._model.geom_bodyid[geom2_id])
            geom1 = self._object_name(self._mujoco.mjtObj.mjOBJ_GEOM, geom1_id, "geom")
            geom2 = self._object_name(self._mujoco.mjtObj.mjOBJ_GEOM, geom2_id, "geom")
            body1 = self._object_name(self._mujoco.mjtObj.mjOBJ_BODY, body1_id, "body")
            body2 = self._object_name(self._mujoco.mjtObj.mjOBJ_BODY, body2_id, "body")
            category1 = self._collision_category(body1, geom1)
            category2 = self._collision_category(body2, geom2)
            contacts.append(
                MujocoContact(
                    geom1=geom1,
                    geom2=geom2,
                    body1=body1,
                    body2=body2,
                    category1=category1,
                    category2=category2,
                    distance=float(raw.dist),
                    forbidden=self._is_forbidden_pair(
                        category1,
                        category2,
                        body1,
                        body2,
                    ),
                )
            )
        return tuple(contacts)

    def _object_name(self, object_type: Any, object_id: int, fallback: str) -> str:
        assert self._mujoco is not None
        assert self._model is not None
        name = self._mujoco.mj_id2name(self._model, object_type, object_id)
        return str(name) if name is not None else f"{fallback}#{object_id}"

    @staticmethod
    def _collision_category(body: str, geom: str) -> str:
        if body.startswith("left_"):
            return "left_arm"
        if body.startswith("right_"):
            return "right_arm"
        if body == "torso":
            return "torso"
        if body == "table":
            return "table"
        if geom == "floor_protected":
            return "floor"
        if body.startswith("task_"):
            return "object"
        return "environment"

    @staticmethod
    def _is_forbidden_pair(
        category1: str, category2: str, body1: str, body2: str
    ) -> bool:
        categories = {category1, category2}
        if categories == {"left_arm", "right_arm"}:
            return True
        protected = {"torso", "table", "floor"}
        if categories & {"left_arm", "right_arm"} and categories & protected:
            return True
        if category1 == category2 and category1 in {"left_arm", "right_arm"}:
            if body1 == body2:
                return False
            side = "left" if category1 == "left_arm" else "right"
            return not (
                MujocoBiSOAdapter._is_gripper_internal_body(body1, side)
                and MujocoBiSOAdapter._is_gripper_internal_body(body2, side)
            )
        return False

    @staticmethod
    def _is_gripper_internal_body(body: str, side: str) -> bool:
        return body in {
            f"{side}_gripper",
            f"{side}_camera_mount",
            f"{side}_moving_jaw_so101_v1",
        }

    def _geom_type_name(self, geom_type: int) -> str:
        assert self._mujoco is not None
        for name in dir(self._mujoco.mjtGeom):
            if not name.startswith("mjGEOM_"):
                continue
            if int(getattr(self._mujoco.mjtGeom, name)) == geom_type:
                return name.removeprefix("mjGEOM_").lower()
        raise MujocoAdapterError(f"unsupported MuJoCo geom type {geom_type}")


def _finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise MujocoAdapterError("MuJoCo object physical profile is not finite")
    return result


def _finite_list(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise MujocoAdapterError("MuJoCo object physical profile is not finite")
    return [float(item) for item in array.tolist()]


def _canonical_profile_bytes(profile: dict[str, object]) -> bytes:
    try:
        return json.dumps(
            profile,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except ValueError as error:
        raise MujocoAdapterError(
            "MuJoCo object physical profile is not strict finite JSON"
        ) from error


__all__ = [
    "CollisionContact",
    "DEFAULT_MJCF_PATH",
    "MUJOCO_CAMERA_NAMES",
    "MUJOCO_JOINT_NAMES",
    "MujocoActuationDisabledError",
    "MujocoAdapterError",
    "MujocoBiSOAdapter",
    "MujocoCollisionError",
    "MujocoContact",
    "MujocoDependencyError",
    "SO101MujocoCoordinates",
]
