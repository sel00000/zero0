from __future__ import annotations

import numpy as np
import pytest

from so101_wam.adapters.lerobot import LeRobotBiSOLeader
from so101_wam.constants import ACTION_DIM, JOINT_KEYS
from so101_wam.contracts import ContractError


class _Arm:
    def __init__(self) -> None:
        self.is_connected = False
        self.disconnect_calls = 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class _Leader:
    def __init__(self) -> None:
        self.left_arm = _Arm()
        self.right_arm = _Arm()
        self.is_calibrated = True
        self.connect_calibrate: bool | None = None
        self.disconnect_calls = 0
        self.action = {key: float(index) for index, key in enumerate(JOINT_KEYS)}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, *, calibrate: bool = True) -> None:
        self.connect_calibrate = calibrate
        self.left_arm.is_connected = True
        self.right_arm.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def get_action(self) -> dict[str, float]:
        return self.action


def test_leader_reads_action() -> None:
    leader = _Leader()
    adapter = LeRobotBiSOLeader(leader)

    adapter.connect()
    action = adapter.get_action()

    assert leader.connect_calibrate is False
    assert action.dtype == np.float32
    assert action.shape == (ACTION_DIM,)
    np.testing.assert_allclose(action, np.arange(ACTION_DIM, dtype=np.float32))


def test_leader_rejects_keys() -> None:
    leader = _Leader()
    leader.action = dict(leader.action)
    leader.action["extra.pos"] = 1.0
    adapter = LeRobotBiSOLeader(leader)
    adapter.connect()

    with pytest.raises(ContractError, match="keys must be exact"):
        adapter.get_action()


def test_leader_rejects_nan() -> None:
    leader = _Leader()
    leader.action = {key: 0.0 for key in JOINT_KEYS}
    leader.action[JOINT_KEYS[0]] = float("nan")
    adapter = LeRobotBiSOLeader(leader)
    adapter.connect()

    with pytest.raises(ContractError, match="NaN or infinity"):
        adapter.get_action()


def test_leader_cleanup() -> None:
    class PartialLeader(_Leader):
        def connect(self, *, calibrate: bool = True) -> None:
            self.connect_calibrate = calibrate
            self.left_arm.is_connected = True
            raise RuntimeError("right leader failed")

        @property
        def is_connected(self) -> bool:
            return self.left_arm.is_connected and self.right_arm.is_connected

        def disconnect(self) -> None:
            self.disconnect_calls += 1
            if not self.is_connected:
                raise RuntimeError("not fully connected")

    leader = PartialLeader()
    adapter = LeRobotBiSOLeader(leader)

    with pytest.raises(RuntimeError, match="right leader failed"):
        adapter.connect()

    assert leader.connect_calibrate is False
    assert leader.disconnect_calls == 1
    assert leader.left_arm.disconnect_calls == 1
    assert leader.left_arm.is_connected is False


def test_uncalibrated_leader() -> None:
    leader = _Leader()
    leader.is_calibrated = False
    adapter = LeRobotBiSOLeader(leader)

    with pytest.raises(RuntimeError, match="is_calibrated=true"):
        adapter.connect()

    assert leader.disconnect_calls == 1
