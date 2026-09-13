from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_wam.config import LeRobotConfig, ProjectConfig, RuntimeConfig, SafetyConfig
from so101_wam.constants import ACTION_DIM, ARM_JOINT_NAMES
from so101_wam.hardware_preflight import (
    HardwarePreflightError,
    HardwarePreflightReportError,
    main,
    run_hardware_preflight,
    write_hardware_preflight_report,
)


class FakeClock:
    def __init__(self, start_s: float = 100.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s


class DummyBus:
    def __init__(self, on_read=None) -> None:
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_args: list[bool] = []
        self.on_read = on_read

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_args.append(disable_torque)
        self.is_connected = False

    def write(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            f"preflight must not write motor registers: {args}, {kwargs}"
        )

    def sync_write(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"preflight must not send Goal_Position: {args}, {kwargs}")

    def sync_read(
        self, data_name: str, *, num_retry: int = 0
    ) -> dict[str, float]:
        assert data_name == "Present_Position"
        assert num_retry == 0
        if self.on_read is not None:
            self.on_read()
        return {joint: 0.0 for joint in ARM_JOINT_NAMES}


class AlwaysAliveThread:
    def is_alive(self) -> bool:
        return True


class RefreshingLock:
    def __init__(self, refresh) -> None:
        self.refresh = refresh

    def __enter__(self):
        self.refresh()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


class DummyCamera:
    def __init__(
        self,
        *,
        clock: FakeClock,
        timestamp_offset_s: float,
        image_value: int,
        expose_timestamp: bool,
    ) -> None:
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.clock = clock
        self.timestamp_offset_s = timestamp_offset_s
        self.latest_frame = np.full((8, 8, 3), image_value, dtype=np.uint8)
        self.frame_lock = RefreshingLock(self._refresh)
        self.thread = AlwaysAliveThread()
        if expose_timestamp:
            self.latest_timestamp: float | None = None

    def _refresh(self) -> None:
        if hasattr(self, "latest_timestamp"):
            self.latest_timestamp = self.clock() + self.timestamp_offset_s

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class DummyArm:
    def __init__(
        self,
        *,
        clock: FakeClock,
        timestamp_offset_s: float,
        image_value: int,
        calibrated: bool,
        expose_timestamp: bool,
        on_read=None,
    ) -> None:
        self.bus = DummyBus(on_read=on_read)
        self.cameras = {
            "wrist": DummyCamera(
                clock=clock,
                timestamp_offset_s=timestamp_offset_s,
                image_value=image_value,
                expose_timestamp=expose_timestamp,
            )
        }
        self.config = SimpleNamespace(
            disable_torque_on_disconnect=True,
            num_read_retries=0,
        )
        self._calibrated = calibrated

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(
            camera.is_connected for camera in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


class DummyBiSOFollower:
    def __init__(
        self,
        clock: FakeClock,
        *,
        capture_latency_s: float = 0.005,
        camera_skew_s: float = 0.005,
        calibrated: bool = True,
        expose_timestamp: bool = True,
        fail_on_observation: bool = False,
    ) -> None:
        self.clock = clock
        self.capture_latency_s = capture_latency_s
        self.camera_skew_s = camera_skew_s

        def left_read() -> None:
            if fail_on_observation:
                raise RuntimeError("camera read failed")

        def right_read() -> None:
            self.observation_calls += 1
            self.clock.now_s += self.capture_latency_s

        self.left_arm = DummyArm(
            clock=clock,
            timestamp_offset_s=-camera_skew_s,
            image_value=0,
            calibrated=calibrated,
            expose_timestamp=expose_timestamp,
            on_read=left_read,
        )
        self.right_arm = DummyArm(
            clock=clock,
            timestamp_offset_s=0.0,
            image_value=1,
            calibrated=calibrated,
            expose_timestamp=expose_timestamp,
            on_read=right_read,
        )
        self.observation_calls = 0
        self.connect_calls = 0
        self.send_calls = 0

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def connect(self, *, calibrate: bool = True) -> None:
        del calibrate
        self.connect_calls += 1
        raise AssertionError("preflight must bypass BiSOFollower.connect()")

    def send_action(self, action: object) -> None:
        del action
        self.send_calls += 1
        raise AssertionError("preflight must never call send_action()")

    def get_observation(self) -> dict[str, object]:
        raise AssertionError(
            "preflight must bypass BiSOFollower.get_observation() to keep "
            "frames and timestamps atomic"
        )


def _config(*, actuation: bool = True, include_home: bool = True) -> ProjectConfig:
    return ProjectConfig(
        runtime=RuntimeConfig(
            backend="lerobot",
            camera_hz=30.0,
            actuation_enabled=actuation,
        ),
        safety=SafetyConfig(
            joint_lower=(-180.0,) * ACTION_DIM,
            joint_upper=(180.0,) * ACTION_DIM,
            max_delta_per_servo_tick=(1.0,) * ACTION_DIM,
            calibrated=True,
        ),
        lerobot=LeRobotConfig(
            left_port="left",
            right_port="right",
            left_wrist_camera=0,
            right_wrist_camera=1,
            camera_width=8,
            camera_height=8,
            camera_fps=30,
            calibration_dir="calibration",
            hardware_id="bench-a",
            calibration_id="cal-a",
            home_joint_position=(0.0,) * ACTION_DIM if include_home else None,
            home_joint_tolerance=(0.1,) * ACTION_DIM if include_home else None,
        ),
    )


def test_preflight_is_goal_write_free_and_reports_automatic_metrics() -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock)

    report = run_hardware_preflight(
        _config(actuation=True),
        seconds=1.0 / 30.0,
        robot_factory=lambda config: robot,
        clock=clock,
        camera_clock=clock,
        sleeper=clock.sleep,
    )

    assert report["evidence_level"] == "real"
    assert report["gate"] == "G6/G7-preflight"
    assert report["result"] == "partial"
    assert report["automated_checks_passed"] is True
    assert report["failures"] == []
    assert report["configured_actuation_enabled"] is True
    assert report["effective_actuation_enabled"] is False
    assert report["goal_position_commands_sent"] == 0
    assert report["camera_keys"] == ["left_wrist", "right_wrist"]
    assert report["camera_resolution"] == [8, 8]
    assert report["camera_fps_estimate_hz"]["left_wrist"] == pytest.approx(30.0)
    assert report["camera_fps_estimate_hz"]["right_wrist"] == pytest.approx(30.0)
    assert report["camera_skew_p95_s"] == pytest.approx(0.005)
    assert report["camera_frame_age_p95_s"] == pytest.approx(0.005)
    assert report["observation_latency_p95_s"] == pytest.approx(0.005)
    assert report["home_within_tolerance"] is True
    assert report["manual_checks_required"]
    assert robot.connect_calls == 0
    assert robot.send_calls == 0
    assert robot.left_arm.bus.connect_calls == 1
    assert robot.right_arm.bus.connect_calls == 1
    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.disconnect_args == [True]
    assert robot.left_arm.cameras["wrist"].disconnect_calls == 1
    assert robot.right_arm.cameras["wrist"].disconnect_calls == 1


def test_preflight_fails_automatic_gates_for_slow_capture_and_camera_skew() -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(
        clock,
        capture_latency_s=0.2,
        camera_skew_s=0.03,
    )

    report = run_hardware_preflight(
        _config(),
        seconds=1.0 / 30.0,
        robot_factory=lambda config: robot,
        clock=clock,
        camera_clock=clock,
        sleeper=clock.sleep,
    )

    assert report["result"] == "fail"
    assert report["automated_checks_passed"] is False
    assert "observation_sampling_fps_below_target" in report["failures"]
    assert "observation_latency_p95_exceeded" in report["failures"]
    assert "camera_skew_p95_exceeded" in report["failures"]
    assert robot.send_calls == 0
    assert robot.is_connected is False


def test_preflight_fails_closed_when_camera_timestamps_are_unavailable() -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock, expose_timestamp=False)

    report = run_hardware_preflight(
        _config(),
        seconds=1.0 / 30.0,
        robot_factory=lambda config: robot,
        clock=clock,
        camera_clock=clock,
        sleeper=clock.sleep,
    )

    assert report["result"] == "fail"
    assert report["camera_timestamp_source"] == "unavailable"
    assert report["camera_skew_p95_s"] is None
    assert "camera_timestamp_unavailable" in report["failures"]
    assert "camera_skew_unavailable" in report["failures"]
    assert "left_wrist_fps_below_target" in report["failures"]
    assert "right_wrist_fps_below_target" in report["failures"]


def test_preflight_rejects_explicit_zero_sample_rate() -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock)

    with pytest.raises(HardwarePreflightError, match="sample_hz.*positive"):
        run_hardware_preflight(
            _config(),
            seconds=1.0,
            sample_hz=0.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.left_arm.bus.connect_calls == 0
    assert robot.right_arm.bus.connect_calls == 0


def test_preflight_disconnects_every_resource_when_capture_fails() -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock, fail_on_observation=True)

    with pytest.raises(RuntimeError, match="camera read failed"):
        run_hardware_preflight(
            _config(),
            seconds=1.0 / 30.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.is_connected is False
    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.right_arm.bus.disconnect_args == [True]
    assert robot.left_arm.cameras["wrist"].disconnect_calls == 1
    assert robot.right_arm.cameras["wrist"].disconnect_calls == 1


def test_preflight_rejects_uncalibrated_arm_and_cleans_up_partial_connect() -> None:
    clock = FakeClock()
    robot = DummyBiSOFollower(clock, calibrated=False)

    with pytest.raises(HardwarePreflightError, match="left arm.*calibration"):
        run_hardware_preflight(
            _config(),
            seconds=1.0 / 30.0,
            robot_factory=lambda config: robot,
            clock=clock,
            camera_clock=clock,
            sleeper=clock.sleep,
        )

    assert robot.left_arm.bus.disconnect_args == [True]
    assert robot.left_arm.cameras["wrist"].connect_calls == 0
    assert robot.right_arm.bus.connect_calls == 0
    assert robot.send_calls == 0


def test_preflight_report_write_is_atomic_and_requires_explicit_overwrite(
    tmp_path,
) -> None:
    report_path = tmp_path / "g6_g7.json"
    report = {"schema_version": 1, "result": "partial", "value": 1.25}

    written = write_hardware_preflight_report(report_path, report)

    assert written == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    with pytest.raises(HardwarePreflightReportError, match="already exists"):
        write_hardware_preflight_report(report_path, report)

    replacement = {**report, "result": "fail"}
    write_hardware_preflight_report(report_path, replacement, overwrite=True)
    assert json.loads(report_path.read_text(encoding="utf-8")) == replacement


def test_preflight_cli_returns_nonzero_when_automatic_checks_fail(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    report_path = tmp_path / "failed-preflight.json"
    config_path = tmp_path / "hardware.toml"
    config_path.write_text("placeholder", encoding="utf-8")
    failed_report = {
        "schema_version": 1,
        "result": "fail",
        "automated_checks_passed": False,
        "failures": ["camera_skew_p95_exceeded"],
    }

    monkeypatch.setattr(
        "so101_wam.hardware_preflight.ProjectConfig.load",
        lambda path: _config(),
    )
    monkeypatch.setattr(
        "so101_wam.hardware_preflight.run_hardware_preflight",
        lambda config, **kwargs: failed_report,
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code != 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == failed_report
    assert json.loads(capsys.readouterr().out) == failed_report
