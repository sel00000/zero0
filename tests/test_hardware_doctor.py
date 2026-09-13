from __future__ import annotations

from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path

import pytest

from so101_wam.hardware_doctor import main, run_hardware_doctor


REQUIRED_SCRIPTS = {
    "lerobot-find-port",
    "lerobot-find-cameras",
    "lerobot-setup-motors",
    "lerobot-calibrate",
    "lerobot-teleoperate",
}


def _missing_package(name: str) -> str:
    raise PackageNotFoundError(name)


def _version(name: str) -> str:
    return {"lerobot": "0.6.1", "torch": "2.11.0"}[name]


def _which(name: str) -> str | None:
    if name in REQUIRED_SCRIPTS or name == "ffmpeg":
        return f"/usr/bin/{name}"
    return None


def _write_ready_config(
    path: Path,
    *,
    left_port: Path,
    right_port: Path,
    left_camera: Path,
    right_camera: Path,
    calibration_dir: Path,
) -> None:
    twelve_zeros = ", ".join("0" for _ in range(12))
    twelve_ones = ", ".join("1" for _ in range(12))
    path.write_text(
        f"""
[runtime]
backend = "lerobot"
actuation_enabled = false

[safety]
calibrated = true
joint_lower = [{', '.join('-180' for _ in range(12))}]
joint_upper = [{', '.join('180' for _ in range(12))}]
max_delta_per_servo_tick = [{twelve_ones}]

[lerobot]
left_port = "{left_port}"
right_port = "{right_port}"
left_wrist_camera = "{left_camera}"
right_wrist_camera = "{right_camera}"
calibration_dir = "{calibration_dir}"
hardware_id = "fixed-torso-a"
calibration_id = "cal-a"
home_joint_position = [{twelve_zeros}]
home_joint_tolerance = [{twelve_ones}]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_doctor_reports_wsl_dependency_and_device_blockers(tmp_path) -> None:
    device_root = tmp_path / "dev"
    device_root.mkdir()

    report = run_hardware_doctor(
        device_root=device_root,
        package_version=_missing_package,
        executable_lookup=lambda name: None,
        executable_directory=tmp_path / "bin",
        usbipd_candidate_paths=(),
        python_version=(3, 12, 3),
        platform_release="6.6.87.2-microsoft-standard-WSL2",
    )

    assert report["result"] == "blocked"
    assert report["environment"]["is_wsl"] is True
    assert report["environment"]["packages"] == {
        "lerobot": None,
        "torch": None,
    }
    assert report["devices"] == {"serial": [], "cameras": []}
    assert {
        "lerobot_not_installed",
        "torch_not_installed",
        "lerobot_scripts_missing",
        "serial_devices_missing",
        "camera_devices_missing",
        "wsl_usbipd_not_available",
        "wsl_usb_serial_not_attached",
        "wsl_video_devices_unavailable",
        "config_not_provided",
    }.issubset(report["blockers"])
    assert report["environment"]["executables"]["usbipd"] is None
    assert report["side_effects"] == {
        "commands_executed": [],
        "device_files_opened": 0,
        "hardware_connections_attempted": 0,
        "motor_writes": 0,
    }


def test_doctor_distinguishes_installed_usbipd_from_unattached_wsl_devices(
    tmp_path,
) -> None:
    device_root = tmp_path / "dev"
    device_root.mkdir()

    def which(name: str) -> str | None:
        if name == "usbipd.exe":
            return "/mnt/c/Program Files/usbipd-win/usbipd.exe"
        return _which(name)

    report = run_hardware_doctor(
        device_root=device_root,
        package_version=_version,
        executable_lookup=which,
        executable_directory=tmp_path / "bin",
        usbipd_candidate_paths=(),
        python_version=(3, 12, 3),
        platform_release="6.6.87.2-microsoft-standard-WSL2",
    )

    assert "wsl_usbipd_not_available" not in report["blockers"]
    assert "wsl_usb_serial_not_attached" in report["blockers"]
    assert report["environment"]["executables"]["usbipd"].endswith(
        "usbipd.exe"
    )
    assert any("attach each USB serial adapter" in item for item in report["next_actions"])


def test_doctor_finds_usbipd_at_standard_wsl_mount_path(tmp_path) -> None:
    device_root = tmp_path / "dev"
    device_root.mkdir()
    usbipd_path = tmp_path / "Program Files" / "usbipd-win" / "usbipd.exe"
    usbipd_path.parent.mkdir(parents=True)
    usbipd_path.touch()
    usbipd_path.chmod(0o755)

    report = run_hardware_doctor(
        device_root=device_root,
        package_version=_version,
        executable_lookup=lambda name: None,
        executable_directory=tmp_path / "bin",
        usbipd_candidate_paths=(usbipd_path,),
        python_version=(3, 12, 3),
        platform_release="6.6.87.2-microsoft-standard-WSL2",
    )

    assert "wsl_usbipd_not_available" not in report["blockers"]
    assert report["environment"]["executables"]["usbipd"] == str(usbipd_path)
    assert any("attach each USB serial adapter" in item for item in report["next_actions"])


def test_doctor_finds_lerobot_scripts_beside_current_interpreter(tmp_path) -> None:
    device_root = tmp_path / "dev"
    device_root.mkdir()
    bin_directory = tmp_path / "venv" / "bin"
    bin_directory.mkdir(parents=True)
    for name in REQUIRED_SCRIPTS:
        script = bin_directory / name
        script.touch()
        script.chmod(0o755)

    report = run_hardware_doctor(
        device_root=device_root,
        package_version=_version,
        executable_lookup=lambda name: None,
        executable_directory=bin_directory,
        python_version=(3, 12, 3),
        platform_release="6.8.0-generic",
    )

    assert "lerobot_scripts_missing" not in report["blockers"]
    assert report["environment"]["missing_lerobot_scripts"] == []
    assert all(
        Path(path).parent == bin_directory
        for name, path in report["environment"]["executables"].items()
        if name in REQUIRED_SCRIPTS
    )


def test_doctor_passes_for_complete_read_only_bringup_environment(tmp_path) -> None:
    device_root = tmp_path / "dev"
    device_root.mkdir()
    left_port = device_root / "ttyACM0"
    right_port = device_root / "ttyACM1"
    left_camera = device_root / "video0"
    right_camera = device_root / "video2"
    for path in (left_port, right_port, left_camera, right_camera):
        path.touch()
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "left.json").write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "hardware.toml"
    _write_ready_config(
        config_path,
        left_port=left_port,
        right_port=right_port,
        left_camera=left_camera,
        right_camera=right_camera,
        calibration_dir=calibration_dir,
    )

    report = run_hardware_doctor(
        config_path,
        device_root=device_root,
        package_version=_version,
        executable_lookup=_which,
        executable_directory=tmp_path / "bin",
        python_version=(3, 12, 3),
        platform_release="6.8.0-generic",
    )

    assert report["result"] == "pass"
    assert report["blockers"] == []
    assert report["config"]["loaded"] is True
    assert report["config"]["actuation_enabled"] is False
    assert len(report["devices"]["serial"]) == 2
    assert len(report["devices"]["cameras"]) == 2
    assert "serial_paths_not_stable" in report["warnings"]
    assert "camera_paths_may_change" in report["warnings"]
    assert "left_right_identity_requires_manual_verification" in report["warnings"]


def test_doctor_rejects_torch_version_incompatible_with_lerobot(tmp_path) -> None:
    device_root = tmp_path / "dev"
    device_root.mkdir()

    def incompatible_version(name: str) -> str:
        return {"lerobot": "0.6.1", "torch": "2.12.0+cpu"}[name]

    report = run_hardware_doctor(
        device_root=device_root,
        package_version=incompatible_version,
        executable_lookup=_which,
        executable_directory=tmp_path / "bin",
        python_version=(3, 12, 3),
        platform_release="6.8.0-generic",
    )

    assert "torch_version_incompatible_with_lerobot_0_6_1" in report["blockers"]


def test_doctor_rejects_missing_configured_device_without_opening_it(
    tmp_path,
) -> None:
    device_root = tmp_path / "dev"
    device_root.mkdir()
    left_port = device_root / "ttyACM0"
    right_port = device_root / "ttyACM1"
    left_camera = device_root / "video0"
    right_camera = device_root / "video2"
    for path in (left_port, right_port, left_camera):
        path.touch()
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    config_path = tmp_path / "hardware.toml"
    _write_ready_config(
        config_path,
        left_port=left_port,
        right_port=right_port,
        left_camera=left_camera,
        right_camera=right_camera,
        calibration_dir=calibration_dir,
    )

    report = run_hardware_doctor(
        config_path,
        device_root=device_root,
        package_version=_version,
        executable_lookup=_which,
        executable_directory=tmp_path / "bin",
        python_version=(3, 12, 3),
        platform_release="6.8.0-generic",
    )

    assert report["result"] == "blocked"
    assert "configured_right_wrist_camera_missing" in report["blockers"]
    assert report["side_effects"]["device_files_opened"] == 0


@pytest.mark.parametrize(
    ("result", "expected_exit"),
    (("pass", 0), ("blocked", 2)),
)
def test_doctor_cli_exit_code_tracks_result(
    result,
    expected_exit,
    monkeypatch,
    capsys,
) -> None:
    report = {"schema_version": 1, "result": result, "blockers": []}
    monkeypatch.setattr(
        "so101_wam.hardware_doctor.run_hardware_doctor",
        lambda config: report,
    )

    assert main([]) == expected_exit
    assert json.loads(capsys.readouterr().out) == report
