"""Read-only host and device discovery for dual SO-101 bring-up."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import Any

from .config import ConfigError, ProjectConfig
from .constants import LEROBOT_VERSION_PIN


REQUIRED_LEROBOT_SCRIPTS = (
    "lerobot-find-port",
    "lerobot-find-cameras",
    "lerobot-setup-motors",
    "lerobot-calibrate",
    "lerobot-teleoperate",
)
LEROBOT_TORCH_MIN = (2, 7, 0)
LEROBOT_TORCH_MAX_EXCLUSIVE = (2, 12, 0)
DEFAULT_WSL_USBIPD_CANDIDATES = (
    Path("/mnt/c/Program Files/usbipd-win/usbipd.exe"),
)


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _device_record(path: Path, *, stable: bool) -> dict[str, Any]:
    return {
        "path": str(path),
        "canonical_path": str(path.resolve(strict=False)),
        "stable_alias": stable,
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
    }


def _discover_devices(
    device_root: Path,
    *,
    stable_subdirectory: str,
    fallback_patterns: Sequence[str],
) -> list[dict[str, Any]]:
    candidates: list[tuple[Path, bool]] = []
    stable_directory = device_root / stable_subdirectory
    if stable_directory.is_dir():
        candidates.extend(
            (path, True)
            for path in sorted(stable_directory.iterdir(), key=lambda item: item.name)
        )
    for pattern in fallback_patterns:
        candidates.extend(
            (path, False)
            for path in sorted(device_root.glob(pattern), key=lambda item: item.name)
        )

    records: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for path, stable in candidates:
        if not path.exists():
            continue
        canonical = str(path.resolve(strict=False))
        if canonical in seen_targets:
            continue
        seen_targets.add(canonical)
        records.append(_device_record(path, stable=stable))
    return records


def _configured_camera_path(source: str | int, device_root: Path) -> Path:
    if isinstance(source, int):
        return device_root / f"video{source}"
    return Path(source)


def _check_configured_device(
    path: Path,
    *,
    label: str,
    blockers: list[str],
) -> dict[str, Any]:
    exists = path.exists()
    readable = exists and os.access(path, os.R_OK)
    writable = exists and os.access(path, os.W_OK)
    if not exists:
        blockers.append(f"configured_{label}_missing")
    elif not (readable and writable):
        blockers.append(f"configured_{label}_permission_denied")
    return {
        "path": str(path),
        "exists": exists,
        "readable": readable,
        "writable": writable,
    }


def _load_package_versions(
    package_version: Callable[[str], str],
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("lerobot", "torch"):
        try:
            result[package] = package_version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _append_once(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _find_executable(
    name: str,
    *,
    executable_lookup: Callable[[str], str | None],
    executable_directory: Path,
) -> str | None:
    located = executable_lookup(name)
    if located is not None:
        return located
    candidate_names = (name,) if Path(name).suffix else (name, f"{name}.exe")
    for candidate_name in candidate_names:
        candidate = executable_directory / candidate_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run_hardware_doctor(
    config_path: str | Path | None = None,
    *,
    device_root: str | Path = "/dev",
    package_version: Callable[[str], str] = version,
    executable_lookup: Callable[[str], str | None] = shutil.which,
    executable_directory: str | Path | None = None,
    usbipd_candidate_paths: Sequence[str | Path] | None = None,
    python_version: tuple[int, int, int] | None = None,
    platform_release: str | None = None,
) -> dict[str, Any]:
    """Inspect readiness without opening a device or executing a LeRobot command."""

    blockers: list[str] = []
    warnings: list[str] = []
    root = Path(device_root)
    effective_python = python_version or (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    release = platform_release if platform_release is not None else platform.release()
    is_wsl = "microsoft" in release.lower()
    effective_executable_directory = (
        Path(sys.executable).parent
        if executable_directory is None
        else Path(executable_directory)
    )
    usbipd_path = None
    if is_wsl:
        usbipd_path = _find_executable(
            "usbipd.exe",
            executable_lookup=executable_lookup,
            executable_directory=effective_executable_directory,
        ) or _find_executable(
            "usbipd",
            executable_lookup=executable_lookup,
            executable_directory=effective_executable_directory,
        )
        if usbipd_path is None:
            candidates = (
                DEFAULT_WSL_USBIPD_CANDIDATES
                if usbipd_candidate_paths is None
                else tuple(Path(path) for path in usbipd_candidate_paths)
            )
            usbipd_path = next(
                (
                    str(path)
                    for path in candidates
                    if path.is_file() and os.access(path, os.X_OK)
                ),
                None,
            )

    packages = _load_package_versions(package_version)
    if packages["lerobot"] is None:
        _append_once(blockers, "lerobot_not_installed")
    elif packages["lerobot"] != LEROBOT_VERSION_PIN:
        _append_once(blockers, "lerobot_version_mismatch")
    if packages["torch"] is None:
        _append_once(blockers, "torch_not_installed")
    else:
        torch_version = _version_tuple(packages["torch"])
        if (
            torch_version is None
            or torch_version < LEROBOT_TORCH_MIN
            or torch_version >= LEROBOT_TORCH_MAX_EXCLUSIVE
        ):
            _append_once(
                blockers,
                "torch_version_incompatible_with_lerobot_0_6_1",
            )
    if effective_python < (3, 12, 0):
        _append_once(blockers, "python_version_below_3_12")

    scripts = {
        name: _find_executable(
            name,
            executable_lookup=executable_lookup,
            executable_directory=effective_executable_directory,
        )
        for name in REQUIRED_LEROBOT_SCRIPTS
    }
    missing_scripts = [name for name, path in scripts.items() if path is None]
    if missing_scripts:
        _append_once(blockers, "lerobot_scripts_missing")
    ffmpeg_path = _find_executable(
        "ffmpeg",
        executable_lookup=executable_lookup,
        executable_directory=effective_executable_directory,
    )
    if ffmpeg_path is None:
        _append_once(warnings, "ffmpeg_missing")

    serial_devices = _discover_devices(
        root,
        stable_subdirectory="serial/by-id",
        fallback_patterns=("ttyACM*", "ttyUSB*"),
    )
    camera_devices = _discover_devices(
        root,
        stable_subdirectory="v4l/by-id",
        fallback_patterns=("video*",),
    )
    if len(serial_devices) < 2:
        _append_once(blockers, "serial_devices_missing")
        if is_wsl:
            if usbipd_path is None:
                _append_once(blockers, "wsl_usbipd_not_available")
            _append_once(blockers, "wsl_usb_serial_not_attached")
    elif not any(record["stable_alias"] for record in serial_devices):
        _append_once(warnings, "serial_paths_not_stable")
    if len(camera_devices) < 2:
        _append_once(blockers, "camera_devices_missing")
        if is_wsl:
            _append_once(blockers, "wsl_video_devices_unavailable")
    if camera_devices:
        _append_once(warnings, "camera_paths_may_change")

    config_report: dict[str, Any] = {
        "path": None if config_path is None else str(config_path),
        "loaded": False,
    }
    if config_path is None:
        _append_once(blockers, "config_not_provided")
    else:
        target = Path(config_path)
        if not target.is_file():
            _append_once(blockers, "config_not_found")
        else:
            try:
                config = ProjectConfig.load(target)
            except (ConfigError, OSError, TypeError, ValueError) as error:
                _append_once(blockers, "config_invalid")
                config_report["error"] = str(error)
            else:
                config_report.update(
                    {
                        "loaded": True,
                        "backend": config.runtime.backend,
                        "actuation_enabled": config.runtime.actuation_enabled,
                        "safety_calibrated": config.safety.calibrated,
                        "hardware_id": config.lerobot.hardware_id,
                        "calibration_id": config.lerobot.calibration_id,
                    }
                )
                if config.runtime.backend != "lerobot":
                    _append_once(blockers, "config_backend_not_lerobot")
                try:
                    config.lerobot.require_hardware_session()
                except ConfigError as error:
                    _append_once(blockers, "config_hardware_session_incomplete")
                    config_report["hardware_session_error"] = str(error)
                if config.runtime.actuation_enabled:
                    _append_once(warnings, "actuation_enabled_in_doctor_config")
                if not config.safety.calibrated:
                    _append_once(blockers, "measured_safety_profile_missing")
                if (
                    config.lerobot.home_joint_position is None
                    or config.lerobot.home_joint_tolerance is None
                ):
                    _append_once(blockers, "measured_home_reference_missing")
                if not config.lerobot.hardware_id.strip():
                    _append_once(blockers, "hardware_id_missing")
                if not config.lerobot.calibration_id.strip():
                    _append_once(blockers, "calibration_id_missing")

                calibration_dir = Path(config.lerobot.calibration_dir)
                config_report["calibration_dir"] = str(calibration_dir)
                if not calibration_dir.is_dir():
                    _append_once(blockers, "calibration_directory_missing")
                elif not any(calibration_dir.rglob("*.json")):
                    _append_once(blockers, "calibration_artifacts_missing")

                configured_devices = {
                    "left_port": _check_configured_device(
                        Path(config.lerobot.left_port),
                        label="left_port",
                        blockers=blockers,
                    ),
                    "right_port": _check_configured_device(
                        Path(config.lerobot.right_port),
                        label="right_port",
                        blockers=blockers,
                    ),
                    "left_wrist_camera": _check_configured_device(
                        _configured_camera_path(
                            config.lerobot.left_wrist_camera,
                            root,
                        ),
                        label="left_wrist_camera",
                        blockers=blockers,
                    ),
                    "right_wrist_camera": _check_configured_device(
                        _configured_camera_path(
                            config.lerobot.right_wrist_camera,
                            root,
                        ),
                        label="right_wrist_camera",
                        blockers=blockers,
                    ),
                }
                config_report["devices"] = configured_devices
                _append_once(
                    warnings,
                    "left_right_identity_requires_manual_verification",
                )

    next_actions: list[str] = []
    if "lerobot_not_installed" in blockers or "lerobot_scripts_missing" in blockers:
        next_actions.append(
            "install the pinned hardware environment with: "
            "python -m pip install -e '.[hardware]'"
        )
    if "torch_version_incompatible_with_lerobot_0_6_1" in blockers:
        next_actions.append(
            "re-resolve the hardware extra so torch satisfies >=2.7,<2.12.0"
        )
    if "wsl_usbipd_not_available" in blockers:
        next_actions.append(
            "install usbipd-win on Windows, reopen WSL, and verify usbipd list"
        )
    elif is_wsl and "serial_devices_missing" in blockers:
        next_actions.append(
            "attach each USB serial adapter to WSL with usbipd-win, then verify /dev/ttyACM*"
        )
    if "camera_devices_missing" in blockers:
        next_actions.append(
            "verify two /dev/video* nodes and run lerobot-find-cameras opencv; "
            "WSL webcam forwarding must be proven on this host"
        )
    if "config_not_provided" in blockers or "config_not_found" in blockers:
        next_actions.append(
            "copy configs/lerobot_hardware.template.toml to the ignored local config and fill measured values"
        )
    if not blockers:
        next_actions.append(
            "manually verify left/right port and camera identity, then run the observation-only preflight"
        )

    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_discovery",
        "result": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "environment": {
            "python_version": ".".join(str(value) for value in effective_python),
            "platform_release": release,
            "is_wsl": is_wsl,
            "packages": packages,
            "executables": {
                **scripts,
                "ffmpeg": ffmpeg_path,
                "usbipd": usbipd_path,
            },
            "missing_lerobot_scripts": missing_scripts,
        },
        "devices": {
            "serial": serial_devices,
            "cameras": camera_devices,
        },
        "config": config_report,
        "side_effects": {
            "commands_executed": [],
            "device_files_opened": 0,
            "hardware_connections_attempted": 0,
            "motor_writes": 0,
        },
        "next_actions": next_actions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect dual-SO-101 host, dependency, device, and config readiness "
            "without connecting to hardware."
        )
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)

    report = run_hardware_doctor(args.config)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if report["result"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_hardware_doctor"]
