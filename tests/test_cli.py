from __future__ import annotations

import json
from pathlib import Path

import pytest

from so101_wam.cli import main, run_fake_smoke
from so101_wam.config import ProjectConfig


def test_fake_smoke_defaults_to_shadow_and_sends_nothing() -> None:
    result = run_fake_smoke(ProjectConfig(), policy_steps=2)

    assert result["mode"] == "fake"
    assert result["state"] == "rollout_ready"
    assert result["shadow"] is True
    assert result["primary_cameras"] == ["left_wrist", "right_wrist"]
    assert result["prompt_duration_s"] == pytest.approx(3.0)
    assert result["policy_steps"] == 2
    assert result["servo_steps"] == 10
    assert result["sent_actions"] == 0
    assert result["fault_latched"] is False


def test_fake_smoke_can_send_only_to_in_memory_robot() -> None:
    result = run_fake_smoke(ProjectConfig(), enable_fake_output=True)

    assert result["shadow"] is False
    assert result["servo_steps"] == 5
    assert result["sent_actions"] == 5


def test_cli_default_config_works_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--steps", "1"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "rollout_ready"
    assert payload["shadow"] is True
    assert payload["sent_actions"] == 0


def test_fake_smoke_rejects_zero_steps() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_fake_smoke(ProjectConfig(), policy_steps=0)
