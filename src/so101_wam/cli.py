"""Small local smoke runner for the fake two-wrist SO101-WAM path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from .adapters.fake import FakeBimanualRobot
from .config import DEFAULT_FAKE_CONFIG_PATH, ProjectConfig
from .constants import ACTION_DIM
from .context import ContextSnapshot
from .contracts import ActionChunk, PhysicalPrompt, SensorimotorFrame
from .runtime import SO101WAMRuntime


def _frame(timestamp_s: float, *, value: float) -> SensorimotorFrame:
    pixel = int(value * 10) % 255
    return SensorimotorFrame(
        timestamp_s=timestamp_s,
        images={
            "left_wrist": np.full((8, 8, 3), pixel, dtype=np.uint8),
            "right_wrist": np.full((8, 8, 3), pixel + 1, dtype=np.uint8),
        },
        joint_position=np.full(ACTION_DIM, value, dtype=np.float32),
        executed_action=np.full(ACTION_DIM, value, dtype=np.float32),
    )


def _prompt() -> PhysicalPrompt:
    return PhysicalPrompt((_frame(0.0, value=0.0), _frame(3.0, value=0.1)))


class _RampPolicy:
    def __init__(self, *, horizon: int, servo_hz: float) -> None:
        self.horizon = horizon
        self.dt_s = 1.0 / servo_hz

    def predict(self, snapshot: ContextSnapshot, *, now_s: float) -> ActionChunk:
        current = snapshot.live_frames[-1].joint_position
        targets = np.stack(
            [current + 0.1 * (index + 1) for index in range(self.horizon)],
            axis=0,
        ).astype(np.float32)
        return ActionChunk(target_joint_position=targets, dt_s=self.dt_s, created_at_s=now_s)


def run_fake_smoke(
    config: ProjectConfig,
    *,
    policy_steps: int = 1,
    enable_fake_output: bool = False,
) -> dict[str, Any]:
    """Run prompt -> policy -> safety -> servo against memory-only fake devices."""

    if policy_steps < 1:
        raise ValueError("policy_steps must be positive")
    config = replace(
        config,
        runtime=replace(config.runtime, actuation_enabled=enable_fake_output),
    )
    robot = FakeBimanualRobot()
    prompt = _prompt()
    runtime = SO101WAMRuntime(
        config=config,
        prompt=prompt,
        robot=robot,
        policy=_RampPolicy(
            horizon=config.runtime.action_horizon,
            servo_hz=config.runtime.servo_hz,
        ),
    )

    now_s = 4.0
    servo_steps = 0
    servo_ticks_per_policy = ceil(config.runtime.servo_hz / config.runtime.policy_hz)
    last_servo = None
    for _ in range(policy_steps):
        runtime.policy_step(now_s=now_s)
        for _ in range(min(runtime.executor.pending, servo_ticks_per_policy)):
            last_servo = runtime.servo_step(now_s=now_s)
            servo_steps += 1
            now_s += runtime.executor.dt_s
    runtime.pause_rollout()

    assert last_servo is not None
    return {
        "mode": "fake",
        "state": runtime.state.value,
        "shadow": last_servo.shadow,
        "primary_cameras": list(config.runtime.primary_cameras),
        "prompt_fingerprint": runtime.prompt_fingerprint,
        "prompt_duration_s": prompt.duration_s,
        "policy_steps": policy_steps,
        "servo_steps": servo_steps,
        "sent_actions": len(robot.sent_actions),
        "final_joint0": float(robot.joint_position[0]),
        "fault_latched": runtime.safety.fault_latch.latched if runtime.safety is not None else True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a fake SO101-WAM smoke rollout.")
    parser.add_argument("--config", type=Path, default=DEFAULT_FAKE_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--actuate-fake",
        "--enable-fake-output",
        dest="actuate_fake",
        action="store_true",
        help="send only to the in-memory fake robot; this command has no real-hardware path",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")

    result = run_fake_smoke(
        ProjectConfig.load(args.config),
        policy_steps=args.steps,
        enable_fake_output=args.actuate_fake,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
