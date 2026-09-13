"""Verify the demo checkpoint and render honest rollout media."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from so101_wam.config import DEFAULT_ROBOT_FREE_CONFIG_PATH, ProjectConfig  # noqa: E402
from so101_wam.adapters.mujoco import MujocoBiSOAdapter  # noqa: E402
from so101_wam.deployment import file_sha256, project_config_sha256  # noqa: E402
from so101_wam.mujoco_semantic_benchmark import SemanticTask  # noqa: E402
from so101_wam.reference_learning import POLICY_STEPS, RAMP_TICKS, REFERENCE_TARGET  # noqa: E402
from so101_wam.sim_reference import SimTrialConfig, TrialKind, run_sim_trial  # noqa: E402


DEFAULT_SOURCE = ROOT / ".worktrees" / "ordered-decoder" / "runs"
DEFAULT_RECOVERY = DEFAULT_SOURCE / "reference_learning_normalized_recovery_001"
DEFAULT_CONTROLS = DEFAULT_SOURCE / "reference_learning_normalized_001"
DEFAULT_OUT = ROOT / "docs" / "assets" / "demo"
DEFAULT_INPUTS = DEFAULT_OUT / "inputs"
DEFAULT_CHECKPOINT = DEFAULT_INPUTS / "reference-nudge-candidate.pt"
DEFAULT_TRAINING_REPORT = DEFAULT_INPUTS / "reference-nudge-training-report.json"
REPORT_NAME = "zero01-demo.json"
MP4_NAME = "zero01-demo.mp4"
GIF_NAME = "zero01-demo.gif"
PREVIEW_NAME = "zero01-preview.png"
OLD_MEDIA = (
    "so101_wam_mujoco_demo.mp4",
    "so101_wam_mujoco_demo.gif",
    "so101_wam_mujoco_demo_preview.png",
    "so101_wam_mujoco_demo.json",
    "zero101-demo.mp4",
    "zero101-demo.gif",
    "zero101-preview.png",
    "zero101-demo.json",
)
SEEDS = (7, 13)
VIDEO_FPS = 12
GIF_FPS = 8
GIF_WIDTH = 640
FRAME_W = 1152
FRAME_H = 648
CAMERA_W = 384
CAMERA_H = 288
TITLE_H = 48
CAPTION_H = 24
FOOTER_H = 24
IMAGE_W = 352
IMAGE_H = 264
OVERVIEW_W = IMAGE_W * 2
OVERVIEW_H = IMAGE_H * 2
OVERVIEW_X = 48
RIGHT_X = 752
CAPTION_1_Y = TITLE_H
IMAGE_1_Y = TITLE_H + CAPTION_H
CAPTION_2_Y = IMAGE_1_Y + IMAGE_H
IMAGE_2_Y = CAPTION_2_Y + CAPTION_H
FOOTER_Y = FRAME_H - FOOTER_H
OVERVIEW_LABEL = "overview replay"
LEFT_LABEL = "left wrist replay"
RIGHT_LABEL = "right wrist replay"
LEARNED_HOLD_S = 3.0
CRF = 28
REPLAY_ATOL_M = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "zero01_verification_001")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_ROBOT_FREE_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replay-from", type=Path)
    return parser.parse_args()


def task() -> SemanticTask:
    return SemanticTask(
        task_id="reference-nudge-block",
        label="sim-reference-nudge-block",
        policy_steps=POLICY_STEPS,
        seeds=SEEDS,
        object_body="task_block",
        initial_object_positions=(
            (7, (0.2162682466, 0.1176982024, 0.475)),
            (13, (0.2162682466, 0.1181982024, 0.475)),
        ),
        target_object_position=(0.210, 0.105, 0.470),
        position_tolerance_m=0.004,
    )


def waypoints(config: ProjectConfig) -> tuple[tuple[float, ...], np.ndarray]:
    target = np.asarray((*REFERENCE_TARGET, *([0.0] * 6)), dtype=np.float32)
    duration_s = POLICY_STEPS / config.runtime.policy_hz
    times = (0.0, (RAMP_TICKS - 1) / config.runtime.servo_hz, duration_s)
    targets = np.stack((target / RAMP_TICKS, target, target)).astype(np.float32)
    return times, targets


def prompt_paths(seed: int, inputs_dir: Path) -> tuple[Path, Path]:
    if seed == 7:
        stem = "seed-7-reference-prompt"
    elif seed == 13:
        stem = "seed-13-reference-prompt"
    else:
        raise ValueError(f"unsupported seed: {seed}")
    return inputs_dir / f"{stem}.npz", inputs_dir / f"{stem}.json"


def run_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"run directory must be empty: {args.run_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.training_report.is_file():
        raise FileNotFoundError(args.training_report)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig.load(args.config)
    config = replace(config, runtime=replace(config.runtime, camera_hz=config.runtime.servo_hz))
    demo_task = task()
    times, targets = waypoints(config)
    duration_s = POLICY_STEPS / config.runtime.policy_hz
    outcomes: list[dict[str, Any]] = []

    for seed in SEEDS:
        prompt_npz, prompt_manifest = prompt_paths(seed, args.inputs_dir)
        if not prompt_npz.is_file() or not prompt_manifest.is_file():
            raise FileNotFoundError(f"missing reference prompt for seed {seed}")
        specs = (
            (TrialKind.REFERENCE, 0, {"times_s": times, "targets": targets}),
            (TrialKind.HOLD, 1, {}),
            (
                TrialKind.LEARNED,
                2,
                {
                    "checkpoint_path": args.checkpoint,
                    "prompt_path": prompt_npz,
                    "prompt_manifest_path": prompt_manifest,
                    "device": args.device,
                },
            ),
        )
        for kind, offset, kwargs in specs:
            output_dir = args.run_dir / f"seed-{seed}" / kind.value
            output_dir.mkdir(parents=True, exist_ok=True)
            trial = SimTrialConfig(
                config=config,
                task=demo_task,
                seed=seed,
                output_dir=output_dir,
                kind=kind,
                duration_s=duration_s,
                episode_index=(seed * 10) + offset,
                **kwargs,
            )
            outcome = run_sim_trial(trial)
            outcomes.append({"seed": seed, "kind": kind.value, "outcome": outcome})
    return outcomes


def find_report(run_dir: Path, seed: int, kind: str) -> Path:
    matches = sorted((run_dir / f"seed-{seed}" / kind).glob(f"{kind}_reference-nudge-block_seed-{seed}_ep-*.json"))
    matches = [path for path in matches if not path.name.endswith("_episode.json")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {kind} report for seed {seed}, got {len(matches)}")
    return matches[0]


def load_outcomes(run_dir: Path) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for seed in SEEDS:
        for kind in ("reference", "hold", "learned"):
            report = json.loads(find_report(run_dir, seed, kind).read_text())
            outcomes.append({"seed": seed, "kind": kind, "outcome": report})
    return outcomes


def load_episode(report: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
    episode = report.get("episode")
    if not isinstance(episode, dict):
        return None, None
    manifest_path = Path(str(episode["manifest_path"]))
    npz_path = Path(str(episode["npz_path"]))
    manifest = json.loads(manifest_path.read_text())
    arrays = dict(np.load(npz_path))
    return manifest, arrays


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def fit(draw: ImageDraw.ImageDraw, text: str, size: int, width: int) -> ImageFont.ImageFont:
    result = font(size)
    while size > 10 and draw.textlength(text, font=result) > width:
        size -= 1
        result = font(size)
    return result


def distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(delta))


def trajectory_error(
    original: list[dict[str, Any]],
    replay: list[tuple[float, float, float]],
) -> float | None:
    if not original or not replay:
        return None
    count = min(len(original), len(replay))
    errors = [
        distance(
            tuple(float(value) for value in original[index]["position"]),
            replay[index],
        )
        for index in range(count)
    ]
    return max(errors)


def outcome_text(item: dict[str, Any], duration_s: float) -> tuple[str, str]:
    outcome = item["outcome"]
    kind = item["kind"]
    success = bool(outcome.get("object_success", False))
    result = "PASS" if success else "FAIL"
    error = outcome.get("object_position_error_m")
    if error is None:
        error_text = "error=n/a"
    else:
        error_text = f"error={float(error) * 1000:.2f}mm / 4.00mm limit"
    reason = outcome.get("failure_reason")
    if reason:
        error_text = f"{error_text} | {reason}"
    title = f"zero01 recorded-action verification | seed {item['seed']} | {kind}"
    detail = f"Final result: {result} | {error_text} | duration={duration_s:.2f}s"
    return title, detail


def label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int) -> None:
    draw.text(xy, text, fill=(245, 246, 248), font=fit(draw, text, 15, width))


def compose(
    item: dict[str, Any],
    images: dict[str, np.ndarray],
    duration_s: float,
    display_t_s: float,
) -> Image.Image:
    out = Image.new("RGB", (FRAME_W, FRAME_H), (8, 10, 14))
    draw = ImageDraw.Draw(out)
    title, detail = outcome_text(item, duration_s)
    draw.text((48, 6), title, fill=(255, 255, 255), font=fit(draw, title, 22, FRAME_W - 96))
    draw.text((48, 29), detail, fill=(220, 225, 232), font=fit(draw, detail, 15, FRAME_W - 96))

    overview = Image.fromarray(images["head_optional"]).resize((OVERVIEW_W, OVERVIEW_H), Image.Resampling.LANCZOS)
    left = Image.fromarray(images["left_wrist"]).resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)
    right = Image.fromarray(images["right_wrist"]).resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)

    label(draw, OVERVIEW_LABEL, (OVERVIEW_X, CAPTION_1_Y + 4), OVERVIEW_W)
    label(draw, LEFT_LABEL, (RIGHT_X, CAPTION_1_Y + 4), IMAGE_W)
    label(draw, RIGHT_LABEL, (RIGHT_X, CAPTION_2_Y + 4), IMAGE_W)

    out.paste(overview, (OVERVIEW_X, IMAGE_1_Y))
    out.paste(left, (RIGHT_X, IMAGE_1_Y))
    out.paste(right, (RIGHT_X, IMAGE_2_Y))

    draw.rectangle((0, FOOTER_Y, FRAME_W, FRAME_H), fill=(0, 0, 0))
    footer = (
        "High-resolution replay of recorded accepted actions; scores from original rollout reports"
        f" | t={display_t_s:.2f}s"
    )
    footer_font = fit(draw, footer, 14, FRAME_W - 96)
    draw.text((48, FOOTER_Y + 5), footer, fill=(245, 246, 248), font=footer_font)
    return out


def replay_clip(
    item: dict[str, Any],
    manifest: dict[str, Any] | None,
    arrays: dict[str, np.ndarray] | None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    if manifest is None or arrays is None:
        raise ValueError("each scored trial must include an episode for replay")
    duration = float(manifest["duration_s"])
    frame_count = max(1, int(round(max(duration, LEARNED_HOLD_S) * VIDEO_FPS)))
    actions = arrays["action"]
    config = ProjectConfig.load(DEFAULT_ROBOT_FREE_CONFIG_PATH)
    config = replace(
        config,
        runtime=replace(config.runtime, camera_hz=config.runtime.servo_hz, use_head_camera=True),
        mujoco=replace(config.mujoco, camera_width=CAMERA_W, camera_height=CAMERA_H),
    )
    demo_task = task()
    adapter = MujocoBiSOAdapter(
        config=config.mujoco,
        servo_hz=config.runtime.servo_hz,
        actuation_enabled=True,
        include_head_camera=True,
        semantic_object_body=demo_task.object_body,
        initial_object_position=demo_task.initial_position(int(item["seed"])),
    )
    original = item["outcome"].get("object_position_trajectory") or []
    replay_positions: list[tuple[float, float, float]] = []
    frames: list[Image.Image] = []
    servo_hz = float(config.runtime.servo_hz)
    next_frame = 0

    adapter.connect(calibrate=False)
    try:
        for servo_index, action in enumerate(actions):
            adapter.send_action(action)
            position = adapter.object_body_position(demo_task.object_body)
            replay_positions.append(position)
            while next_frame < frame_count and next_frame / VIDEO_FPS <= (servo_index + 1) / servo_hz:
                observation = adapter.get_observation(timestamp_s=next_frame / VIDEO_FPS)
                frames.append(compose(item, observation.images, duration, next_frame / VIDEO_FPS))
                next_frame += 1
        while next_frame < frame_count:
            observation = adapter.get_observation(timestamp_s=next_frame / VIDEO_FPS)
            frames.append(compose(item, observation.images, duration, next_frame / VIDEO_FPS))
            next_frame += 1
        final_position = adapter.object_body_position(demo_task.object_body)
    finally:
        adapter.disconnect()

    max_error = trajectory_error(original, replay_positions)
    terminal = item["outcome"].get("terminal_object_position")
    final_error = None
    final_match = None
    if terminal is not None:
        final_error = distance(tuple(float(value) for value in terminal), final_position)
        final_match = final_error <= REPLAY_ATOL_M
    return frames, {
        "seed": item["seed"],
        "kind": item["kind"],
        "source_npz_sha256": item["outcome"]["episode"]["npz_sha256"],
        "source_manifest_sha256": item["outcome"]["episode"]["manifest_sha256"],
        "action_count": int(actions.shape[0]),
        "frame_count": len(frames),
        "max_trajectory_error_m": max_error,
        "final_position_error_m": final_error,
        "final_match": final_match,
        "visualization": "recorded accepted action replay with high-resolution cameras",
    }


def run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode == 0:
        return
    raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())


def write_media(out_dir: Path, frames: list[Image.Image]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OLD_MEDIA + (MP4_NAME, GIF_NAME, PREVIEW_NAME):
        path = out_dir / name
        if path.exists():
            path.unlink()
    with TemporaryDirectory() as temp:
        frames_dir = Path(temp)
        for index, frame in enumerate(frames):
            frame.save(frames_dir / f"frame_{index:05d}.png")
        pattern = str(frames_dir / "frame_%05d.png")
        mp4_path = out_dir / MP4_NAME
        gif_path = out_dir / GIF_NAME
        preview_path = out_dir / PREVIEW_NAME
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(VIDEO_FPS),
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                str(CRF),
                str(mp4_path),
            ]
        )
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp4_path),
                "-vf",
                f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos",
                "-loop",
                "0",
                str(gif_path),
            ]
        )
        frames[0].save(preview_path)
    return {
        "mp4": {"path": str(mp4_path.relative_to(ROOT)), "sha256": file_sha256(mp4_path)},
        "gif": {"path": str(gif_path.relative_to(ROOT)), "sha256": file_sha256(gif_path)},
        "preview": {"path": str(preview_path.relative_to(ROOT)), "sha256": file_sha256(preview_path)},
    }


def summarize(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for item in outcomes:
        outcome = item["outcome"]
        episode = outcome.get("episode") if isinstance(outcome.get("episode"), dict) else None
        summary.append(
            {
                "seed": item["seed"],
                "kind": item["kind"],
                "status": outcome.get("status"),
                "object_success": outcome.get("object_success"),
                "command_count": outcome.get("command_count"),
                "failure_reason": outcome.get("failure_reason"),
                "object_position_error_m": outcome.get("object_position_error_m"),
                "initial_object_position": outcome.get("initial_object_position"),
                "terminal_object_position": outcome.get("terminal_object_position"),
                "episode": episode,
            }
        )
    return summary


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if args.replay_from is not None:
        outcomes = load_outcomes(args.replay_from)
        run_dir = args.replay_from
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        outcomes = load_outcomes(args.run_dir)
        run_dir = args.run_dir
    else:
        outcomes = run_trials(args)
        run_dir = args.run_dir
    frames: list[Image.Image] = []
    replays: list[dict[str, Any]] = []
    for item in outcomes:
        manifest, arrays = load_episode(item["outcome"])
        clip, replay = replay_clip(item, manifest, arrays)
        frames.extend(clip)
        replays.append(replay)
    media = write_media(args.out_dir, frames)
    config = ProjectConfig.load(args.config)
    report = {
        "schema_version": 1,
        "artifact_kind": "zero01.demo_policy_verification",
        "scope": "known reference-nudge task only",
        "limitations": [
            "not one of the seven requested household tasks",
            "requested task scenes/prompts are absent",
            "simulation only; not hardware evidence",
            "not zero-shot evidence",
            "current run uses the current fixed source tree; earlier recovery reports are historical",
            "checkpoint is a limited IFP0/legacy-mean diagnostic, not a full architecture reproduction",
        ],
        "reproduction_command": (
            ".venv/bin/python scripts/verify_demo_policy.py "
            f"--run-dir {rel(args.run_dir)}"
        ),
        "replay_command": (
            ".venv/bin/python scripts/verify_demo_policy.py "
            f"--replay-from {rel(run_dir)}"
        ),
        "config": rel(args.config),
        "config_sha256": project_config_sha256(config),
        "checkpoint": rel(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "training_report": rel(args.training_report),
        "training_report_sha256": file_sha256(args.training_report),
        "inputs_dir": rel(args.inputs_dir),
        "input_snapshot_provenance": {
            "checkpoint_source": rel(DEFAULT_RECOVERY / "candidate.pt"),
            "training_report_source": rel(DEFAULT_RECOVERY / "training-report.json"),
            "seed_7_prompt_source": rel(
                DEFAULT_CONTROLS
                / "control-0"
                / "reference_reference-nudge-block_seed-7_ep-000000_episode"
            ),
            "seed_13_prompt_source": rel(
                DEFAULT_CONTROLS
                / "control-2"
                / "reference_reference-nudge-block_seed-13_ep-000002_episode"
            ),
        },
        "run_dir": rel(run_dir),
        "video_fps": VIDEO_FPS,
        "frame_count": len(frames),
        "display_duration_s": len(frames) / VIDEO_FPS,
        "trials": summarize(outcomes),
        "recorded_action_replay": replays,
        "media": media,
    }
    report_path = args.out_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
