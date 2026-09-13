# zero01 architecture

## Target result

zero01 is an independent, workstation-scale research prototype. It is not the
official Zero-WAM implementation. Its benchmark embodiment is a fixed humanoid
torso made from two mirrored SO-101 follower arms. The policy must complete the
core benchmark using exactly two RGB inputs, one camera on each wrist. A head
camera may be recorded for diagnostics but is forbidden as a required policy
feature.

The first scientific milestone is:

> After receiving one 3-12 second bimanual sensorimotor demonstration, execute a
> held-out tabletop task without a gradient update, using only two wrist RGB
> streams and dual-arm proprioception.

## Evidence and deliberate reductions

| Source idea | Preserved contract | MVP reduction |
| --- | --- | --- |
| GEN-1.5 | 3-12 s physical prompt, 30 s rolling sensorimotor context, no deployment-time weight update | Policy video tokens run at 10 Hz; the servo executor runs at 50 Hz rather than claiming the original model's 100 Hz implementation. [Direct video evidence](VIDEO_REFERENCE.md) is separated from inference. |
| Zero-WAM | Causal next-video prediction followed by inverse-dynamics action decoding; prompt is prefix memory | Compact latent future prediction rather than Wan-2.2-TI2V-5B pixel generation; this prototype caches pre-temporal prompt tokens but still recomputes full temporal attention on each policy call |
| Zero-WAM IFP | Training-only strided future targets to discourage ignoring the prompt | Default checkpoints retain a compact `K=2` head; a removable fused-module ablation supports `K=0/2/4`, stride 2, but remains latent-MSE rather than flow matching |
| MACT | Historical wrist images and short closed-loop action chunks | Two shared RGB encoders, bimanual joint targets, no depth requirement |
| LeRobot 0.6.1 | `bi_so_follower`, 12 positional features, named camera features, 30 FPS datasets | Adapter boundary remains optional so contracts and safety tests run without hardware packages |

The source paper reports a 5B Wan-derived model and 15,360 GPU-hours. Reproducing
that scale is explicitly outside the MVP. The architecture keeps compatible
boundaries so an official future checkpoint can replace the compact model.

## Component and data flow

```text
3-12 s sensorimotor demo
  wrist RGB L/R + 12 joint states + 12 executed targets
                         |
                         v
                 pinned prompt prefix ------------------+
                                                        |
live wrist RGB L/R + joint state + executed history     |
                         |                              |
                         v                              v
              30 s causal rolling context ---> CausalVideoCore
                                                        |
                                             predicted future
                                             dual-wrist latents
                                                        |
                                                        v
                                             InverseDynamicsHead
                                                        |
                                             joint action chunk
                                                        |
                                                        v
   fault latch + freshness + limits + delta gate -> SafetySupervisor
                                                        |
                                      +-----------------+-----------------+
                                      |                                   |
                                      v                                   v
                        MuJoCo dual-SO101                    LeRobot `bi_so_follower`
                    collision/render evidence                    real hardware
```

## Canonical interfaces

### Sensorimotor frame

- `images.left_wrist`: `uint8[H,W,3]`
- `images.right_wrist`: `uint8[H,W,3]`
- `joint_position`: `float32[12]`
- `executed_action`: `float32[12]` for prompt/history frames
- `timestamp_s`: monotonic timestamp

The 12-axis order is fixed:

1. left `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
   `wrist_roll`, `gripper`
2. right, in the same order

LeRobot feature keys append `.pos`, for example
`left_shoulder_pan.pos`. Camera names configured per arm as `wrist` become
`left_wrist` and `right_wrist` in `BiSOFollower` observations.

### Action chunk

`target_joint_position: float32[H,12]` contains absolute, calibrated LeRobot
position targets: five body joints per arm in degrees and each gripper in
0–100 units. The LeRobot factory explicitly fixes `use_degrees=true`; MuJoCo
converts this public contract to radians internally. The MVP defaults to ten targets at 50 Hz (200 ms). The safety
layer validates one target per servo tick. A rejection clears the remaining
horizon and enters explicit recovery; raw multi-row policy output never reaches
the adapter. MuJoCo additionally preflights the target pose and rolls back a
transition that creates a forbidden contact.

### Context

The prompt is validated once and frozen. At a 10 Hz policy rate, a 30 second
window has at most 300 frame slots. Prompt slots are pinned and the remaining
slots form a FIFO of live frames. Inference never mutates model weights.
`CompactWAMPolicy` reuses the prompt image/sensorimotor token encoding while the
modality, content fingerprint, model instance, and device remain unchanged.
Changing the prompt or calling `clear_prompt_cache()` invalidates it. Cached and
uncached inference have exact parity tests. This is not a Transformer KV/prefix
cache: causal temporal attention still processes prompt and live tokens together
on every policy call. Callers that mutate or reload model weights in place must
call `clear_prompt_cache()` first.

Human video and language are separate offline task-spec contracts rather than
nullable fields on `PhysicalPrompt`. The adapter keeps real runtime on the robot
episode path and exists only to validate modality shape, provenance, and signal
isolation. Its compact text encoder is deterministic, not learned. See
[TASK_SPECS.md](TASK_SPECS.md).

## Model boundary

The compact model has four separable modules:

1. `DualWristEncoder`: shared convolutional tokenizer plus view, time, and
   prompt/live segment embeddings.
2. `CompactWAM` temporal core plus `FutureLatentPredictor`: causal transformer
   that predicts future dual-wrist latent tokens.
3. `InverseDynamicsActionHead`: sees robot history and predicted future robot latents,
   but not prompt frames directly. This preserves Zero-WAM's factorization.
4. `ifp_head`: legacy compact training-only future-latent predictions; excluded
   at inference. The ablation-only `FusedIFP` instead uses one copied temporal
   module per future target over fused layer summaries and is never stored in the
   inference checkpoint.

Local candidate training now runs in two stages: inverse dynamics with
ground-truth future wrist latents, then end-to-end training with predicted
latents and training-only IFP. Prompt/target episodes are same-task but distinct,
and validation tasks are held out by task id and label. Per-axis statistics come
from training data only and are stored in checkpoint buffers. A direct
ACT/MACT-style head remains only a measurable baseline, not evidence that the WAM
hypothesis works. See [TRAINING.md](TRAINING.md) and
[IFP_ABLATION.md](IFP_ABLATION.md).

## Runtime state machine

```text
BOOT -> PROMPT_CACHED -> ROLLOUT_READY -> ROLLING
                              ^             |
                              |             +-> policy refresh replaces horizon
                              +-------------+-> horizon consumed / explicit pause
                                            |
                                            +-> RECOVERY -> explicit recover
                                            |
                                            +-> HALT (terminal, fault latched)
```

Prompt recording, calibration, collision validation, and success scoring are
external gates rather than invented runtime states. A `lerobot` backend requires
`calibrated=true`, distinct arm ports, hardware/calibration identities, and a
measured arms-down home pose/tolerance before configuration can enable output.
After connection, output additionally requires LeRobot `is_calibrated=true` and
both wrist camera timestamps. Every measured joint must remain within the home
tolerance throughout history priming. `SensorimotorFrame` carries optional per-image
timestamps; real actuation requires both primary timestamps and rejects measured
skew above the configured limit. The LeRobot adapter copies each
`latest_frame`/`latest_timestamp` pair under the same OpenCV camera lock and rejects
camera-buffer age above the observation-age limit before an action can be sent.

## Timing targets

- camera capture: 30 Hz
- policy video sampling: 10 Hz
- receding policy/action-chunk refresh: 10 Hz target
- servo interpolation: 50 Hz initially
- maximum wrist-camera skew: 17 ms
- maximum observation age: 100 ms
- stale policy chunk: discard after 100 ms; servo rows retain the original
  chunk timestamp so the watchdog cannot be refreshed accidentally

These are test targets, not claims about unmeasured hardware. Real end-to-end
latency and reliable bus rate must be measured before enabling motors.

## Safety boundary

- external E-stop hardware is not a project requirement
- the process-level fault latch requires an explicit reset
- per-joint calibrated position and relative-target limits
- stale atomic camera frame and watchdog rejection: no bus command, then managed fault latch/disconnect
- missing real wrist timestamps or wrist-camera skew above 17 ms: no bus command
- protected torso/table volumes and dual-arm collision corridor: required but
  implemented for the nominal MuJoCo scene; real geometry/transforms still require measurement
- camera identity and calibration checks
- motor current/temperature monitoring when exposed by the hardware layer
- optional future-visual divergence detector for recovery

No configuration in this repository enables real actuation by default. Values
in `configs/fake.toml` are normalized fake-device ranges and must never be copied
into a hardware deployment unchanged. `configs/lerobot_hardware.template.toml`
is also disabled and uncalibrated by construction.

The observation-only hardware preflight is `hardware_preflight.py` -> raw calibrated
motor-bus reads plus locked frame/timestamp snapshots -> atomic G6/G7 JSON report. It deliberately bypasses
`SOFollower.configure()` and never writes `Goal_Position`.

The policy hardware path is `checkpoint.py` -> `dataset.physical_prompt_from_episode`
-> `lerobot_factory.py` -> `LeRobotBiSOAdapter` -> `run_managed_rollout`.
The factory imports LeRobot lazily, verifies version `0.6.1`, places each camera
under its arm-local `"wrist"` key, and leaves connection/output ownership to the
managed rollout. See [BRINGUP.md](BRINGUP.md).

## Growth path

1. Contract, prompt buffer, fake devices, and fail-closed safety tests.
2. MACT/ACT baseline using two wrist views.
3. Compact latent-video predictor plus inverse dynamics.
4. IFP structure/comparability ablation, then meaningful held-out task evaluation.
5. Human-hand video prompts after robot-to-robot prompting is reliable.
6. Optional head-camera ablation; the core score remains wrist-only.
7. Replace compatible modules with official Zero-WAM artifacts after release and
   license/API review.

The simulator embodiment and evidence contract are specified in [MUJOCO.md](MUJOCO.md).
The official Zero-WAM project page is an additional method source; MuJoCo is a
local embodiment/verification choice rather than a claim about the authors'
runtime.
