# CompactWAM offline training

## What this stage produces

The local trainer produces an **offline candidate**, not a real-output-certified
policy. It connects the existing G9 episode format to the CompactWAM loss surface
without changing the project's evidence boundary:

- prompt and target are different episodes of the same task;
- train and validation tasks are fully disjoint by both `task_index` and label;
- future wrist frames supervise the latent-video branch;
- 30 Hz episode actions are linearly interpolated to the 50 Hz, ten-row action
  horizon;
- IFP uses training-only strided future wrist targets;
- the checkpoint stores `artifact_kind="compact_wam_candidate"`,
  `evidence_level="offline"`, `offline_trained=true`, `trained=false`, and
  `deployment_ready=false`.

Consequently, this artifact can be used for offline evaluation, MuJoCo work, and
real-hardware **shadow** rollout. The real-output gate rejects the trainer-emitted
candidate markers before inspecting forged readiness booleans. Re-saving the same
weights with deployment-looking metadata still cannot use the normal real-output
path without a separately verified deployment certification bound to the exact
checkpoint bytes, validated config, hardware, calibration, and G6-G9 evidence.
That promotion is performed only by the fail-closed issuer described in
[CERTIFICATION.md](CERTIFICATION.md); the trainer itself never emits deployment
metadata.

## Robot-free training path

When no SO-101 hardware is available, use the robot-free acceptance CLI contract:

```shell
so101-wam-robot-free --output-dir runs/robot_free_001
```

The output directory must be new or empty. The path is expected to create
synthetic 30 Hz episodes, train a tiny offline candidate, run the exact emitted
checkpoint through the MuJoCo G8 checkpoint-session path, and verify that the
real-output guard rejects the offline candidate.

The CLI also accepts `--config`, `--device`, `--seed`, and `--policy-steps`.
Its default config is bundled with the installed package, so the command does not
depend on the current working directory. The repository copy at
`configs/mujoco_robot_free.toml` is the byte-identical review/override template.
The current contract requires `policy_steps=1`. The run publishes `candidate.pt`,
`candidate.training.json`, a MuJoCo G8 JSON report, `robot_free_result.json`, and
the exact effective config as `effective_config.toml`, plus immutable
`episodes/train` and `episodes/validation` episode payloads with same-stem JSON
manifests. All manifest paths and the duplicated `training.checkpoint_path` and
`training.report_path` references are bundle-relative. The result JSON records
`schema_version`, `result`, `mode`, `evidence_level`, `trained`,
`deployment_ready`, `synthetic_data`, `evidence_inputs`, `training`, `g8`,
`real_output_authorized`, `real_output_rejection_reason`, `artifacts`, and
`limitations`.

An archived or moved bundle can be checked without training or simulation:

```shell
so101-wam-verify-robot-free --result runs/robot_free_001/robot_free_result.json
```

This verifier checks path containment, artifact SHA-256 values, episode manifests,
the offline checkpoint/training contract, the actuated G8 report binding, and a
fresh real-output guard rejection. It is a consistency check rather than an
authenticity signature, and it does not create new simulation or hardware evidence.

This is an offline/simulation acceptance path only. Its training report and G8
JSON can show that the dataset, optimizer, checkpoint, simulator, and fail-closed
guard contracts connect. They do not prove G6 camera bench, G7 motor bench, G9
real prompt capture, G10 real dry rollout, or G11 task success.

## Data layout and split contract

Provide separate directories (or explicit `.npz` paths) for training and
validation. A directory scan is non-recursive and reads every top-level `.npz`
plus its same-stem manifest.

Each split must satisfy all of the following:

1. Every task has at least two episodes with distinct sensorimotor contents, so
   renumbering a copy cannot create a valid prompt/target pair.
2. `task_index` and task label form a one-to-one mapping.
3. Train and validation share no episode fingerprint, sensorimotor content
   fingerprint, task index, or task label.
4. All wrist recordings have one source resolution.
5. Every source passes manifest checksum and episode-schema validation, including
   equal RGB, timestamp, state, and action frame counts.

Content duplicate detection hashes timestamps, RGB, state, action, and FPS without
task labels, episode IDs, or metadata. The existing metadata-bound integrity
fingerprint and split digest are unchanged. This detects exact contents, not
cropped, time-shifted, or otherwise transformed copies.

For example, record multiple immutable demonstrations with a stable task index:

```shell
PYTHONPATH=src python -m so101_wam.prompt_recorder \
  --config configs/lerobot_hardware.local.toml \
  --output-dir data/train \
  --task "place block" \
  --task-index 10 \
  --duration 3.0 \
  --episode-index 100
```

Repeat with a new episode index. Validation must use different held-out tasks,
not another take of a training task.

## Training command

```shell
PYTHONPATH=src python -m so101_wam.train_cli \
  --train-episodes data/train \
  --validation-episodes data/validation \
  --output checkpoints/compact_wam_candidate_001.pt \
  --report reports/compact_wam_candidate_001.training.json \
  --checkpoint-id compact-wam-candidate-001 \
  --device cuda:0 \
  --sampling-strategy task_balanced \
  --stage1-steps 1000 \
  --stage2-steps 5000
```

Installed environments may use `so101-wam-train`. Checkpoint and report paths
are immutable; reruns require new names.

Stage 1 trains inverse dynamics from ground-truth future wrist latents. Stage 2
trains predicted future latents, normalized action loss, and IFP end to end.
AdamW defaults match the paper's reported `1e-4` learning rate and `0.01` weight
decay. The compact reduction uses latent MSE rather than the paper's Wan-based
flow-matching objective.

`task_balanced` is the default sampler. It chooses a task before drawing a
shuffled window within that task, so trajectory-rich tasks do not dominate the
optimizer budget. `window_shuffle` remains available for legacy comparisons.
The report stores the seed, per-task draw histogram, schedule digest, and the
next sampler cursor. Replaying with that cursor reproduces the remaining draw
sequence; it is not a model or optimizer resume checkpoint.

Per-axis normalization is computed from training episodes only and stored as
checkpoint buffers. Wrist RGB is deterministically reduced to a maximum side of
64 pixels before tensor stacking, with the same preprocessing used by runtime
inference. The default compatibility path remains a compact linear `K=2`, stride
2 head. The separate fused ablation path supports removable `K=0/2/4` modules at
stride 2 and applies the paper's reported per-module weights for `K=4`. See
[IFP_ABLATION.md](IFP_ABLATION.md).

## Action decoder comparison

Ordinary offline training defaults to the legacy mean action decoder.
`--action-decoder` supports the wide-head `mean_repeat_control` and
`ordered_concat` modes for the bounded offline comparison in
[DECODER_ABLATION.md](DECODER_ABLATION.md). That comparison is teacher-forced,
offline, and candidate-only; it does not authorize hardware output.

Checkpoint architecture metadata records the decoder mode. Current checkpoints
use the v3 architecture schema, while genuine v2 legacy architecture checkpoints
remain loadable without rewriting existing weights or reports.
Loading rejects non-finite tensors and non-positive normalization scales before
returning a model. Valid legacy decoder inputs may still use different positive
future lengths; empty or wrong-width latent inputs are rejected.

## Action-label meaning

G9 kinesthetic episodes store measured `Present_Position` as an action proxy and
send no `Goal_Position`. The trainer interpolates that measured trajectory onto
the servo clock. It records the source in both report and checkpoint metadata;
it does not relabel the result as a sent motor command.

The JSON report includes split digests, episode/task/window counts, action-source
provenance, and sorted `{task_index, task}` inventories for both train and
validation. It also includes architecture, normalization, optimizer settings,
final losses, validation metrics, initial-model and sampled-schedule hashes, plus
null-prompt and temporal-reversal sensitivity diagnostics. Those deltas are not
causal task-understanding evidence. `training_evidence_sha256` binds the task
inventories and the rest of this report core to the checkpoint metadata, while
the report also stores the final checkpoint SHA-256. A downstream benchmark must
still prove how its own task identity maps to these recorded dataset identities.

## Paired human-video prompt controls

Paired training preserves its raw `mismatched_prompt_*` diagnostics. Those mix
prompt content and length changes. New
`length_controlled_mismatched_prompt_*` diagnostics uniformly resample the
alternate prompt's frame indices to the matched prompt length using nearest
neighbors. The live inputs and matched prompt mask stay fixed; pixels are not
interpolated. Equal-length prompts retain their original frame order.

The report records the control algorithm and original/controlled prompt lengths
in its digest-bound mismatch pairing audit. Historical reports without the new
metrics do not establish length-controlled sensitivity. These controls remove
the prefix-length difference, not every temporal or semantic confound, and do
not establish zero-shot performance or prompt causality.

## Relation to Zero-WAM

The preserved method boundary is teacher-forced trajectory history, future
visual prediction followed by inverse dynamics, human/physical prompt as prefix
memory, and training-only IFP. The reductions are substantial: two wrist views,
a small CNN/Transformer, latent MSE, local SO-101 episodes, and no HumanGen or
Wan-2.2 pretraining. An offline candidate therefore demonstrates that the local
pipeline trains and reloads; it does not demonstrate zero-shot success.
