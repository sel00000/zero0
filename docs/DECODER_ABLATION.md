# Ordered decoder ablation

This is an offline, teacher-forced decoder diagnostic. It does not prove
zero-shot behavior, task success, closed-loop control, sim-to-real transfer, or
physical SO-101 readiness. The physical robot is not constructed by this path.

The study compares three action-decoder modes behind the same CompactWAM
predictor:

- `legacy_mean` is the default compatibility head. It averages future latents
  over time before action decoding.
- `mean_repeat_control` repeats that same time mean into a wide head. Its head
  shape and parameter count match `ordered_concat`, but its future input is
  order-invariant.
- `ordered_concat` concatenates future latents in time order into the same wide
  head shape.

Wide-head shape and parameter-count matching does not equal matching
information, effective capacity, or optimization geometry. The compact predictor
is unchanged: shared context projection plus fixed learned time and view offsets.

## Reproduction

Use an existing installed Python environment. From the repository root, choose a
new output directory each time. In a linked worktree without its own `.venv`,
substitute the installed interpreter path:

```shell
PYTHONPATH=src .venv/bin/python -m so101_wam.robot_free_cli --output-dir runs/decoder_data_seed7_001 --seed 7 --device cpu --policy-steps 1
PYTHONPATH=src .venv/bin/python -m so101_wam.decoder_ablation run --train-episodes runs/decoder_data_seed7_001/episodes/train --validation-episodes runs/decoder_data_seed7_001/episodes/validation --output-dir runs/decoder_ordered_001
PYTHONPATH=src .venv/bin/python -m so101_wam.decoder_ablation verify --bundle runs/decoder_ordered_001
```

The robot-free prep run and its MuJoCo smoke are excluded from the comparison.
They only publish synthetic input episodes, a one-step offline candidate, a
one-step G8 smoke, and a real-output guard rejection. The actual nine-condition
comparison initializes every candidate from scratch and does not reuse prep
config or weights. Current prep config is `stage1_steps=0`, `stage2_steps=1`,
`latent_dim=8`, `future_steps=1`, `action_history_steps=1`,
`ifp_steps=1`.

The first synthetic prep seed creates two train episodes of task
`101:synthetic-train-reach` and two validation episodes of
`201:synthetic-validation-place`. Each episode has 91 frames, 3 seconds at
30 Hz, and 24x32 RGB wrist images. Its action source is
`synthetic_deterministic_no_robot_no_goal_write`. One held-out synthetic task ID
is not independent semantic generalization. Histories are recorded
teacher-forced.

## Fixed Protocol

The comparison protocol is fixed by the public CLI:

- seeds `3`, `7`, and `11`;
- modes `legacy_mean`, `mean_repeat_control`, and `ordered_concat`;
- CPU device, one Torch thread, deterministic algorithms enabled, warn-only
  disabled;
- `latent_dim=32`, `future_steps=3`, one transformer layer, four heads,
  `action_horizon=10`, `action_history_steps=4`;
- 10 Hz policy rate, 50 Hz servo rate, `max_context_steps=300`;
- 100 head-only stage-1 optimizer steps plus 100 end-to-end stage-2 optimizer
  steps per slot;
- task-balanced sampling;
- AdamW with learning rate `1e-4`, weight decay `0.01`, gradient clip norm `1`;
- loss weights future/action/IFP of `1`, `1`, and `0.25`;
- compact linear IFP with `K=2`, stride `2`, and no separate IFP window.

This is a new fixed protocol/study, not a validation-selected extension.

## Immutable Artifacts

`protocol.json` is frozen before optimization. It binds the copied train and
validation NPZ/JSON bytes, source bytes, source revision/status, runtime
versions, normalization, validation order, config, and the nine slots. Each slot
publishes `runs/<slot>/candidate.pt`, `runs/<slot>/training.json`, and immutable
status. The final `comparison.json` uses schema
`so101_wam.decoder_ablation.v1` and preserves all nine slots.

The verifier rebuilds the comparison from stored artifacts and rejects bad,
stale, mixed, or falsely complete bundles. It checks byte hashes and semantic
consistency. If any slot is incomplete, `summary` is `null` and the CLI exits
`2`. Output roots must be new; no overwrite, resume, or retry is part of the
protocol. Hashes bind stored artifacts, not external execution attestation.

The coordinator checks the frozen protocol, source, and runtime both before and
after training. A post-training mismatch fails that slot and leaves later slots
unattempted; an interrupted preflight leaves its remaining slots unattempted.
Verification also cross-checks report protocol claims and checkpoint metadata
against the frozen configuration, data provenance, and recorded validation.

For current-source numerical replay of final stage-2 checkpoint diagnostics, use
the separate `replay --bundle` command described in
[ROBOT_FREE_EXTENSIONS.md](ROBOT_FREE_EXTENSIONS.md). It leaves the bundle
unchanged and does not replay the unsaved stage-1 model.

All candidates remain `offline_trained=true`, `trained=false`, and
`deployment_ready=false`. No decoder mode authorizes real hardware output.

## Metrics

The primary metric is final normalized task-macro action MSE:

```text
ordered_concat - mean_repeat_control
```

Negative values favor the ordered decoder. Window averages first reduce action
rows and axes, then average within each task, then average tasks equally.
Per-seed paired deltas, their mean, and sample standard deviation are descriptive
only. Three seeds are not a significance test, and overlapping windows are not
independent replicas.

Stage 1 reports oracle diagnostics from stored future images. Stage 2 reports
predicted future-latent diagnostics. Native 12-axis MAE is reported separately
because the axes keep their native units. Parameter counts and timings are
diagnostic metadata.

Reverse and permute diagnostics alter only the future-latent time axis; they do
not reverse the prompt. They use independent diagnostic seed `0`. Mean modes
must remain float32-invariant under `rtol=1e-5`, `atol=1e-6`. Ordered sensitivity
may be zero, and nonzero sensitivity does not prove temporal understanding.

The immutable `comparison.json`, not this runbook, is the numeric truth. A small
positive or negative MSE delta is a bounded offline diagnostic, not evidence of
robot efficacy.

## Compatibility

New checkpoints use schema 3 with a mandatory decoder mode. Genuine valid v2
legacy eight-field architecture checkpoints remain loadable without rewriting
weights or reports. Ordinary training still defaults to `legacy_mean`; use
`--action-decoder` only for the wide-head comparison modes.
