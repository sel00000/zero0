# Robot-free follow-up checks

These extensions do not authorize physical robot output or establish zero-shot
success. Existing checkpoint formats and default training/demo behavior remain
unchanged. Use an installed environment and a new output directory for each run.

## Final-checkpoint numerical replay

```shell
PYTHONPATH=src .venv/bin/python -m so101_wam.decoder_ablation replay --bundle runs/decoder_ordered_001
```

Artifact verification runs first. Numerical replay then evaluates the final
checkpoint on the frozen validation windows and compares stage-2 rows and
summaries, including reverse/permuted-future diagnostics. It does not retrain or
write into the study bundle. Current evaluator provenance is distinct from the
historical frozen source; bundled Python is never executed.

The stage-1 model was not saved separately, so stage-1 metrics remain
artifact-verified, not numerically replayed. A replay pass is a local numerical
consistency check, not proof of historical execution or independent replication.

## Paired temporal windows

Add `--window-mode all_complete` to the existing
`python -m so101_wam.paired_train_cli` command to use every complete temporal
anchor. Omit it to retain the earliest-anchor default.

Every window retains the action-free human prompt and uses a trailing robot
history, future images, and an action target horizon that fit the recording.
Training draws task, then pair, then anchor with the configured optimizer-step budget; more
available windows do not automatically increase that budget. Anchor identities
are bound into the all-window sampling audit and validation records.

Validation scalar metrics are window-weighted, not independent-trial success
rates. Overlapping windows do not create new independent demonstrations.
Imported `human_reviewed` declarations are not independently verified by this
software. Test fixtures are not real reviewed demonstrations.

## Bounded semantic pilot

```shell
PYTHONPATH=src .venv/bin/python -m so101_wam.semantic_pilot --output-dir runs/semantic_pilot_001
```

The local pilot fixes model seeds `3, 7, 11`, data seed `7`, four declared
block/cylinder left/right cases, and rollout initial-condition seeds `7, 13, 29`.
Each model receives the same synthetic input split. Each model trains for 10
stage-1 and 10 stage-2 steps; each rollout permits 10 policy steps. Inputs,
configuration, and budget are frozen before optimization. There is no automatic
retry or result-dependent budget extension.

This is 36 scheduled simulation trials across three model initializations,
four task declarations, and three initial conditions. Four labels still cover
only two physical object profiles. They are not four independently established
semantic task families. The budget is a bounded diagnostic, not an adequacy
claim. Failed or interrupted slots must not produce a complete aggregate;
structured execution failures remain in trial denominators.

The existing two-case, one-step robot-free semantic demo is unchanged. Mappings
remain `unverified`; independent semantic mapping, prompt causality, real-world
success, and official Zero-WAM reproduction are not claimed. Meaningful
zero-shot evaluation still requires provenance-backed diverse demonstrations
and independent mapping review not supplied by this synthetic pilot.

## Recorded checks (2026-09-08)

The full suite passed: 751 tests, zero skips. Ruff, source mypy (59 files),
compileall, and dependency checks passed. Final-checkpoint replay matched all
432 stage-2 windows across the original nine conditions. All 95 original study
files remained unchanged. Replay output is `runs/decoder_replay_001.json`.

The single fixed pilot at `runs/semantic_pilot_001` completed all 36 scheduled
trials: 36 scored, zero execution failures, and **zero task successes**. Each
model seed had 0/12 successes. All trials failed the object-position tolerance
criterion; mean position error was 0.123019 m. No retries or budget changes were
made. This is a negative result for this bounded synthetic protocol, not proof
that zero-shot transfer is impossible. Task mappings remain unverified.

The final report is `runs/semantic_pilot_001/semantic-pilot-report.json`, SHA-256
`e5506a23978760e5d66005af78f2375f449f4386f80e167e0c662fd6846c573c`.
These run artifacts are local and ignored by Git. Code/test completion does not
establish task success; diverse, provenance-backed data and independent mapping
review remain missing.
