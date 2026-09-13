# IFP K=0/2/4 ablation

This workflow compares the compact model with `K=0`, `K=2`, and `K=4`
training-only in-context future prediction (IFP). It runs without a robot and
publishes descriptive offline and MuJoCo evidence. It does not certify
open-ended task generalization or real output.

The design follows the structural boundary in the
[Zero-WAM paper](https://arxiv.org/html/2608.26103v2#S3.SS3): `K` parallel
future modules consume a fused representation from the main video branch, and
the modules are removed at inference. The paper uses `K=4`, stride 2, and loss
weights `(0.5, 0.25, 0.15, 0.15)`.

## Local reduction

The local fused path is selected with `ifp_architecture=fused_modules`:

- every temporal layer contributes its final live-state representation;
- an MLP fuses those representations;
- each future module is a separate copy initialized from the last main temporal
  layer;
- each module predicts one strided dual-wrist latent target in parallel;
- the auxiliary modules are never stored in the candidate checkpoint.

This remains a compact latent-MSE approximation. It is not Wan video
flow-matching, and its fused state is much smaller than the paper's token-level
multi-layer representation.

## Fair comparison contract

All three variants reserve the `K=4` target horizon, then use the same:

- train and validation split hashes;
- initial main-model state hash;
- shuffled training-window schedule hash;
- train/validation window counts;
- seed and optimizer-step budget;
- closed-loop protocol digest, trial count, and terminal-error basis.

`K=0` ignores the reserved IFP targets, `K=2` uses the first two, and `K=4`
uses all four. The report is rejected if any comparability field differs.

`closed_loop.protocol_sha256` hashes evaluation inputs, not results or model
weights. Terminal mode binds the config, prompt/manifest, target, tolerance,
policy-step count, and device. Semantic mode binds suite/mapping/config hashes,
each case's prompt/manifest/task/physical-profile hashes, and device. The
ablation and study share the result validator in `ifp_results.py`.

Both modes also bind `mujoco_model_identity`: the loaded compiled-model MJB
SHA-256, byte count, and MuJoCo engine version. This includes compiled scene and
asset content, not just a model filename or selected object profile. It is an
engine/build-specific fingerprint, not a cross-version canonical format or a
GPU-driver attestation.

Reports now use `so101_wam.ifp_ablation.v4` and `so101_wam.ifp_study.v4`.
The study rejects older reports: their missing compiled-model binding cannot be
reconstructed from outcome summaries. Rerun the evaluation; do not relabel an
old report or invent its digest.

## Run the terminal-joint proxy

```shell
PYTHONPATH=src python -m so101_wam.ifp_ablation \
  --train-episodes data/train \
  --validation-episodes data/validation \
  --output-dir reports/ifp-ablation-001 \
  --checkpoint-id-prefix ifp-ablation-001 \
  --mujoco-config configs/mujoco_robot_free.toml \
  --closed-loop-mode terminal-joint-proxy \
  --terminal-tolerance 5.0 \
  --policy-steps 10 \
  --stage1-steps 1000 \
  --stage2-steps 5000
```

Installed environments may use `so101-wam-ifp-ablation`. Output paths are
immutable. The terminal tolerance must be chosen before the run from the
synthetic task's native joint-unit acceptance criterion; it must not be tuned
after seeing variant outputs.

## Run the semantic object-state suite

Use the same predeclared suite and mapping for every `K` variant:

```shell
PYTHONPATH=src python -m so101_wam.ifp_ablation \
  --train-episodes data/train \
  --validation-episodes data/validation \
  --output-dir reports/ifp-semantic-001 \
  --checkpoint-id-prefix ifp-semantic-001 \
  --mujoco-config configs/mujoco_robot_free.toml \
  --closed-loop-mode semantic-suite-proxy \
  --semantic-suite runs/semantic_suite/suite.json \
  --semantic-mapping runs/semantic_suite/mapping.json \
  --stage1-steps 1000 \
  --stage2-steps 5000
```

Each variant runs the existing multi-case MuJoCo semantic suite. The ablation
report preserves total, scored, and execution-failure counts, the failure
taxonomy, total-trial success rate, and mean object-position error over scored
trials. The suite report and its SHA-256 are bound into the variant evidence.
Metrics are read from the same persisted bytes that supply that hash. A complete
suite with no scored trials is retained: all trials remain execution failures,
success rate is zero, and terminal error is `null`, not zero.
Semantic benchmark/suite reports use schema 5. Each trial labels model identity
as `rollout_session` or `post_failure_inspection`. The latter is an inspection of
the expected scene after failure, not proof of a completed rollout. Sources are
retained in summaries but excluded from the result-independent protocol digest.
Different compiled model identities across trials or cases are rejected.
The mapping remains local and is not independently verified.

## Aggregate several seeds

Place at least three completed seed reports below one study directory, then
aggregate them without rerunning training or MuJoCo:

```shell
so101-wam-ifp-study \
  --reports \
    reports/ifp-study/seed-3/ifp_ablation.json \
    reports/ifp-study/seed-7/ifp_ablation.json \
    reports/ifp-study/seed-11/ifp_ablation.json \
  --preregistration reports/ifp-study/preregistration.json \
  --report reports/ifp-study/ifp_study.json
```

Source reports must stay below the output report's parent directory. The
command requires unique seeds, the exact `K=0/2/4` variant set, and identical declared
data, optimizer, metric-schema, and closed-loop protocols. It records each
source report's SHA-256, reports mean/sample standard deviation/range for each
validation metric, pools closed-loop success counts, and weights terminal error
by scored-trial count. Execution failures stay in the success-rate denominator
but do not fabricate terminal object error. If a variant has no scored trials
across all seeds, its aggregate terminal error stays `null`. The output is
immutable.

`--preregistration` is optional. Copy
`configs/ifp_study_preregistration.template.json` into the study bundle and fill
it before inspecting result metrics. The strict plan contains only pre-result
values: exact seeds, split hashes, `K=0/2/4`, optimizer protocol, closed-loop
scope/trial floor, endpoints, and margins. Source-report, checkpoint, MuJoCo,
loss, metric, and winner values are not accepted in the plan.
This v1 plan does not bind the full evaluation-protocol digest; digest equality
is checked across source reports, not against the preregistration.

With a matching plan, the report records its SHA-256 and evaluates two fixed
`K=4` versus `K=0` deltas:

- `success_rate(K4) - success_rate(K0)` must meet the declared minimum;
- `terminal_error_mean(K4) - terminal_error_mean(K0)` must stay at or below the
  declared maximum.

An endpoint miss remains `result="complete"` with
`endpoint_evaluation.passed=false`; it is not converted into an execution
error or a forced winner. The report also fixes
`external_preregistration_timing_verified=false`. Without a plan, the existing
descriptive `scientific_claim="not_evaluated"` output remains unchanged.
If either variant has no terminal metric, the secondary delta is `null`,
`secondary_endpoint_available=false`, and the secondary/overall endpoint checks
fail. Missing or malformed fields are still rejected.

Neither mode reruns or validates nested checkpoint and MuJoCo artifacts or
proves that the optimization budget was adequate. Plan mode establishes only a
local, hash-bound diagnostic contract; it does not establish external
preregistration timing or select a winning `K`.

## Evidence and interpretation

Each variant publishes a checkpoint, training report, and selected MuJoCo
evaluation report. The combined report includes:

- normalized action MSE and native-unit action MAE;
- future-latent MSE;
- mean absolute action/future change under null and temporally reversed prompts;
- either one terminal joint-target proxy or a multi-case semantic object-state
  suite result, with scored and execution-failure counts;
- proof that the exported checkpoint has `ifp_steps=0` and no IFP parameters.

The prompt deltas are sensitivity diagnostics, not causal evidence. The
terminal mode is a synthetic joint-space proxy. The semantic mode evaluates
declared simulator object-state criteria, but does not establish independent
task mapping, held-out generalization, or real-world success. The local plan can
predeclare thresholds and several seeds, but adequate optimization still
requires an actual experiment. This command records negative or flat outcomes
rather than forcing a pass ordering.
