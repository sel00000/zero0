# G10 deployment certification

## What this gate does

`so101-wam-certify` is the project-provided path that promotes an offline
`compact_wam_candidate` into a `compact_wam_deployment`. It derives every
authorization digest from files, copies the exact candidate model state into a new
checkpoint, and publishes that checkpoint with an external certification JSON using
race-safe no-overwrite links. If the second publish or final verification fails
during the running process, it removes the first output. A host crash between the
two links can leave an orphan checkpoint, which remains unusable without its matching
certification. It never opens a camera or motor bus.

The result authorizes only a bounded `G10_dry_rollout`. It does not certify task
success, generalization, collision safety, or G11 production use. The current
repository contains no real preflight, prompt, manual signoff, certified deployment
checkpoint, or task-success report, so nothing is presently authorized for real
output.

## Required immutable evidence

The issuer requires all of the following exact files:

1. An offline candidate checkpoint and its task-disjoint training report. The report
   must bind the checkpoint SHA-256 and architecture; candidate metadata remains
   `trained=false`, `offline_trained=true`, and `deployment_ready=false`.
2. A real observation-only G6/G7 preflight JSON with all automated checks passing,
   zero goal-position commands, valid calibration/home measurements, and the exact
   hardware/config identity. Its `result` remains `partial` because human checks are
   separate.
3. A G8 MuJoCo checkpoint-session JSON generated with `policy="compact_wam"`, the
   exact candidate SHA-256/checkpoint ID, the exact G9 prompt fingerprint, collision
   rejection enabled, and at least one simulated action.
4. A real G9 kinesthetic prompt `.npz` and same-stem manifest with zero goal-position
   commands, torque-off capture provenance, checksum validation, and replayable
   prompt fingerprint.
5. A human signoff JSON bound to the SHA-256 of every preceding artifact and the full
   deployment config SHA-256. Every required real-hardware check must be exactly
   `true`.

The issuer treats evidence files and final outputs as immutable and will not
overwrite an existing output. Re-running any stage requires a new filename/checkpoint
ID. Preserve the source tree with access controls because an operating-system user
who can rewrite those files is outside this no-overwrite guarantee.

## Produce the G8 candidate report

Use the exact candidate and exact G9 prompt intended for certification:

```shell
PYTHONPATH=src python -m so101_wam.mujoco_cli \
  --config configs/mujoco.toml \
  --checkpoint checkpoints/compact_wam_candidate_001.pt \
  --prompt data/prompts/prompt_000001.npz \
  --manifest data/prompts/prompt_000001.json \
  --steps 20 \
  --report reports/g8_candidate_001.json
```

A wrist-roll smoke report is deliberately rejected because it is not evidence for
the candidate checkpoint.

The report also binds the validated MuJoCo config digest. Certification and real
runtime re-run the reported number of policy steps on CPU with the exact candidate,
prompt pair, and MuJoCo config, then require the complete regenerated report to equal
the stored JSON. The prompt and simulated live wrist images must use one resolution;
the provided MuJoCo and hardware templates both use 640×480. Therefore the issuing
and real-output hosts need the `sim` extra and a working GL backend; missing MuJoCo or
render support fails before robot construction.

## Create the manual signoff

Copy `configs/manual_signoff.template.json` to a new report path. Fill the real
hardware/calibration/operator IDs and a timezone-aware `signed_at_utc`. Calculate the
six source hashes from the exact files:

```shell
sha256sum \
  checkpoints/compact_wam_candidate_001.pt \
  reports/compact_wam_candidate_001.training.json \
  reports/g6_g7_preflight.json \
  reports/g8_candidate_001.json \
  data/prompts/prompt_000001.npz \
  data/prompts/prompt_000001.json
```

Calculate the validated hardware config digest with project code:

```shell
PYTHONPATH=src python - <<'PY'
from so101_wam.config import ProjectConfig
from so101_wam.deployment import project_config_sha256

config = ProjectConfig.load("configs/lerobot_hardware.local.toml")
print(project_config_sha256(config))
PY
```

Set each check to `true` only after observing it on the named hardware/calibration:

- left/right wrist camera identity;
- collision-free arms-down pose;
- left/right and per-joint direction labels;
- measured per-servo-tick delta limits;
- operator halt stopping subsequent commands;
- cable strain relief;
- arm/torso/table/camera clearance;
- contact-free real slow jog.

The project does not require an external E-stop. The operator-halt item instead
attests that the configured software/session halt path stopped further commands.
This JSON is an operator attestation, not a cryptographic signature or tamper-proof
hardware attestation; preserve it and the source artifacts in access-controlled
storage.

## Issue the immutable deployment pair

The hardware config used here must already contain the measured profile with
`backend="lerobot"`, `actuation_enabled=true`, and `safety.calibrated=true`.

```shell
PYTHONPATH=src python -m so101_wam.certify_cli \
  --config configs/lerobot_hardware.local.toml \
  --candidate checkpoints/compact_wam_candidate_001.pt \
  --training-report reports/compact_wam_candidate_001.training.json \
  --preflight-report reports/g6_g7_preflight.json \
  --mujoco-config configs/mujoco.toml \
  --mujoco-report reports/g8_candidate_001.json \
  --prompt data/prompts/prompt_000001.npz \
  --manifest data/prompts/prompt_000001.json \
  --manual-signoff reports/manual_signoff_001.json \
  --output checkpoints/compact_wam_deployment_001.pt \
  --certification reports/compact_wam_deployment_001.certification.json \
  --checkpoint-id compact-wam-deployment-001 \
  --max-policy-steps 1
```

Installed environments may use `so101-wam-certify`. The issuer validates all
sources, re-hashes them immediately before publication to detect mid-run changes,
verifies that output weights exactly equal candidate weights, verifies the final
checkpoint/certification pair, and rolls back the first output if the second cannot
be published.

It also recomputes the preflight FPS/latency/frame-age/skew/home thresholds, G8
physics-step accounting, and G9 capture/camera timing thresholds from stored raw
fields instead of trusting only top-level `pass` booleans.

The certification schema is `2`. It includes hashes for the candidate, training
report, preflight report, MuJoCo report, prompt NPZ, prompt manifest, and manual
signoff. The certification core digest is also embedded in checkpoint metadata, and
the certification separately binds the final checkpoint byte hash.

## Run the bounded G10 dry rollout

Only after successful issuance, use the same config, prompt, and maximum-or-lower
step count:

```shell
PYTHONPATH=src python -m so101_wam.hardware_cli \
  --config configs/lerobot_hardware.local.toml \
  --checkpoint checkpoints/compact_wam_deployment_001.pt \
  --prompt data/prompts/prompt_000001.npz \
  --manifest data/prompts/prompt_000001.json \
  --deployment-certification reports/compact_wam_deployment_001.certification.json \
  --source-candidate checkpoints/compact_wam_candidate_001.pt \
  --source-training-report reports/compact_wam_candidate_001.training.json \
  --source-preflight-report reports/g6_g7_preflight.json \
  --source-mujoco-config configs/mujoco.toml \
  --source-mujoco-report reports/g8_candidate_001.json \
  --source-manual-signoff reports/manual_signoff_001.json \
  --steps 1 \
  --enable-real-output \
  --ack-hardware-id fixed-torso-a
```

The hardware CLI re-hashes the deployment checkpoint and runtime prompt before and
after strict loading. It then revalidates every named G6–G9/manual-signoff source,
re-runs the exact G8 MuJoCo session, requires every loaded deployment state tensor to
equal the revalidated source candidate, and matches the resulting source digests
against the certification before constructing the robot. A different config,
hardware ID, calibration ID, checkpoint, model weight, source artifact, prompt,
simulation result, or step count fails before real output.

This is an accidental-misconfiguration and auditable-evidence boundary, not a
cryptographic trust root. An attacker who can replace the program, certification,
and every source artifact can fabricate a coherent set. Use detached signatures or
an external artifact registry if protection from that threat is required.
