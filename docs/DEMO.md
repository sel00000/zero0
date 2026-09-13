# Demo Verification Media

The demo media is generated from actual MuJoCo rollouts of the local
CompactWAM checkpoint on the known `reference-nudge-block` diagnostic. It is
not a concept animation and not a scripted success reel.

Reproduce from bundled inputs:

```shell
.venv/bin/python scripts/verify_demo_policy.py \
  --run-dir runs/zero01_verification_001
```

Re-render media from an existing run without re-running the policy:

```shell
.venv/bin/python scripts/verify_demo_policy.py \
  --replay-from runs/zero01_verification_001
```

The published media replays the original local records in
`runs/zero101_verification_001`; that historical directory retains its name.

Requirements: the project virtualenv must include Pillow, NumPy, Torch, and
MuJoCo, and `ffmpeg`/`ffprobe` must be available on `PATH`.

The published policy run used Python 3.12.3, PyTorch 2.11.0+cpu,
NumPy 2.2.6, MuJoCo 3.12.0, and one OpenMP/MKL thread. Run with
`MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` in a headless environment.
The exact trajectory-match results refer to replays in this environment.

Bundled inputs:

- `docs/assets/demo/inputs/reference-nudge-candidate.pt`
- `docs/assets/demo/inputs/reference-nudge-training-report.json`
- `docs/assets/demo/inputs/seed-7-reference-prompt.npz`
- `docs/assets/demo/inputs/seed-7-reference-prompt.json`
- `docs/assets/demo/inputs/seed-13-reference-prompt.npz`
- `docs/assets/demo/inputs/seed-13-reference-prompt.json`

Outputs:

- `docs/assets/demo/zero01-demo.mp4`
- `docs/assets/demo/zero01-demo.gif`
- `docs/assets/demo/zero01-preview.png`
- `docs/assets/demo/zero01-demo.json`

Current result, generated with the current fixed source tree:

- Reference controller: seed 7 and seed 13 both succeed.
- Stationary hold control: seed 7 and seed 13 both fail the object-position
  tolerance.
- Learned checkpoint: seed 7 fails the object-position tolerance; seed 13
  succeeds.

Scope:

- Simulation only; no hardware evidence.
- Same-task diagnostic only; no zero-shot claim.
- Requested household tasks are not validated by this media because their
  scenes, prompts, and task-specific scorers are absent. Proposed evaluation
  criteria are recorded in [TASK_VERIFICATION.md](TASK_VERIFICATION.md).
- Scores and outcomes come from the original rollout reports in this run.
- Older recovery reports that showed two learned safety rejections are historical
  outputs from earlier code; this demo does not rewrite them.
- The bundled checkpoint is a limited IFP0/legacy-mean diagnostic, not a full
  architecture reproduction claim.
- Its metadata records `offline_trained=true` after 1,000 optimizer steps;
  `trained=false` and `deployment_ready=false` record that deployment-level
  capability has not been established.
- Video is a high-resolution MuJoCo replay of the recorded accepted action
  arrays from those episodes. The JSON records replay trajectory/final-position
  match checks against the original rollout trajectory.
