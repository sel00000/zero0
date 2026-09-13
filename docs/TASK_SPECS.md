# Offline task-spec interfaces

[Zero-WAM](https://arxiv.org/abs/2608.26103) treats either human video or
language as a task instruction within one policy. The local compact model now
exposes that interface without widening the robot `PhysicalPrompt` schema. This
is an offline structural smoke, not an official model reproduction.

## Contracts

| Type | Allowed source fields | Model adapter |
| --- | --- | --- |
| `RobotEpisodePrompt` | existing wrist RGB, joint state, executed action | unchanged `PhysicalPrompt` tensors |
| `HumanVideoPrompt` | one RGB frame and timestamp per step; optional text metadata | RGB duplicated into the compact two-view slots; prompt axes fixed to the checkpoint normalization mean |
| `LanguagePrompt` | bounded UTF-8 text only | black visual carrier, neutral prompt axes, deterministic text conditioning vector |

Every type carries `TaskSpecProvenance`. A source ID is required; a lowercase
SHA-256 may also be recorded. The schema validates its format but cannot verify
an external file it was not given. Fingerprints cover task content and exclude
source labels so duplicate content remains detectable across provenance records.

The paired-data bridge is the checksum-verifying exception to that last
limitation. A compatible pair manifest may reference a derived NPZ with exactly
`timestamp` and `rgb`; `human_prompt_from_pair()` verifies the file again and
constructs `HumanVideoPrompt`. See [PAIRED_DATA.md](PAIRED_DATA.md). The bridge
does not decode the source MP4. A separate bounded adapter can materialize a
declared human-reviewed pair for one real optimizer-step regression.

`HumanVideoFrame` has no joint, proprioception, or action field. Human and live
frames must have the same resolution in this compact adapter. It inserts neutral
model inputs only after schema validation. Optional human-video text metadata is
fingerprinted but is not used as conditioning.

## Policy boundary

`CompactWAMPolicy.predict_task(task_spec, live_frames, now_s=...)` accepts exactly
one robot episode, human video, or language task specification. Existing
`predict(ContextSnapshot, ...)` behavior and checkpoint state dictionaries are
unchanged. `SO101WAMRuntime` still uses only `PhysicalPrompt`; the task-spec
method is not wired into the real-output runtime.

Human RGB enters the existing causal visual prefix. Language uses a dependency-free
UTF-8 byte tokenizer and fixed deterministic encoder, then adds that vector to
the compact context state. This only proves that the tokenizer/encoder boundary
and shared inference call are executable. The encoder has no learned language
semantics, and the primary candidate trainer has not produced a checkpoint
trained on these modalities.

## Verification

```shell
pytest -q tests/test_task_specs.py tests/test_paired_training.py
```

The tests lock schema-level robot-signal exclusion, timestamp/image validation,
stable provenance and fingerprints, neutral adapter tensors, exact robot-path
parity, finite matched/mismatched human and language inference, and invalid
conditioning rejection. The paired-training regression also verifies a real
optimizer update and a nonzero gradient to action-free human prompt pixels.

Passing these tests is not human-video following, language understanding,
terminal task success, causality, or real-robot evidence. Those claims require
multi-pair checkpoint training and held-out closed-loop evaluation.
