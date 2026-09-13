# Compatible human/robot pair manifests

This robot-free boundary binds one human task video to one checksum-validated
local robot episode. It is deliberately named
`compatible_human_robot_pairs_not_official_humangen`.

Zero-WAM describes HumanGen as semantically matched human-video/robot-trajectory
pairs while retaining the robot trajectory as executable supervision
([paper §2](https://arxiv.org/html/2608.26103v2)). The official HumanGen asset
schema is not established by that paper or the project page. This repository
therefore validates only declared task identity, bytes, provenance, and split
isolation. It does not infer semantic equivalence or claim task success.

LeRobotDataset v3 stores tabular state/action/timestamps in Parquet, camera video
in MP4, and dataset metadata separately
([v3 documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)).
The local reader consumes robot episodes already converted to this repository's
NPZ/JSON format plus a human-video file. It does not read v3 Parquet/video shards
or their offsets, so `source_format` is
`lerobot_dataset_v3_compatible_manifest`, not full LeRobot v3 compatibility.

## Manifest

All file paths are normalized, bundle-relative, and confined beneath the
manifest directory. Both the human video and robot NPZ have explicit SHA-256
bindings; the robot episode's same-stem JSON manifest is also verified by the
existing dataset loader.

```json
{
  "schema_version": "so101_wam.paired_data.v1",
  "artifact_kind": "compatible_human_robot_pairs_not_official_humangen",
  "source_format": "lerobot_dataset_v3_compatible_manifest",
  "source_url": "hf://datasets/example/repo",
  "pairs": [
    {
      "pair_id": "pair-0001",
      "task": "place block",
      "task_index": 9,
      "semantic_match": "unverified",
      "robot_episode": "episodes/episode_000003.npz",
      "robot_episode_sha256": "<64 lowercase hex characters>",
      "human_video": {
        "path": "human/place_block.mp4",
        "sha256": "<64 lowercase hex characters>",
        "view": "third_person",
        "task_spec": {
          "format": "so101_wam.human_video_prompt_npz.v1",
          "path": "human/place_block.task_spec.npz",
          "sha256": "<64 lowercase hex characters>"
        }
      },
      "provenance": {
        "source_dataset": "example/repo",
        "source_layout": "meta/info.json,data/,videos/",
        "license": "apache-2.0",
        "transformation": "robot episode converted to local NPZ; human MP4 copied byte-for-byte"
      }
    }
  ]
}
```

`semantic_match` is an enum, never a boolean. Use `unverified` unless a human
actually reviewed the task correspondence; the other accepted value is
`human_reviewed`. The loader never upgrades this status.

`human_video.task_spec` is optional. It points to a derived NPZ containing
exactly two arrays: `timestamp` with shape `[T]` and `rgb` with uint8 shape
`[T, H, W, 3]`. Any extra key, including action, joint, or proprioception, is
rejected. The manifest loader checks its SHA-256 and the task-spec bridge checks
the bytes again immediately before loading. Frame extraction itself is outside
this repository and must be described in `provenance.transformation`.

## Validate and audit

```shell
PYTHONPATH=src python -m so101_wam.paired_data_cli \
  --manifest data/pairs/pairs.json
```

Installed environments may use `so101-wam-paired-data`. The JSON audit records
pair/task counts, declared task-spec artifact count, review status, pair and
source hashes, and explicit negative flags for official HumanGen, a full
LeRobot v3 reader, and evaluated human-video task success. The declared count
only confirms the manifest format, path, and checksum; bridge validation
determines whether the NPZ is usable.

`validate_pair_disjoint_split()` rejects overlapping pair IDs/fingerprints,
human-video hashes, robot-episode fingerprints, task indices, and task labels.
`paired_manifest_as_json()` provides a canonical bundle-relative round-trip.

`paired_episode_records()` can expose the robot trajectories to the current
trainer after pair validation. The primary `train_offline_candidate()` workflow
still uses robot sensorimotor prompts.

When the optional task-spec artifact exists,
`human_prompt_from_pair()` in `paired_task_specs.py` converts it to the existing
`HumanVideoPrompt`. A focused test runs that prompt through
`CompactWAMPolicy.predict_task()` and verifies a finite action chunk. This is an
interface smoke with random compact weights, not learned human-video following.

## Train an offline paired candidate

`paired_training.py` accepts only records declared
`semantic_match="human_reviewed"`, rechecks the source video, robot episode, and
extracted task-spec checksums, and uses the robot episode only for live context
and future/action supervision. Human prompt proprioception and action inputs use
the train split's neutral axis mean; the single human view is duplicated into
the compact model's two view slots.

Training requires at least two train pairs and at least two task-disjoint
validation tasks. It draws train pairs with a task-balanced schedule, optimizes
the inverse-dynamics head followed by the full compact model, and immutably
publishes one checkpoint/report pair.

```shell
PYTHONPATH=src python -m so101_wam.paired_train_cli \
  --train-manifest data/pairs/train.json \
  --validation-manifest data/pairs/validation.json \
  --output checkpoints/paired_candidate_001.pt \
  --report reports/paired_candidate_001.training.json \
  --checkpoint-id paired-candidate-001
```

Installed environments may use `so101-wam-train-paired`. The report binds every
pair, human video, human task-spec, robot episode, split digest, sampling
schedule, model architecture, and exact checkpoint hash. Held-out metrics cover
matched, null, and different-task prompt action/future-latent deltas.

These are offline proxy metrics. The checkpoint deliberately stores
`offline_trained=true`, `trained=false`, and `deployment_ready=false`; the report
keeps semantic success, prompt causality, real-world success, and official
Zero-WAM claims false. `human_reviewed` remains an imported declaration rather
than independently verified semantic evidence.
