# Offline prompt controls

이 진단은 하나의 CompactWAM checkpoint와 하나의 live context를 고정하고 prompt만
바꿔 action chunk 차이를 기록한다. 로봇과 MuJoCo를 사용하지 않는다.

## 입력

[`configs/prompt_controls.template.json`](../configs/prompt_controls.template.json)을
새 bundle에 복사한다. 모든 경로는 manifest 위치 기준의 정규화된 상대 경로여야
하며 bundle 밖으로 나갈 수 없다.

필요한 episode는 네 개다.

- `live_episode`, `matched_prompt`, `same_task_alternate_prompt`: 같은 task와
  `task_index`, 서로 다른 원본 내용
- `wrong_task_prompt`: task와 `task_index`가 모두 다름
- 네 episode: 동일 해상도와 FPS, config safety limits 안의 joint/action

## 실행

```shell
so101-wam-prompt-controls \
  --manifest runs/prompt_control_001/manifest.json \
  --artifact-dir runs/prompt_control_001/artifacts \
  --report runs/prompt_control_001/report.json \
  --device cpu
```

기존 report나 condition artifact는 덮어쓰지 않는다.

## 조건

| 조건 | 변경 |
| --- | --- |
| `matched` | 지정한 matched prompt |
| `same_task_alternate` | 같은 task의 다른 episode |
| `wrong_task` | 다른 task의 episode |
| `temporal_shuffle` | matched frame payload 순서 |
| `image_frame_shuffle` | matched image 순서만 |
| `null` | image와 sensorimotor 값을 0으로 대체 |
| `counterfactual` | image는 유지하고 joint/action을 safety 범위 안에서 반전 |

각 조건은 matched action 대비 L2, 평균 절대차, 최대 절대차, 축별 최대 절대차를
남긴다. matched 조건은 두 번 실행해 반복성도 기록한다. report에는 입력 episode의
상대 경로, 파일 SHA-256, episode fingerprint, content SHA-256이 포함된다.

`result="complete"`는 artifact 생성 완료만 뜻한다. 현재 임계값을 정의하지 않으므로
과학적 pass/fail, terminal task success, prompt 인과성, 실제 로봇 성공을 주장하지
않는다.

## MuJoCo joint proxy 연결

동일 checkpoint와 동일 benchmark task/seed를 7개 prompt 조건에 반복 적용하려면
별도 simulation 진단을 실행한다.

```shell
so101-wam-mujoco-prompt-controls \
  --config configs/mujoco_robot_free.toml \
  --prompt-control-manifest runs/prompt_control_001/manifest.json \
  --benchmark-manifest configs/mujoco_heldout_benchmark.json \
  --artifact-dir runs/prompt_control_001/mujoco_artifacts \
  --report runs/prompt_control_001/mujoco_report.json \
  --device cpu
```

source config와 MuJoCo config의 camera/policy/servo rate, context 길이, action
horizon, primary camera 계약은 같아야 한다. 입력 episode의 joint/action도 MuJoCo
safety 범위 안에 있어야 한다. 파생 prompt는 원본 recording rate에서 생성해 condition별
immutable NPZ/manifest로 남긴 뒤 기존 CompactWAM checkpoint benchmark에 전달한다.
각 condition은 같은 checkpoint SHA/ID, benchmark manifest, task/seed schedule을
사용하며 trial artifact와 condition benchmark report의 hash를 상위 report가 다시
검증한다.

선택적으로 `--direction-preregistration`을 넘기면 실행 전에 고정한 방향성 규칙을
판정한다. [`configs/prompt_direction_preregistration.template.json`](../configs/prompt_direction_preregistration.template.json)을
복사한 뒤 `suite_id`와 네 입력 SHA-256을 실제 값으로 바꾼다. 추가 필드나 다른
endpoint는 허용하지 않는다.

```json
{
  "schema_version": 1,
  "suite_id": "prompt-control-local-v1",
  "hypothesis_id": "joint-proxy-direction-v1",
  "claim_scope": "preregistered_mujoco_joint_proxy_directionality_only",
  "prompt_control_manifest_sha256": "<sha256>",
  "benchmark_manifest_sha256": "<sha256>",
  "checkpoint_sha256": "<sha256>",
  "mujoco_config_sha256": "<project-config-sha256>",
  "minimum_trials_per_condition": 3,
  "primary_endpoint": "success_rate",
  "secondary_endpoint": "final_error_mean",
  "same_task_noninferiority_margin": 0.34,
  "negative_success_margin": 0.34,
  "negative_error_margin": 0.05,
  "negative_conditions": ["wrong_task", "null", "counterfactual"],
  "descriptive_conditions": ["temporal_shuffle", "image_frame_shuffle"]
}
```

판정 규칙은 네 개로 고정된다.

1. 모든 condition의 trial 수가 최소값 이상이다.
2. matched 성공률은 same-task alternate보다 비열등성 margin 이상 뒤처지지 않는다.
3. matched 성공률은 세 negative condition 중 최고 성공률보다 지정 margin 이상 높다.
4. 세 negative condition 중 최저 평균 final error도 matched보다 지정 margin 이상 높다.

shuffle 두 조건은 기술 통계로만 남긴다. 상위 `result="complete"`는 계속 실행 완료만
뜻하고, `directionality_evaluation.result`가 joint-proxy 가설의 `pass`/`fail`을
보존한다. 입력 manifest 원본은 artifact 안에 복사되고 SHA-256으로 다시 묶인다.
`preregistration_level="local_input_hash_bound_before_rollout"`이며 외부 공개 등록이나
독립 timestamp는 검증하지 않는다.

상위 `result="complete"`는 모든 condition 실행과 evidence 기록이 끝났다는 뜻이다.
condition별 joint-target tolerance 결과와 matched 대비 success-rate/final-error 차이는
기술 통계다. 이 경로는 MuJoCo terminal joint proxy를 평가하지만 recorded
`live_episode` 대신 MuJoCo live context를 사용한다. checkpoint training split,
prompt-task semantic match는 검증하지 않는다. 선택적 방향성 판정도 semantic held-out
success, prompt causality, 통계적 유의성 또는 실제 로봇 성공을 주장하지 않는다.

## MuJoCo 객체 상태 연결

같은 7개 조건을 joint target 대신 semantic manifest가 선택한 named free body의 3D
position criterion에 적용하려면 별도 semantic 진단을 실행한다.

```shell
so101-wam-mujoco-semantic-prompt-controls \
  --config configs/mujoco_robot_free.toml \
  --prompt-control-manifest runs/prompt_control_001/manifest.json \
  --semantic-manifest configs/mujoco_semantic_benchmark.json \
  --training-report checkpoints/compact_wam_candidate_001.training.json \
  --direction-preregistration runs/prompt_control_001/semantic_direction.json \
  --artifact-dir runs/prompt_control_001/semantic_artifacts \
  --report runs/prompt_control_001/semantic_report.json \
  --device cpu
```

checkpoint, training report, semantic manifest와 seed schedule은 모든 조건에서 같다.
matched, alternate, shuffle, null, counterfactual 조건은 manifest의 dataset task
label/index와 일치해야 한다. `wrong_task`만 `expected_mismatch` 대조군으로 명시해
label과 index가 모두 다른지 검증한다. 단일 semantic benchmark의 기본 동작은 계속
task mismatch를 rollout 전에 거부한다.

condition별 success rate, object-position error와 matched 대비 차이를 immutable
prompt/trial/report hash에 묶는다. schema-4 child와 상위 report는 모든 조건이 같은
`object_body`와 이름을 제외한 compiled physical-profile SHA-256을 사용했는지도
재검증한다. `result="complete"`는 7조건 실행 완료일 뿐이다. 기존 방향성 plan은
joint-proxy 범위라 이 경로에 재사용하지 않는다.

semantic child의 schema-3 `execution_failure`는 condition 전체를 중단하지 않는다.
condition/report summary는 `scored_trial_count`, `execution_failure_count`를 보존하고
terminal metric이 없으면 object-position 평균과 matched 대비 차이를 `null`로 둔다.
모든 condition이 미채점이면 `semantic_mujoco_object_state_evaluated=false`다.

선택적으로 `--direction-preregistration`을 넘기면 별도 object-state plan을 rollout 전에
판정한다. [`configs/semantic_prompt_direction_preregistration.template.json`](../configs/semantic_prompt_direction_preregistration.template.json)을
복사한 뒤 `suite_id`와 다섯 입력 SHA-256을 실제 값으로 바꾼다.

```json
{
  "schema_version": 1,
  "suite_id": "semantic-prompt-control-local-v1",
  "hypothesis_id": "object-state-direction-v1",
  "claim_scope": "preregistered_mujoco_object_state_directionality_only",
  "prompt_control_manifest_sha256": "<sha256>",
  "semantic_manifest_sha256": "<sha256>",
  "training_report_sha256": "<sha256>",
  "checkpoint_sha256": "<sha256>",
  "mujoco_config_sha256": "<project-config-sha256>",
  "minimum_trials_per_condition": 3,
  "primary_endpoint": "success_rate",
  "secondary_endpoint": "object_position_error_mean_m",
  "same_task_noninferiority_margin": 0.34,
  "negative_success_margin": 0.34,
  "negative_object_position_error_margin_m": 0.05,
  "negative_conditions": ["wrong_task", "null", "counterfactual"],
  "descriptive_conditions": ["temporal_shuffle", "image_frame_shuffle"]
}
```

규칙은 네 개다.

1. 모든 condition의 trial 수가 최소값 이상이다.
2. matched 성공률은 same-task alternate보다 비열등성 margin 이상 뒤처지지 않는다.
3. matched 성공률은 wrong/null/counterfactual 중 최고 성공률보다 margin 이상 높다.
4. wrong/null/counterfactual 중 가장 가까운 평균 object-position error도 matched보다
   margin 이상 높다.

temporal/image shuffle 조건은 기술 통계로만 남긴다. 상위 `result="complete"`는 계속
실행 완료이고, `directionality_evaluation.result`가 object-state 가설의 `pass`/`fail`을
담는다. plan 원본은 artifact 안에 복사되고
`preregistration_level="local_input_hash_bound_before_rollout"`,
`external_preregistration_timestamp_verified=false`를 기록한다.
plan이 없으면 `directionality_preregistered=false`,
`scientific_pass_fail_evaluated=false`를 기록한다.
preregistered secondary endpoint에 필요한 terminal metric이 하나라도 없으면 해당
object-position margin 규칙은 `endpoint_available=false`, `result="fail"`이며 observed
margin은 `null`이다. 실행 자체를 hard error로 바꾸지는 않는다.

dataset task와 object task의 의미 대응은 manifest 선언에 한정된다. 따라서 이 결과는
prompt causality, semantic held-out success, 실제 로봇 성공, 공식 Zero-WAM 성공을
주장하지 않는다.

## Paired human-video task-spec 조건

Reviewed paired candidate는 sensorimotor episode로 변환하지 않고 action-free
`HumanVideoPrompt` 상태로 같은 object-state criterion에 연결한다.

```shell
so101-wam-paired-semantic-controls \
  --config configs/mujoco_robot_free.toml \
  --validation-pair-manifest data/pairs/validation.json \
  --semantic-manifest configs/mujoco_semantic_benchmark.json \
  --checkpoint checkpoints/paired_candidate_001.pt \
  --training-report reports/paired_candidate_001.training.json \
  --artifact-dir runs/paired_semantic_001/artifacts \
  --report runs/paired_semantic_001/report.json
```

조건은 세 개로 고정된다.

1. `matched_human_video`: semantic dataset task와 label/index가 같은 reviewed pair
2. `wrong_task_human_video`: label/index가 모두 다른 reviewed validation pair
3. `null_human_video`: matched frame의 timestamp·shape·task identity는 유지하고 RGB만 0

각 조건은 원본 또는 파생 task-spec fingerprint/source hash, paired training inventory,
checkpoint/training/semantic manifest hash와 child trial artifact를 결합한다. runtime의
physical prompt는 MuJoCo context 생성용 자리표시자이며 모델에는 전달되지 않는다.
paired report schema 3은 child trace에서 future-latent MSE를 다시 계산하고, matched보다
MSE와 terminal success rate가 모두 엄격히 낮은 조건만 latent-proxy/outcome mismatch로
표시한다. 정렬 pair 부재와 동률은 제외하며 child report SHA가 판정 대상 trial/trace
SHA를 결합한다. 세 조건의 차이는 local 기술 통계다. pixel-space video quality, 독립
review, 사전등록, prompt causality, semantic held-out 성공이나 실제 로봇 성공을
증명하지 않는다.
