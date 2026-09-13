# Dual SO-101 MuJoCo simulation

## 구현 결과와 증거 범위

MuJoCo는 이 프로젝트의 첫 물리 시뮬레이터다. 로봇 모델은 Google DeepMind
MuJoCo Menagerie의 공식 `robotstudio_so101` MJCF를 두 번 부착하며, 각 instance에
`left_` / `right_` prefix를 붙인다. upstream asset은 Apache-2.0이고 commit
`da76818e269b82289eba39808e2fb91d679d6994`를 그대로 vendoring했다.

로컬 장면 `src/so101_wam/assets/dual_so101_wam.xml`은 다음을 포함한다.

- 고정 중앙 토르소와 어깨 위치 `y=±0.19 m`, `z=0.96 m`
- +90° Y 회전으로 neutral chain을 아래로 내린 두 SO-101 arm
- upstream 손목 카메라 `left_wrist_cam`, `right_wrist_cam`
- 선택 진단 카메라 `head_optional`
- 보호 대상 floor/table/torso와 접촉 시험용 free block
- 좌우 합계 12개 position actuator

이 장면은 사용자가 지정한 arms-down humanoid 형상을 구현한 설계 모델이다. 어깨
간격, 토르소 크기, base transform은 아직 실제 조립체 실측값이 아니므로 실제
충돌 안전을 증명하지 않는다.

Zero-WAM 정책과 시뮬레이터의 역할은 분리한다. 공식 Zero-WAM 프로젝트가 설명하는
human-video in-context task specification, causal video-action policy, IFP 구조는
정책/학습 계약의 근거다. MuJoCo는 그 정책이 낸 12축 target을 실제 embodiment와
접촉 모델에서 검사하고 두 손목 관측을 생성하는 실행·검증층이다.

## 설치와 실행

기존 프로젝트 Python 환경에 optional simulator dependency를 설치한다.

```shell
python -m pip install -e '.[sim]'
PYTHONPATH=src python -m so101_wam.mujoco_cli \
  --config configs/mujoco.toml \
  --steps 2 \
  --report reports/g8_smoke.json
```

desktop GUI로 장면과 joint slider를 직접 보려면 GLFW/display가 있는 terminal에서
공식 viewer를 연다.

```shell
python -m mujoco.viewer \
  --mjcf=src/so101_wam/assets/dual_so101_wam.xml
```

Linux/WSL headless 기본값은 `mujoco.gl_backend="egl"`이다. `MUJOCO_GL`이 이미
설정되어 있으면 MuJoCo의 기존 환경값이 우선한다. EGL이 없는 desktop 환경에서는
config를 `glfw`로 바꿀 수 있다.

기본 명령은 학습 성능을 주장하지 않는 작은 wrist-roll smoke policy를 사용한다.
실제 physics control을 끄고 렌더/계약만 검사하려면 `--shadow`, 선택 head view까지
검사하려면 `--head-camera`를 사용한다.

```shell
PYTHONPATH=src python -m so101_wam.mujoco_cli --steps 1 --shadow
PYTHONPATH=src python -m so101_wam.mujoco_cli --steps 1 --head-camera
```

`--report PATH`는 stdout 출력을 유지하면서 같은 JSON을 G8 evidence file로
atomic/no-overwrite publish한다. 기존 파일은 덮어쓰지 않으므로 새 evidence name을
사용해야 한다. report top-level에는 `schema_version=1`, `gate="G8-simulation"`,
`result="pass"`, 전체 validated MuJoCo config의 `config_sha256`이 포함된다. smoke policy report의 `checkpoint_sha256` /
`checkpoint_id`는 `null`이다.

기존 CompactWAM checkpoint와 기록된 physical prompt를 사용하는 경로도 hardware
CLI와 같은 artifact 계약을 쓴다.

```shell
PYTHONPATH=src python -m so101_wam.mujoco_cli \
  --checkpoint checkpoints/compact_wam_candidate_001.pt \
  --prompt data/prompts/prompt_000001.npz \
  --manifest data/prompts/prompt_000001.json \
  --steps 20 \
  --report reports/g8_candidate_000001.json
```

이 저장소에는 학습 완료 checkpoint가 없으므로 위 명령이 task success를 보장하지
않는다. checkpoint session report는 `policy="compact_wam"`이며 checkpoint 파일의
실제 SHA-256과 checkpoint metadata의 `checkpoint_id`를 같이 기록해 deployment
issuer가 정확한 candidate artifact와 G8 simulation evidence를 결합할 수 있게 한다.
G10 발급기와 실제-output runtime은 이 저장 JSON의 요약값만 신뢰하지 않는다. 보고된
policy-step 수를 사용해 정확히 같은 candidate, prompt pair, MuJoCo config를 CPU에서
다시 실행하고 생성된 전체 report dictionary가 저장본과 동일한지 확인한다.

폐쇄 루프 held-out 진단 benchmark는 사전 고정된 JSON manifest와 자동 terminal
checker를 사용한다.

```shell
PYTHONPATH=src python -m so101_wam.mujoco_benchmark \
  --config configs/mujoco.toml \
  --manifest configs/mujoco_heldout_benchmark.json \
  --artifact-dir reports/mujoco_heldout_artifacts \
  --report reports/mujoco_heldout_benchmark.json
```

manifest는 `train_task_ids`와 `heldout_tasks[].task_id`의 분리를 요구하고, 각 held-out
task는 3개 이상의 seed를 가져야 한다. seed마다 초기 body joint에 결정론적 perturbation을
적용한다. report는 `gate="G8-heldout-benchmark"`,
`evidence_level="simulation"`, 전체 및 task별 성공률과 95% Wilson interval,
terminal error, 실패 사유 count를 기록한다. 각 seed trial은 고유 `trial_id`, artifact
경로, artifact SHA-256으로 상위 report에 결합된다. 모든 trial이 terminal criterion을
만족해야 `result="pass"`이며, 하나라도 실패하면 `result="fail"`과 종료 코드 1을
반환한다. report와 artifact는 기존 파일을 덮어쓰지 않는다.

이 경로의 `policy="joint_target_reference"`는 목표 관절값을 반복 전송하는 진단용
reference controller다. 따라서 폐쇄 루프 actuation과 terminal checker를 검증하지만
학습된 WAM 정책의 일반화 성능, 실제 SO-101 task success, 논문 규모 RoboTwin 성능을
증명하지 않는다.

동일 manifest의 joint target을 CompactWAM checkpoint로 진단하려면 checkpoint
session 입력을 benchmark에 같이 제공한다.

```shell
PYTHONPATH=src python -m so101_wam.mujoco_benchmark \
  --config configs/mujoco.toml \
  --manifest configs/mujoco_heldout_benchmark.json \
  --artifact-dir reports/mujoco_candidate_joint_proxy_artifacts \
  --report reports/mujoco_candidate_joint_proxy.json \
  --checkpoint checkpoints/compact_wam_candidate_001.pt \
  --prompt data/prompts/prompt_000001.npz \
  --prompt-manifest data/prompts/prompt_000001.json
```

이 경로는 `gate="G8-candidate-joint-proxy"`,
`benchmark_scope="candidate_joint_proxy_not_heldout_verified"`를 사용한다. report와
각 seed artifact는 checkpoint ID/SHA-256, prompt NPZ/manifest SHA-256, prompt
fingerprint를 함께 기록한다. `task_disjoint=true`는 manifest 안의 ID 분리만 뜻하며
`checkpoint_task_split="not_verified"`, `prompt_task_match="not_verified"`를 명시한다.
따라서 terminal criterion 성공은 해당 prompt/checkpoint가 joint target proxy를
통과했다는 뜻일 뿐, checkpoint가 그 task를 학습에서 보지 않았거나 prompt가 task와
semantic하게 맞는다는 증거가 아니다. terminal tolerance 불일치는 trial failure로
집계하고 checkpoint/prompt/session 형식 및 rollout 안전 오류는 hard error로 중단한다.

같은 checkpoint와 task/seed를 matched, alternate, wrong-task, shuffled, null,
counterfactual prompt에 반복 적용하는 경로는
[PROMPT_CONTROLS.md](PROMPT_CONTROLS.md)에 있다. 파생 prompt와 condition별 benchmark를
모두 hash로 결합하지만 joint proxy 비교일 뿐 semantic task success나 prompt
causality 판정은 아니다.

객체 상태를 기준으로 판정하려면 별도 benchmark를 사용한다.

```shell
PYTHONPATH=src python -m so101_wam.mujoco_semantic_benchmark \
  --config configs/mujoco.toml \
  --manifest configs/mujoco_semantic_benchmark.json \
  --artifact-dir reports/mujoco_object_artifacts \
  --report reports/mujoco_object_benchmark.json \
  --checkpoint checkpoints/compact_wam_candidate_001.pt \
  --training-report checkpoints/compact_wam_candidate_001.training.json \
  --prompt data/prompts/prompt_000001.npz \
  --prompt-manifest data/prompts/prompt_000001.json
```

schema-3 manifest는 정확히 하나의 held-out task마다 `object_body`, seed별
`initial_object_positions`, `target_object_position`과 metre 단위 Euclidean tolerance를
실행 전에 고정한다. 번들 scene은 box `task_block`과 물성·형상이 다른 cylinder
`task_cylinder`를 제공한다. adapter가 선택한 free body의 초기값을 설정하고 terminal
world position을 읽으므로 benchmark나 policy가 MuJoCo 내부 배열에 직접 접근하지
않는다. 각 artifact는 object body, 초기/목표/최종 position, position error, robot joint
state, checkpoint/training-report/prompt/config/manifest hash를 기록한다. 기본 manifest는
들어 올리기 목표이므로 학습되지 않은 checkpoint의 `result="fail"`은 정상적인 negative
result다. runner는 manifest, checkpoint, training report, prompt NPZ/manifest bytes를
첫 load 전에 한 번 캡처하고 그 bytes와 내부 snapshot만 parse, hash, rollout에 사용한다.
semantic manifest와 training report는 duplicate key와 비표준 숫자를 거부한다.

semantic report schema 5는 terminal criterion을 계산한 trial을 `status="scored"`, 안전하게
중단돼 terminal state가 없는 trial을 `status="execution_failure"`로 구분한다. 후자는
`object_position_error_m`, final joint/object position을 `null`로 두고 다음 seed를 계속
실행한다. 선택한 body의 compiled mass/inertia와 geom type/size/pose/friction/contact
parameter는 이름을 제외한 canonical profile로 기록한다. profile SHA-256은 성공 여부와
무관하게 checkpoint benchmark가 보존한다. 이는 simulator model 다양성 증거이며 실측
geometry/dynamics 증거가 아니다. `failure_reason` taxonomy와 구조화 근거는 다음과 같다.

Checkpoint trial은 `trial_id`와 동일한 `rollout_id`로 bounded executability trace를
별도 JSON에 기록한다. decoded action inventory는 최대 64개이고, 잘리더라도 전체 policy
action, servo target, executed action stream hash는 유지된다. trace와 trial artifact는
policy/servo/sent/shadow count, safety accept/reject/clip count와 reason을 연결하며 trace
파일 SHA-256은 상위 trial index까지 전달된다. safety rejection은 예외를 올리기 전에
해당 servo 판정을 관찰하므로 partial trace가 남는다. trace schema 3은 safety joint
range의 nearest-bound/span으로 decoded policy target, safe servo target, measured joint,
executed action의 최소 여유를 기록한다. 각 완료 servo 뒤의 모든 MuJoCo contact를 bounded
progression과 전체 stream SHA-256으로 보존하고 object/forbidden contact 수와 최소 거리를
집계한다. collision 중단은 phase와 contact identity를 같은 trace에도 복사한다. `scored`
trial은 manifest의 policy step 전부와 하나 이상의 servo step, 음수가 아닌 네 margin,
servo별 contact sample, safety rejection/forbidden contact 0건, 모든 servo step의
sent/shadow 분류가 있어야 한다. bounded progression이 잘려도 전체 servo/contact index
stream SHA-256이 같아야 하므로 중복·누락 sample은 거부한다. checkpoint policy가 opt-in한
future-latent telemetry는 예측 시점의 current live frame과 이후 policy-rate live frame만
사용한다. 학습 target과 같은 live-segment encoder로 관측 frame을 인코딩하고 예측의
`+1…+N` offset과 정렬해 prediction/observation producer SHA-256과 재계산 가능한 index
SHA-256, offset별 MSE,
tail censored count를 기록한다. 원본 latent는 저장하지 않으므로 validator는 index hash만
재계산하고 prediction/observation producer digest는 형식만 검사한다. trace 파일 SHA-256은
해당 digest를 결합할 뿐 원본 stream을 재검증하지 않는다.
semantic report는 squared-error/element count로 이를 다시
집계한다. terminal frame은 policy deadline target이 아니므로 사용하지 않는다. 이 metric은
pixel/video quality가 아닌 compact latent proxy이며 runtime latent success marker와 future
visual availability/success marker는 계속 `false`다.

Checkpoint semantic benchmark/suite schema 5는 `mujoco_model_identity`도 trial artifact와
상위 report에 결합한다. adapter가 준비 단계에서 실제 compiled MJB의 SHA-256·크기와
engine version을 한 번 기록하며 disconnect 후에도 보존한다. policy/servo loop에서
모델을 다시 해시하지 않는다. `rollout_session`은 실행에 사용한 adapter의 정보이고,
`post_failure_inspection`은 실패 후 예상 장면을 별도로 조사한 정보다. 서로 다른 모델
identity가 섞인 benchmark/suite는 거부한다. 이는 GPU driver 증명이나 다른 MuJoCo build
간 canonical scene hash가 아니다.

- `object_body_position_tolerance`: scored terminal position이 tolerance 밖
- `mujoco_collision`: collision phase와 geom/body/category/distance contact 목록
- `mujoco_adapter_error`: 물리 관절 범위 초과 등 adapter 오류의 type과 message.
  해당 trial을 미채점 실행 실패로 기록하며 관절 한계를 완화하지 않는다.
- `safety_joint_limit`: `_joint_limit:` safety reason
- `safety_watchdog`: `stale_action` watchdog reason
- `safety_rejection`: 그 밖의 action/observation safety reason
- `policy_model_error`: `PolicyError` type과 message

MuJoCo 통합 test는 zero-parameter checkpoint의 axis mean을 joint limit 안의 결정적
target으로 고정해 실제 `target preflight` torso/right-arm contact를 만들고, 별도
5 Hz policy/50 Hz servo 설정에서는 원래 chunk timestamp가 0.1초 watchdog을 넘도록 해
실제 `stale_action`을 만든다. 둘 다 `run_checkpoint_task_trial()` production 경로에서
`execution_failure`로 변환된다. 이는 simulator failure-path 증거이지 실물 충돌·지연
증거가 아니다.

성공률과 Wilson interval의 분모에는 모든 scheduled trial이 들어간다. object-position
평균/최댓값은 `scored_trial_count`만 사용하고 전부 execution failure이면 `null`이다.
이때 report와 trial artifact의 `semantic_mujoco_object_state_evaluated`도 `false`다.
manifest/config/backend, checkpoint/training report, prompt identity, clock 또는 알 수 없는
rollout 오류는 provenance나 infrastructure 문제이므로 계속 hard error로 중단한다.

checkpoint 경로는 training report core digest, exact checkpoint hash/ID/architecture,
offline-only marker와 정렬·중복 제거된 train/validation task inventory를 검증한다. 선언한
dataset task는 train inventory에 없어야 하고 validation inventory 및 prompt의 task
label/index와 정확히 일치해야 한다. 한 prompt를 여러 held-out object task에 재사용하면
rollout 전에 거부한다. object task와 dataset task가 의미상 같다는 근거는
`semantic_task_mapping="manifest_declared"`로 한정된다.

`semantic_mujoco_object_state_evaluated=true`는 정의된 simulation object criterion을
계산했다는 뜻뿐이다. report는 `semantic_heldout_success_claimed=false`,
`future_visual_metric_available=false`, `future_visual_success_claimed=false`,
`prompt_causality_claimed=false`, `real_world_success_claimed=false`,
`official_zero_wam_claimed=false`,
`checkpoint_task_split="training_report_verified"`,
`prompt_task_match="dataset_identity_verified"`를 기록한다.

동일 object-state task와 seed에 7개 prompt 조건을 반복하려면 semantic
prompt-control suite를 사용한다.

```shell
PYTHONPATH=src python -m so101_wam.mujoco_semantic_prompt_controls \
  --config configs/mujoco_robot_free.toml \
  --prompt-control-manifest runs/prompt_control_001/manifest.json \
  --semantic-manifest configs/mujoco_semantic_benchmark.json \
  --training-report checkpoints/compact_wam_candidate_001.training.json \
  --direction-preregistration runs/prompt_control_001/semantic_direction.json \
  --artifact-dir reports/mujoco_semantic_prompt_artifacts \
  --report reports/mujoco_semantic_prompt_controls.json
```

여섯 조건은 dataset task identity 일치를 요구하고 `wrong_task`만 명시적
`expected_mismatch`로 실행한다. 상위 report는 checkpoint/training evidence와 모든
condition prompt·trial·nested report hash를 결합한다. object-position 차이는 아직
기본적으로 기술 통계이며 기존 joint-proxy 방향성 plan의 claim scope를 넓히지 않는다.
schema-3 execution failure는 condition별 분모와 taxonomy에 남고 terminal metric과
matched 대비 object-error delta는 `null`이다.
일반·paired semantic prompt-control 상위 report schema 4는 trial, artifact, summary의
모델 정보·출처를 대조하고 조건 간 compiled 모델·엔진 불일치를 거부한다. 모델 정보와
출처 목록은 condition 및 최상위 report에도 보존한다. 기존 보고서는 재실행해야 한다.
paired human-video 상위 report schema 4는 trace에서 재계산한 future-latent MSE가 matched
조건보다 낮으면서 success rate도 낮은 조건만 latent-proxy/outcome mismatch로 센다.
정렬 pair가 없거나 MSE가 같으면 세지 않는다. condition의 child report SHA를 따라가면
판정에 사용한 trial artifact와 executability trace SHA를 확인할 수 있다. 이는 future
video quality, semantic success 또는 prompt causality 판정이 아니다.
선택적 semantic 방향성 plan은 prompt-control manifest, semantic manifest, training
report, checkpoint, MuJoCo project config SHA를 rollout 전에 결속하고
`claim_scope="preregistered_mujoco_object_state_directionality_only"`만 허용한다. 상위
`result="complete"`는 실행 완료이고 nested `directionality_evaluation.result`가
object-state 방향성의 `pass`/`fail`이다. 이 등록 수준은
`local_input_hash_bound_before_rollout`이며 외부 timestamp는 검증하지 않는다.
`semantic_task_mapping="manifest_declared"`와 모든 negative-safe claim도 그대로
유지한다.

여러 dataset/object task를 같은 checkpoint 아래 집계하려면 case별 task-specific
prompt를 갖는 semantic suite를 사용한다.

```shell
PYTHONPATH=src python -m so101_wam.mujoco_semantic_suite \
  --config configs/mujoco_robot_free.toml \
  --suite runs/semantic_suite/suite.json \
  --mapping runs/semantic_suite/mapping.json \
  --checkpoint checkpoints/compact_wam_candidate_001.pt \
  --training-report checkpoints/compact_wam_candidate_001.training.json \
  --artifact-dir reports/mujoco_semantic_suite_artifacts \
  --report reports/mujoco_semantic_suite.json
```

`configs/mujoco_semantic_suite.template.json`은 2개 이상의 case와 case별 semantic
manifest, prompt NPZ/manifest의 bundle-relative 경로를 선언한다. 각 child semantic
manifest는 계속 정확히 하나의 held-out object task만 가진다. 따라서 한 prompt를 여러
object task의 증거로 재사용하는 기존 guard는 유지된다.

`configs/mujoco_semantic_mapping.template.json` schema 3은 같은 case를 정확히 한 번씩 포함하고
서로 다른 dataset label/index, object task ID, semantic manifest SHA-256을 결합한다.
`object_task_id`는 해당 manifest의 유일한 `heldout_tasks[].task_id`와 정확히 같아야 한다.
`object_task_signature_sha256` schema 2는 criterion, object body, seed별 초기 object position,
target position과 tolerance의 canonical JSON SHA-256이다. loader가 manifest에서 이를
다시 계산하고 signature 중복을 거부하므로 label/ID만 바꾼 동일 geometry case는 통과하지
못한다. 초기 position 순서는 seed로 정렬해 순서 변경도 새 task로 취급하지 않는다.
값은 `load_semantic_manifest()`로 manifest를 읽은 뒤 `object_task_signature()`로 계산한다.
`semantic_match="human_reviewed"`는 비어 있지 않은 `reviewer_id`가 필요하며 실제 독립
검토가 없으면 `unverified`를 사용해야 한다. loader는 이 상태를 자동 승격하지 않는다.
scope는 `local_dataset_object_mapping_attestation_only`로 고정된다.

runner는 suite, mapping, checkpoint, training report와 모든 case 입력을 output 생성
전에 읽고 검증한 뒤 frozen snapshot만 child benchmark에 전달한다. child report의
config/checkpoint/training/manifest/prompt hash, task/seed schedule, trial artifact hash,
schema-3 failure evidence와 physical profile hash를 다시 계산한다. suite report schema 5는
전체 `scored_trial_count`, `execution_failure_count`를 보존하며 미채점 object 오차를
집계에서 제외한다. `distinct_object_task_count`, `distinct_object_body_count`, 이름을
제외한 `distinct_object_physical_profile_count`와
`physical_object_diversity_observed`도 기록한다. body 이름만 다른 동일 profile은 물리
다양성으로 세지 않는다. 이는 simulator의 compiled profile 증거이지 의미 대응의 독립
검토나 실측 물성 증거는 아니다. 상위 `result="complete"`는 모든 child 실행과 검증이 끝났다는
뜻이고 case별 `result="pass"|"fail"` 및 전체 성공률/Wilson interval은 별도로 남는다.
`independent_mapping_verified=false`, `semantic_heldout_success_claimed=false`,
`prompt_causality_claimed=false`, `real_world_success_claimed=false`,
`official_zero_wam_claimed=false`는 항상 유지된다.

합성 episode 생성, offline checkpoint 학습, 두 case suite 실행을 한 번에 묶은
robot-free 경로도 있다.

```shell
so101-wam-robot-free-semantic \
  --output-dir runs/robot_free_semantic_001
```

이 명령은 train task와 두 validation task를 task-disjoint하게 만들고, validation
task마다 다른 prompt를 고정한다. `task_block`/`task_cylinder`의 seed `7/13/29`를
production checkpoint runner로 실행해 schema-5 report와 child artifact를 발행한다.
mapping status는 `unverified`이며 학습 budget은 stage 2 한 step뿐이다. 따라서
재현 가능한 multi-object negative/positive outcome을 남길 수는 있지만 semantic success,
prompt causality, 실측 물성 또는 실물 안전을 증명하지 않는다.

로봇 없는 acceptance 경로는 이 checkpoint-session G8 실행을 자동화된 중간 단계로
사용한다.

```shell
so101-wam-robot-free --output-dir runs/robot_free_001
```

이 명령 계약은 합성 30 Hz episode로 tiny offline candidate를 만든 뒤 동일 checkpoint를
MuJoCo G8에서 재사용하고, 마지막에 real-output guard가 그 offline candidate를
거부하는지 확인한다. 따라서 산출물은 `evidence_level="offline"` training report와
`evidence_level="simulation"` G8 report까지의 증거다. 실제 조립체의 G6/G7/G9/G10
증거 또는 task success로 해석하지 않는다.

설치 패키지에 포함된 기본 config는 32×24 wrist render를 사용해 acceptance runtime을
작게 유지한다. 저장소의 `configs/mujoco_robot_free.toml`은 명시적 override와 검토용
동일 사본이다. 이 경로는 collision gate와 MuJoCo actuation이 켜진 config만 허용하고
G8 report의 모든 servo step이 실제 simulation action으로 전송됐으며 shadow step이
0인지 재검증한다. 결과 JSON의 `g8` 객체는 `result`, `gate`,
`policy_steps`, `checkpoint_id`, `checkpoint_sha256`, `report_sha256`를 기록한다.
실제 파일 위치는 `robot_free_result.json`의 `artifacts.g8_report.path`와 SHA-256을
기준으로 추적한다.

실행에 사용한 config bytes는 결과 root의 `effective_config.toml`로 no-overwrite
복사되고 G8은 그 복사본을 다시 로드해 사용한다. 따라서 결과 manifest의 config
경로도 bundle-relative다. `so101-wam-verify-robot-free --result <result.json>`은 이
config와 G8/checkpoint/prompt 결합을 정적으로 재검증하지만 G8 자체를 재실행하지는
않는다.

## Sim-real joint 단위 계약

LeRobot `0.6.1` SO follower는 `use_degrees=true`일 때 body motor를 degrees로,
gripper를 `RANGE_0_100`으로 노출한다. factory는 이 값을 명시적으로 고정한다.
MuJoCo adapter도 외부 `SensorimotorFrame`과 `ActionChunk`에서 같은 단위를 사용한다.

| 축 | runtime/dataset 단위 | MuJoCo 내부 |
| --- | --- | --- |
| shoulder pan/lift, elbow flex, wrist flex/roll | degree | radian |
| gripper | 0–100 | MJCF hinge range의 선형 mapping |

따라서 fake, MuJoCo, LeRobot backend의 12축 key/order는 같고 simulator 경계에서만
단위 변환한다. joint sign, zero offset, gripper endpoint가 실제 조립체와 같은지는
G7 motor bench에서 축별로 확인해야 한다.

## Clock과 카메라 계약

upstream SO-101 model timestep은 0.005초(200 Hz)다. adapter는 50 Hz servo period
0.02초가 physics timestep의 정수배인지 확인하고, action row 하나마다 정확히 네 번
`mj_step`을 호출한다. 정수배가 아니면 connect 단계에서 실패한다.

정책에 노출되는 필수 image key는 계속 정확히 다음 두 개다.

- `left_wrist`: MuJoCo `left_wrist_cam`
- `right_wrist`: MuJoCo `right_wrist_cam`

두 frame은 같은 설정 해상도의 `uint8[H,W,3]`다. `head_optional`은 config/runtime이
명시적으로 요청한 경우에만 추가되며 `PRIMARY_CAMERA_KEYS`에는 들어가지 않는다.
CompactWAM checkpoint session에서는 기록 prompt와 live MuJoCo frame의 해상도도
정확히 같아야 한다. 기본 MuJoCo config와 LeRobot hardware template은 모두
640×480으로 맞춰져 있다.

`so101-wam-mujoco`의 기본 config는 설치 패키지 내부에 포함되므로 현재 작업
디렉터리와 무관하다. 저장소의 `configs/mujoco.toml`은 명시적 `--config` override와
검토용 동일 사본이며 자동 test가 packaged copy와의 byte parity를 확인한다.

## Collision gate

`MujocoBiSOAdapter.send_action()`은 한 servo row마다 두 단계로 fail closed한다.

1. 현재 전체 integration state를 probe data로 복사하고 target joint pose에서
   `mj_forward`를 실행한다. 금지 접촉이면 실제 state를 전혀 진행하지 않는다.
2. target이 통과하면 네 physics substep을 실행한다. 이동 도중 금지 접촉이 생기면
   `mjSTATE_INTEGRATION` snapshot으로 rollback하고 예외를 발생시킨다.

금지 접촉:

- left arm ↔ right arm
- 각 arm ↔ torso/table/floor
- 같은 arm의 비인접 self collision

허용 접촉:

- gripper 내부 fixed/moving jaw assembly
- task object ↔ gripper/robot/table
- static environment끼리의 접촉

runtime은 adapter 예외를 recovery로 전환하고 managed rollout은 fault latch 후
disconnect한다. 이 gate는 nominal MJCF에 대한 simulation evidence다. 실제 로봇에
적용하려면 실측 `torso_T_arm_base`, camera mount, table pose, 링크/케이블 여유를
반영하고 slow-jog 검증을 추가해야 한다.

## 검증

optional dependency가 없는 기본 환경에서는 MuJoCo integration test만 skip되고
나머지 suite가 계속 동작한다. simulator 환경에서는 다음을 실행한다.

```shell
python -m pip check
python -m ruff check src tests
python -m mypy src
python -m compileall -q src tests
python -m pytest -q
bash scripts/validate_mujoco.sh
```

CI도 같은 원칙을 따른다. Python 3.12 Ubuntu runner가 wheel-backed `.[sim]` 설치 후
checkout 밖에서 세 packaged config 로드와 fake smoke를 먼저 실행한다. 이어서
`pip check`, Ruff, mypy, compileall, 전체 pytest, `scripts/validate_mujoco.sh`를
실행한다. 별도의 장시간 robot-free E2E 명령은 CI step으로 중복 실행하지 않고 test
suite가 해당 계약을 검증한다.

검증 항목은 scene compile, 12개 named joint/actuator, 두 wrist render, optional head
render, degrees/percent round trip, 4 physics substeps, collision preflight 무상태 변경,
이동 중 충돌의 전체 integration-state rollback, 기존 10/50 Hz managed rollout,
candidate G8 report의 exact CPU 재실행 비교, 3-seed held-out 진단 benchmark,
robot-free 경로의 G8/guard 계약이다.

## 근거

- [Zero-WAM official project](https://robbyant-research.github.io/Zero-WAM/)
- [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Menagerie SO-101 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
- [LeRobot v0.6.1 SO follower config](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/config_so_follower.py)
