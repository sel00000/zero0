# zero01 validation

## 검증 원칙

이 문서는 zero01 MVP를 fake, offline, simulation, real hardware 증거로 나누어
검증한다. 논문 결과, local unit test, fake adapter 결과, simulation 결과, 실제 로봇
결과를 서로 섞지 않는다.

현재 저장소에서 증명할 수 있는 것은 계약과 일부 offline 동작뿐이다. 실제 SO-101
팔, 손목 카메라, 케이블링, 충돌 여유, latency, 성공률은 하드웨어
없이 증명할 수 없다.

## 증거 등급

| 등급 | 의미 | 사용 가능 주장 |
| --- | --- | --- |
| Fake | fake robot/camera 또는 unit test | shape, key, fail-closed 계약 |
| Offline | 저장된 episode, local tensor, config 검사 | dataset schema, prompt/context/model 계약 |
| Simulation | 물리 sim 또는 collision model | 충돌 gate, task rollout 후보 성능 |
| Real | 실제 SO-101 하드웨어 | operator halt, latency, camera sync, motor safety, task success |

Real 증거 없이는 "로봇에서 성공했다", "안전하다", "latency 목표를 만족한다"를
완료 주장으로 쓰지 않는다.

## 단계별 게이트

| Gate | 증거 | 통과 기준 | 실패 기준 | 현재 상태 |
| --- | --- | --- | --- | --- |
| G0. 범위 고정 | 문서/코드 검사 | fixed torso, dual SO-101, `left_wrist`/`right_wrist` only, optional head diagnostics 명시 | head/base/depth가 core requirement가 됨 | 문서화됨 |
| G1. Config 계약 | Fake/offline | wrist-only fake default, backend/actuation flag 일치, hardware session에 ports + wrist camera IDs + calibration dir, real output에 measured safety + hardware/calibration IDs + 12축 arms-down home/tolerance 요구, LeRobot `0.6.1` | fake config로 LeRobot 출력, head-required 입력, identity/home/tolerance 없는 real enable | `tests/test_config.py`, `tests/test_runtime.py` 통과 |
| G2. Frame/prompt 계약 | Fake/offline | RGB `uint8[H,W,3]` 두 장, optional per-image timestamp key 일치, joint/action width 12, prompt duration 3.000-12.000 s, timestamp strictly increasing | 카메라/timestamp 누락·불일치, NaN/Inf, wrong width, prompt duration out of range | `tests/test_contracts.py` 통과 |
| G3. Context 계약 | Offline | 10 Hz 기준 30 s 이하 context, prompt pinned, live FIFO, lower-triangular causal mask | prompt가 live FIFO에 의해 밀려나거나 causal order가 깨짐 | `tests/test_context.py` 통과 |
| G4. Compact WAM 계약 | Offline | exactly 2 wrist views, action shape `[B,H,12]`, inverse-dynamics head가 raw prompt embedding을 직접 받지 않음, inference no-grad, `ContextSnapshot -> ActionChunk` policy wrapper | 1/3 view 허용, prompt mask non-causal, wrong proprio/action width, bad/NaN model output 통과 | `tests/test_model.py`, `tests/test_policy.py` 통과 |
| G5. Dataset/checkpoint/factory/adapter/safety/runtime unit suite | Fake/offline | 전체 `pytest -q` 통과, 30→10 Hz prompt 변환, strict checkpoint load, v0.6.1 bimanual factory, connect/disconnect, policy/servo clock 분리, original chunk timestamp watchdog, shadow/actuated fake 통과 | artifact/lifecycle/unit failure, shadow target를 executed로 기록, stale chunk timestamp 갱신 | fresh CI/local command가 source of truth |
| G6. Camera bench | Real bench | 좌/우 wrist RGB 30 FPS, 같은 해상도, locked frame/timestamp pair, skew p95 <= 17 ms, camera-buffer age 및 observation assembly latency p95 <= 100 ms | frame drop, swapped cameras, timestamp 부재/감소, skew/age 초과 | 관측-only preflight 구현됨; 실제 JSON report와 영상 identity 확인 필요 |
| G7. Motor bench, no policy | Real bench | arms-down home 재현, 12축 feature 순서 확인, per-tick delta 한계 내 teleop, operator halt 시 명령 중단 | 좌우/축 반전, halt 후 명령 지속, 케이블 간섭 | home 반복성 계측 구현됨; 실제 teleop/축 방향 확인 필요 |
| G8. Collision envelope | Simulation + real slow jog | torso/table/dual-arm 금지 부피 안으로 target이 들어가면 무전송·rollback, slow jog에서 접촉 없음 | 충돌 후보 action 통과 또는 실제 간섭 | nominal MuJoCo gate/test와 robot-free simulation leg는 CI/local로 검증; 실측 scene + real slow jog 필요 |
| G9. Prompt record/replay | Real data | 3-12 s prompt에 모든 frame action 포함, 30 FPS 저장, checksum round trip, core feature만 policy 입력 | action 누락, timestamp drift > 5%, checksum mismatch | observation-only recorder 구현됨; 실제 episode와 수동 시연 확인 필요 |
| G10. Dry rollout | Real, motors limited | actuation gate 명시 enable, issuer가 exact candidate weights와 G6-G9 source SHA를 결합해 `artifact_kind="compact_wam_deployment"`, `trained=true`, `deployment_ready=true` checkpoint + schema-2 외부 certification을 no-overwrite pair로 발행, runtime이 robot 생성 전에 모든 원본 source를 재검증하고 exact candidate/prompt/config G8 세션 전체 report를 재실행 비교하며 deployment/candidate state tensor 완전 동일성을 재확인, `is_calibrated=true`, 최초 자세 home tolerance 이내, stale observation/action에서 무전송·fault disconnect, no autonomous motion before prompt fingerprint | issuer/certification/source 원본 없이 motion, offline candidate/prompt로 motion, source/checkpoint/config/hardware/weights substitution, 저장 G8 요약 위조, step scope 초과, uncalibrated/home mismatch/stale frame에서 command 전송 | fail-closed issuer/runtime 재검증 test 구현됨; 실제 source evidence와 dry rollout은 없음 |
| G11. Task rollout | Real | 사전 등록 task/protocol에서 N회 trial, success/failure 모두 기록, head camera 없이 core score 산출 | trial 누락, 실패 숨김, head camera 필수화 | 하드웨어 필요 |

## CI/local 검증 source of truth

고정된 `179 passed`, `186 passed` 같은 과거 수치는 검증 결과의 source of truth가
아니다. 현재 checkout, Python/MuJoCo 버전, optional dependency 상태에 따라 skip 수와
test 수가 바뀔 수 있다. 최신 CI run 또는 같은 checkout에서 다시 실행한 아래 명령의
출력이 현재 상태의 기준이다.

```shell
python -m pip check
python -m ruff check src tests
python -m mypy src
python -m compileall -q src tests
python -m pytest -q
bash scripts/validate_mujoco.sh
```

CI는 Python 3.12 Ubuntu에서 wheel-backed `.[sim]`과 workflow-level test tooling을
설치하고 checkout 밖에서 packaged config 세 개와 fake smoke를 먼저 검증한 뒤 위
gate를 실행한다. MuJoCo headless 실행을 위해 Mesa/EGL system package와
`MUJOCO_GL=egl`을 사용한다.

현재 기준선 증거는 2026-09-02, MuJoCo `3.12.0` 환경에서
`bash scripts/validate_mujoco.sh`가 MuJoCo 및 benchmark target test suite를
통과한 것이다. G8 smoke는 physics steps `40`, sent actions `10`, final state
`rollout_ready`를 기록했다. 진단용 `joint_target_reference` benchmark는 서로 다른
초기 자세의 3개 seed에서 3/3 terminal success와 95% Wilson interval
`[0.4385029682449546, 1.0]`을 기록했다. benchmark 계약은 task별 통계와 실패 사유를
계산하고 각 seed artifact를 trial ID와 SHA-256으로 상위 report에 결합한다. 별도의
통합 test는 실제 CompactWAM checkpoint와 prompt를 seeded MuJoCo session에 로드해
action이 전송되는 경로를 검증한다. 이는 학습 정책의 held-out semantic 성능이나 실물
성공 증거가 아니다.

별도 semantic MuJoCo test는 manifest가 선택한 named free body의 seed별 초기 위치를
adapter 경계에서 설정하고 실제 CompactWAM rollout의 terminal world position을 읽어
metre 단위 목표 오차를 계산한다. 번들 scene의 `task_block`과 `task_cylinder`는 서로
다른 compiled physical-profile SHA-256을 갖고, profile payload에는 object 이름이 들어가지
않는다. report 계약은 seed artifact, object body와 초기/목표/최종 위치, physical profile,
Wilson CI, failure count와 checkpoint/training-report/prompt/config/manifest hash를
결합한다. 이는 local simulation object-state criterion이다. checkpoint 경로는 training report core digest,
exact checkpoint hash, task-disjoint train/validation inventory와 prompt task label/index를
rollout 전에 검증한다. dataset task와 object task의 semantic mapping은 manifest
자체 선언이며 prompt causality, real-world success와 official Zero-WAM compatibility는
검증하지 않는다. production checkpoint benchmark integration test는 `task_block`과
`task_cylinder`를 각각 3개 seed로 실제 rollout한다. checkpoint/training-report/prompt
identity 결속과 terminal body position, geom type, 상위·trial canonical profile hash를
확인한다.

schema-3 failure contract test는 MuJoCo collision, joint-limit safety rejection,
stale-action watchdog, 기타 safety rejection과 invalid policy output을 구조화된 seed 실패로
주입한다. 알려진 실패 뒤에도 남은 seed를 실행하고, 성공률 분모에는 포함하되 terminal
object 오차 통계에서는 제외하는지 검증한다. 전 trial이 미채점이면 평균/최댓값과 final
state는 `null`, `semantic_mujoco_object_state_evaluated=false`다. compiled object profile은
실행 실패와 무관하게 별도 inspection 경계에서 보존한다. 일반 `RolloutError`와
`MujocoCLIError`는 hard error로 유지된다. 이는 예외/report 계약의 unit evidence이며 실제
multi-seed 계속 실행 계약을 검증한다. 별도 MuJoCo 통합 test는 실제 CompactWAM
checkpoint/prompt/session 경로에서 joint-limit 안의 결정적 target이 만드는
`target preflight` torso/right-arm collision과, 5 Hz policy/50 Hz servo chunk가 만드는
`stale_action` watchdog을 각각 관측한다. 이는 simulator event 증거이며 실물 충돌·지연
증거가 아니다.

Checkpoint semantic runner는 각 trial에 같은 `rollout_id`의 별도
`*.executability.json`을 발행한다. trace는 decoded action을 최대 64개까지만 보존하되
전체 policy action, servo target, executed action stream의 SHA-256을 누적한다. policy,
servo, sent, shadow, safety accepted/rejected/clipped 수와 safety reason count도 함께
기록하며 trial artifact가 trace 파일 hash를 다시 결합한다. watchdog safety rejection은
거부된 servo step까지 partial trace에 남는다. schema 3은 safety bounds에 정규화한
decoded/safe/measured/executed 최소 joint-limit margin과 servo별 bounded MuJoCo contact
progression을 더한다. progression이 잘려도 전체 contact stream SHA-256과 contact/object/
forbidden 수, 최소 거리, collision failure phase/contact는 유지된다. scored trial은 전체
servo/contact index-pair stream SHA-256 일치로 contact sample 중복·누락을 거부한다.
opt-in CompactWAM checkpoint rollout은 predicted future latent와 후속 policy observation의
live-encoder latent를 policy index로 정렬한다. trace는 producer stream hash, aligned/censored
pair count, squared-error/element count와 offset별 MSE를 기록하고 semantic report와 paired
상위 report가 trace artifact에서 다시 집계한다. policy step이 하나뿐이면 future target이
없어 proxy availability는 `false`다. 이 proxy는 future visual metric이나 latent success가
아니므로 해당 success marker는 계속 `false`다. validator는 index hash를 재계산하지만
prediction/observation hash는 원본 latent를 저장하지 않아 형식만 검사한다. trace 파일
SHA-256은 이 producer digest를 artifact에 결합할 뿐 원본을 재검증하지 않는다.

같은 환경의 과거 schema-2 CLI smoke는 zero-parameter integration checkpoint와 synthetic
validation prompt 및 그 checkpoint-bound training report를 기본 block-lift manifest의
3개 seed에 실행했다. 결과는 의도한 negative
`0/3`, `failure_counts={"task_block_position_tolerance": 3}`이었고 position error는
`0.1101`–`0.1137 m`였다. 이는 실패도 artifact로 보존되고 lift를 거짓 pass로 만들지
않는다는 실행 증거이지 모델 성능 증거가 아니다.

prompt-control MuJoCo 진단은 7개 조건을 immutable episode로 만들고 동일 checkpoint,
benchmark manifest, task/seed schedule로 기존 checkpoint benchmark를 실행한다. unit
test는 condition별 prompt/report/trial hash, schedule, matched 대비 기술 통계와
negative-safe claim을 검증한다. 선택적 방향 plan은 prompt-control manifest,
benchmark manifest, checkpoint, MuJoCo config hash를 첫 rollout 전에 결속하고 고정된
비열등성·negative margin·최소 trial 규칙을 중첩 `pass`/`fail`로 남긴다. 이는 외부
timestamp가 검증된 등록이 아니며 joint proxy만 평가하므로 prompt 인과성 또는
semantic held-out success 증거가 아니다.

별도 semantic prompt-control test는 같은 7개 조건을 object-state criterion에
연결한다. 모든 조건이 exact checkpoint와 training report, semantic manifest,
task/seed schedule을 공유하는지 검증하며 `wrong_task`만 label/index가 모두 다른
명시적 mismatch 대조군으로 허용한다. nested report의 training/prompt/artifact hash나
이름을 제외한 object physical-profile hash가 다르면 상위 report를 발행하지 않는다.
일반·paired 상위 report schema 4는 compiled MJB hash·엔진 버전도 비교한다. 모델 정보
누락, trial/artifact/summary 불일치, 잘못된 출처 및 조건 간 모델 변경을 거부한다.
schema-3 회귀 test는 condition 하나 또는 전체가
execution failure여도 21개 trial 분모와 taxonomy를 보존하고 미채점 terminal metric과
delta를 `null`로 유지한다. 선택적 semantic 방향 plan은 prompt-control
manifest, semantic manifest, training report, checkpoint, MuJoCo project config hash를
첫 rollout 전에 결속하고 success rate와 평균 object-position error의 네 fixed rule을
중첩 `pass`/`fail`로 남긴다. terminal endpoint가 없으면 해당 error-margin rule은
`endpoint_available=false`로 fail한다. shuffle 조건은 기술 통계이며 dataset→object 의미 대응도
manifest 선언이다.

같은 robot-free candidate를 사용한 실제 CLI smoke는 7조건 × 3 seed의 21 trial을
모두 실행했다. 모든 조건이 `0/3`이고 평균 object-position error가 동일한
`0.1125015992142786 m`였으며 전체 범위는 `0.11011864171792357`–
`0.11369307971289551 m`였다. `wrong_task`는 `control_mismatch_verified`로 기록됐다.
다섯 입력 hash를 고정한 semantic 방향 plan으로 재실행한 결과도 상위
`result="complete"`, nested `result="fail"`, 4개 규칙 중 2개 통과였다. negative
성공률 margin과 object-position error margin은 모두 `0.0`이었다. 동일 입력 두 실행의
report SHA-256은 모두
`d7925ada02f9ae1539d3afdcb60c8e3f7dbe9634064243b97853c693e69448c9`였고,
복사된 plan SHA-256은 원본과 같은
`56847e502d574169133924695ffd66fb5ad0809cec045ee16d505ed3305c164e`였다. 이는 현재
candidate에서 prompt sensitivity가 관찰되지 않았다는 preregistered negative
evidence이다. semantic success나 prompt causality 증거는 아니다.

multi-case semantic suite 계약 test는 2개 case와 6개 seed trial을 fake child runner로
집계한다. suite/mapping strict schema, duplicate key, bundle-relative path, 1:1 case
coverage, unique dataset/object identity와 object-task geometry signature, semantic manifest
hash, reviewer 상태, frozen input,
nested report/trial artifact hash와 no-overwrite를 검사한다. child 하나가 실패해도 상위
`result="complete"`는 실행 완료로 유지하고 child `pass`/`fail`, 전체 성공률과 failure
count를 보존한다. 혼합 execution failure와 전 trial 미채점 child도 검증하며, 후자의
object 오차 통계는 `null`로 유지한다. 이는 orchestration 계약 증거이며 실제 multi-task checkpoint 성능이나
독립 semantic mapping 검토 증거가 아니다.

현재 mapping schema 3의 object-task signature schema 2는 criterion/body, seed별 초기 object position, target position과
tolerance의 canonical signature를 manifest에서 다시 계산한다. signature가 stale이거나
서로 다른 ID가 같은 object-state 선언을 가리키면 output 생성 전에 거부한다. suite
report schema 5는 `distinct_object_task_count`, `distinct_object_body_count`, 이름을 제외한
`distinct_object_physical_profile_count`와 `physical_object_diversity_observed`를 기록한다.
이 simulator profile 다양성은 실측 물성이나 독립 semantic mapping을 뜻하지 않으며
`independent_mapping_verified=false`는 유지한다.

별도 production suite integration test는 deterministic zero-parameter offline candidate와
서로 다른 validation identity/prompt를 `task_block`과 `task_cylinder`의 각 3개 seed에
실행한다. 6개 trial은 모두 terminal object-state로 채점된 negative result이며, 상위
schema-5 report는 두 body, 두 canonical compiled profile과 각 child report hash를
보존한다. 두 mapping은 `unverified`이고 learned generalization 또는 독립 semantic
mapping 증거가 아니다.

같은 환경의 과거 schema-2 실제 CLI smoke는 기존 robot-free state를 offline integration candidate로
다시 결속하고 synthetic validation identity 2개, 서로 다른 task-specific prompt와
각 3개 seed를 실행했다. 상위 결과는 `complete`, 두 case는 각각 `0/3 fail`, 전체는
`0/6`, `scored_trial_count=6`, `execution_failure_count=0`,
`failure_counts={"task_block_position_tolerance": 6}`이었다. 전체 평균 object
position error는 `0.11952665944228581 m`, 최대는 `0.12721979395724456 m`였고 성공률
95% Wilson interval은 `[2.7755575615628914e-17, 0.3903342879021653]`였다. 두 mapping은
모두 `unverified`이고 모든 semantic/causal/real/official claim은 false다. 동일 입력의
schema-2 두 실행 report와 child artifact tree는 byte-identical했고 report SHA-256은
`f02e29f0f885bf91f15a028222f71a0bfd54a1cb4ac5bdbb3d4a0917ed825ad9`였다. 이는 실제
MuJoCo child runner와 suite 집계의 재현 가능한 negative integration evidence이지 학습된
multi-task 일반화나 독립 mapping 검토 증거가 아니다.

같은 suite를 joint-limit을 위반하는 random integration candidate로 실행한 schema-2
failure smoke도 두 case × 3 seed를 중단 없이 모두 기록했다. 결과는
`scored_trial_count=0`, `execution_failure_count=6`,
`failure_counts={"safety_joint_limit": 6}`이며 object error와 final joint/object state는
모두 `null`, `semantic_mujoco_object_state_evaluated=false`였다. 각 trial은 실제 safety
reason `action[0]_joint_limit:right_gripper.pos:below`를 보존했다. 동일 입력 두 report와
artifact tree는 byte-identical했고 report SHA-256은
`3fa2cb50f46e5d7ae878f9d338a352c78e6b880ea5d4e5f22a82ea5e80ef233c`였다. 이는 실제
MuJoCo joint-limit rejection/계속 실행 evidence다. collision·watchdog은 위 별도
production-path 통합 test가 검증하며, 어느 것도 실물 안전 증거는 아니다.

## Robot-free acceptance

```shell
so101-wam-robot-free --output-dir runs/robot_free_001
```

이 CLI 계약은 로봇 없이 실행 가능한 end-to-end acceptance 경로다. 출력 디렉터리는
새 경로이거나 비어 있어야 하며, 경로는 다음 순서를 검증한다.

- synthetic 30 Hz train/validation episodes를 만든다.
- tiny offline CompactWAM candidate를 학습하고 immutable checkpoint/report를 남긴다.
- 정확히 그 checkpoint와 prompt를 MuJoCo G8 checkpoint-session으로 실행한다.
- real-output guard가 offline candidate를 로봇 출력 전에 거부하는지 확인한다.

지원 flag는 `--config`, `--device`, `--seed`, `--policy-steps`다. 기본 config는 설치
패키지에 포함되어 현재 작업 디렉터리에 의존하지 않는다. 저장소의
`configs/mujoco_robot_free.toml`은 byte-identical 검토/override 사본이고 현재 계약은
`policy_steps=1`을 요구한다.
출력 root에는 `robot_free_result.json`, offline candidate checkpoint/report, MuJoCo
G8 JSON report, exact `effective_config.toml`, `episodes/train` 및
`episodes/validation`의 episode `.npz`와 JSON manifest가 생긴다. G8은 bundle 안의
effective config를 다시 로드해 실행한다. `robot_free_result.json`의 `artifacts`
manifest가 모든 실제 bundle-relative 경로와 SHA-256을 기록하므로 검증 보고서는 그
manifest를 기준으로 인용한다. `training.checkpoint_path`와 `training.report_path`도
동일한 bundle-relative manifest 경로를 사용한다. 결과 JSON 자체는 재귀 self-hash
문제를 피하기 위해 해당 manifest에 포함하지 않는다.

결과 JSON은 `schema_version`, `result`, `mode`, `evidence_level`, `evidence_inputs`, `trained`,
`deployment_ready`, `synthetic_data`, `training`, `g8`, `real_output_authorized`,
`real_output_rejection_reason`, `artifacts`, `limitations`를 포함해야 한다.
`trained=false`, `deployment_ready=false`, `real_output_authorized=false`와 offline
candidate rejection reason이 없으면 real-output guard 검증으로 보지 않는다.

보관/이동 후 정적 무결성 확인:

```shell
so101-wam-verify-robot-free \
  --result runs/robot_free_001/robot_free_result.json
```

이 verifier는 result schema, bundle path containment, artifact hash, episode payload와
manifest, offline checkpoint/training report, exact G8 binding과 actuated rollout,
fresh real-output guard rejection을 검사한다. training과 MuJoCo G8을 다시 실행하지
않으며 hash consistency는 서명·출처 인증 또는 새로운 simulation/real evidence가
아니다.

이 결과는 offline/simulation evidence다. G6/G7/G9/G10의 real evidence, 실제
hardware safety, 또는 task success로 보고하지 않는다. CI는 이 명령을 장시간 별도
step으로 직접 중복 실행하지 않는다. robot-free 계약 검증은 전체 test suite에 둔다.

## Robot-free semantic suite

```shell
so101-wam-robot-free-semantic \
  --output-dir runs/robot_free_semantic_001
```

이 별도 acceptance는 synthetic train episode 2개와 validation episode 4개를 만들고,
validation의 `synthetic-place-left`/`synthetic-place-right`를 task-specific prompt로
사용한다. tiny offline checkpoint/report를 한 번 발행한 뒤 `task_block`과
`task_cylinder`를 각각 seed `7/13/29`로 production semantic runner에서 실행한다.
완료 기준은 schema-5 report, case/task/body/compiled-profile 각 2개, 총 6개 trial,
checkpoint/training/prompt/config hash 결합이다.

출력 root에는 exact `effective_config.toml`, episode bundle, candidate checkpoint/report,
semantic suite/mapping manifest, `semantic-artifacts/`, `semantic-suite-report.json`이 남는다.
mapping은 독립 검토 전까지 `unverified`이며 상위 `result="complete"`는 실행 완료만
뜻한다. case success, prompt causality, 실물 성공, 공식 재현을 뜻하지 않는다. 이 경로는
기존 `robot_free_result.json` v1 또는 `so101-wam-verify-robot-free`의 입력이 아니다.

## Paired human-video offline training

```shell
so101-wam-train-paired \
  --train-manifest data/pairs/train.json \
  --validation-manifest data/pairs/validation.json \
  --output checkpoints/paired_candidate_001.pt \
  --report reports/paired_candidate_001.training.json \
  --checkpoint-id paired-candidate-001
```

이 경로는 모든 pair의 `human_reviewed` 선언과 video/task-spec/robot checksum을 학습
직전에 다시 검사한다. train/validation은 pair, source hash, robot fingerprint,
task index와 label이 분리돼야 하며 validation에는 서로 다른 task가 두 개 이상 있어야
한다. 완료 기준은 task-balanced optimizer schedule, immutable checkpoint/report,
exact checkpoint hash, held-out matched/null/mismatched action 및 future-latent 지표다.

이 검증은 선언된 review의 독립성이나 semantic correctness를 판정하지 않는다. offline
delta는 prompt causality 또는 task success가 아니며 checkpoint는 계속
`trained=false`, `deployment_ready=false`다.

## Paired human-video semantic controls

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

입력 검증은 checkpoint/report digest와 architecture, task-disjoint inventory,
validation pair digest·전체 inventory, `human_reviewed` 선언과 task-spec checksum을
rollout 전에 결합한다. semantic dataset task와 일치하는 pair 및 label/index가 모두
다른 pair가 각각 필요하고 모든 human task-spec 해상도는 MuJoCo camera 설정과 같아야
한다. 이후 동일 semantic manifest와 seed schedule로
`matched_human_video`, `wrong_task_human_video`, `null_human_video`를 실행한다.

Task-spec session은 기존 runtime이 요구하는 MuJoCo home-hold physical prompt를 context
자리표시자로 유지하지만 wrapper policy가 `snapshot.live_frames`만
`CompactWAMPolicy.predict_task()`에 전달한다. child trial과 상위 report 모두
`runtime_placeholder_prompt_used_by_model=false`를 기록한다. 각 trial, child report,
checkpoint, training report, semantic manifest와 pair manifest는 SHA-256으로 결합되고
출력은 덮어쓰지 않는다. paired report schema 4는 각 child의 future-latent summary를
신뢰하지 않고 trace artifact를 다시 읽어 prediction/observation/aligned/censored count와
weighted MSE를 재계산한다. matched보다 latent MSE와 terminal success rate가 모두 엄격히
낮은 조건은 `future_latent_proxy_outcome_mismatch=true`로 분류한다. 정렬 pair 부재와
동률은 제외하며 condition의 child report SHA가 trial/trace SHA를 결합한다.

상위 `result="complete"`는 세 조건 실행과 artifact 기록의 완료만 뜻한다. 사람 review의
독립성, dataset→object 의미 대응, prompt causality, semantic held-out 성공, 실제 로봇
성공, pixel-space future video quality 또는 공식 Zero-WAM 재현을 뜻하지 않는다.

Runtime smoke command shape:

```shell
PYTHONPATH=src python -m so101_wam.cli --steps 2
PYTHONPATH=src python -m so101_wam.cli --steps 1 --actuate-fake
```

기대 출력은 10 Hz policy refresh당 50 Hz servo 5틱을 실행한다. 2-step shadow mode는
`servo_steps=10`, `sent_actions=0`; 1-step fake actuation mode는
`servo_steps=5`, `sent_actions=5`; 둘 다 남은 horizon을 명시적으로 폐기한 뒤
`state=rollout_ready`다.

fresh CI/local command가 green이면 G1-G5와 G8의 nominal simulation 절반은 green으로
주장할 수 있다. G6/G7 계측 코드와 fail-closed real camera-skew gate도 자동 test
범위에 들어간다. 하지만 실제 report, 영상 identity, 축 방향, G8의 실측 scene/real
slow-jog, G9 이후 prompt record/replay와 real rollout 증거가 없으므로 real gate 완료
주장은 금지다.

MuJoCo optional 환경에서는 bundled dual-arm scene compile, exact 12축 binding,
두 wrist RGB render, 200 Hz physics의 50 Hz servo당 4 substep, canonical
degrees/gripper-percent round trip, colliding target preflight의 무상태 거부,
transition contact의 전체 integration-state rollback, 10/50 Hz managed rollout을
`tests/test_mujoco.py`로 검증한다. 실행 절차와 모델 provenance는
[MUJOCO.md](MUJOCO.md)에 있다.

추가로 fake `BiSOFollower` lifecycle을 사용한 hardware-session test에서 checkpoint
load, 30→10 Hz prompt, connect, history prime, 10/50 Hz shadow rollout, disconnect가
통과했다. 이는 실제 USB/camera/motor 증거가 아니다.

`tests/test_hardware_preflight.py`는 normal `BiSOFollower.connect()`와
`send_action()`을 호출하지 않는 bus/camera 관측 경로, 동일 lock에서 복사된 frame과
timestamp의 FPS/skew/age 및 관측 조립 latency, home 반복성,
uncalibrated/missing timestamp fail-closed, 모든 예외의 disconnect, 원자적 report
no-overwrite를 fake hardware로 검증한다. 실제-output adapter는 timestamp 누락과
100 ms 초과 camera-buffer age를 거부하고 `SafetySupervisor`는 17 ms 초과 skew를
reject한다. 부분 bimanual connect 실패도 이미 연결된 arm을 독립적으로 해제한다.

`tests/test_prompt_recorder.py`는 양쪽 bus 연결 뒤 torque-off, 30 Hz atomic
관측, 실제 capture-start drift 5% gate, camera timestamp advance/age/skew gate,
측정 12축과 action proxy의 동일성, `Goal_Position` 0회, 예외 cleanup,
no-overwrite, checksum 및 30→10 Hz fingerprint replay를 fake hardware로 검증한다.
이는 recorder 경로의 증거이며 실제 SO-101 시연 데이터가 아니다.

`tests/test_training_data.py`, `tests/test_training.py`는 같은 task의 서로 다른
episode pairing, task-level train/validation holdout, 30→10 Hz future frame 및
runtime과 같은 row-0 즉시 시작의 50 Hz action 보간, IFP stride target, train-only
축 정규화, 2단계 gradient update, immutable checkpoint/report checksum을 검증한다.
trainer 산출물은 `artifact_kind="compact_wam_candidate"`,
`evidence_level="offline"`, `offline_trained=true`, `trained=false`,
`deployment_ready=false`이고 real-output checkpoint gate가 후보 표식 자체를
거부한다. 이는
합성 episode를 이용한 offline pipeline 증거이지 zero-shot task 성공 증거가 아니다.
`tests/test_ifp_preregistration.py`, `tests/test_ifp_study.py`는 result-dependent hash와
metric을 허용하지 않는 strict local plan, exact seed/split/optimizer/scope 대조,
K4-vs-K0 endpoint pass/fail, negative result 보존과 no-winner semantics를 검증한다.
`tests/test_ifp_ablation.py`는 같은 K 변형을 기존 multi-case semantic MuJoCo suite에
연결하고 total/scored/execution-failure count, failure taxonomy, suite report hash를
보존하는지 검증한다. study 집계는 success rate는 전체 trial, terminal object error는
scored trial을 분모로 사용한다.
plan hash는 local interpretation contract이며 외부 등록 시점, 충분한 budget,
object manipulation 또는 prompt causality 증거가 아니다.
후보 weights를 deployment-looking metadata로 다시 저장해도 별도 certification JSON이
없으면 로봇 생성 전에 거부한다. certification core digest는 checkpoint metadata에,
정확한 checkpoint bytes SHA-256은 외부 JSON에 결합된다.
`tests/test_deployment_issuer.py`는 exact model-state promotion, G6-G9/manual-signoff
source hash 결합, source 변조, 자동 check 실패, smoke-policy 대체, simulation-only
signoff, signoff 누락/false, G8 exact-session 전체 report 재실행 불일치,
no-overwrite와 pair rollback을 검증한다. `tests/test_hardware_cli.py`는 실제 출력에서
원본 source path 누락/대체를 robot 생성 전에 거부하는 것도 검증한다. 이는 issuer의
fail-closed 동작 증거이지 실제 hardware check가 수행됐다는 증거가 아니다.

## 정량 기준

| 항목 | 기준 |
| --- | ---: |
| Prompt duration | 3.000 s 이상, 12.000 s 이하 |
| Policy context | 30 s 이하 |
| Camera capture target | 30 Hz |
| Policy sampling target | 10 Hz |
| Servo interpolation target | 50 Hz |
| Action horizon MVP | 10 targets |
| Wrist camera skew | p95 <= 17 ms |
| Observation age | p95 <= 100 ms |
| Watchdog timeout | <= 100 ms |
| Dataset timestamp drift | frame step error <= expected dt의 5% |
| Required cameras | exactly 2 RGB wrist cameras |
| Required joint/action width | 12 |
| Real rollout report | 모든 trial success/failure/error 원본 로그 보존 |

이 값은 현재 architecture/config 계약과 fake/offline test 목표다. 실제 bus rate,
latency, thermal envelope, collision margin은 하드웨어 측정값으로 갱신해야 한다.

## Real hardware에서만 증명 가능한 항목

- operator halt 후 추가 명령이 전송되지 않는지
- 좌우 SO-101 feature가 실제 축 방향과 일치하는지
- arms-down neutral home pose의 실제 `.pos` 12개 값
- 손목 카메라 좌우 식별, focus, exposure, mount rigidity
- wrist roll 중 케이블 꼬임과 strain relief
- 팔-팔, 팔-토르소, 팔-테이블, 팔-카메라 mount 충돌 여유
- 30 FPS capture, 17 ms skew, 100 ms observation age가 장시간 유지되는지
- real task success rate와 failure taxonomy

## 보고 규칙

검증 로그는 다음 형식으로 남긴다.

| 필드 | 내용 |
| --- | --- |
| `evidence_level` | `fake`, `offline`, `simulation`, `real` 중 하나 |
| `gate` | G0-G11 |
| `hardware_id` | real일 때 arm/camera 조립체 식별자 |
| `config_id` | LeRobot calibration id 또는 zero01 config hash |
| `prompt_fingerprint` | rollout이면 필수 |
| `result` | `pass`, `fail`, `blocked` |
| `metrics` | skew, age, fps, success count 등 원시 수치 |
| `notes` | 실패 원인과 재현 조건 |

## 근거

- Local: `docs/ARCHITECTURE.md`, `docs/TRAINING.md`
- Local: `docs/MUJOCO.md`, `configs/mujoco.toml`,
  `src/so101_wam/adapters/mujoco.py`, `tests/test_mujoco.py`
- Local: `configs/fake.toml`, `configs/lerobot_hardware.template.toml`,
  `docs/VIDEO_REFERENCE.md`
- Local: `src/so101_wam/constants.py`, `src/so101_wam/config.py`,
  `src/so101_wam/contracts.py`, `src/so101_wam/context.py`,
  `src/so101_wam/model.py`, `src/so101_wam/dataset.py`,
  `src/so101_wam/training_data.py`, `src/so101_wam/training.py`,
  `src/so101_wam/deployment.py`, `src/so101_wam/deployment_issuer.py`,
  `src/so101_wam/certify_cli.py`, `src/so101_wam/hardware_doctor.py`,
  `src/so101_wam/prompt_recorder.py`,
  `src/so101_wam/adapters/lerobot.py`, `src/so101_wam/safety.py`,
  `src/so101_wam/tensorizer.py`, `src/so101_wam/policy.py`,
  `src/so101_wam/runtime.py`
- Local tests: `tests/test_config.py`, `tests/test_contracts.py`,
  `tests/test_context.py`, `tests/test_model.py`, `tests/test_adapters.py`,
  `tests/test_dataset.py`, `tests/test_safety.py`, `tests/test_tensorizer.py`,
  `tests/test_policy.py`, `tests/test_runtime.py`, `tests/test_cli.py`
- Local preflight tests: `tests/test_hardware_doctor.py`,
  `tests/test_hardware_preflight.py`,
  `tests/test_prompt_recorder.py`, `tests/test_hardware_cli.py`,
  `tests/test_deployment.py`, `tests/test_training_data.py`,
  `tests/test_training.py`, `tests/test_deployment_issuer.py`,
  `tests/test_rollout.py`
- Paper: [Zero-WAM](https://arxiv.org/abs/2608.26103) p.1-p.2, human video as in-context task
  specification; p.7, future video chunk plus inverse dynamics factorization;
  p.9, prompt cache and inference flow; p.11, 원 논문 평가 결과.
- Paper: [MACT](https://arxiv.org/abs/2411.04050) p.3-p.5,
  historical image buffer와 near-future action chunking.
- Official source: [LeRobot v0.6.1 `BiSOFollower`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/bi_so_follower/bi_so_follower.py)
- Official source: [LeRobot v0.6.1 `BiSOLeader`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/teleoperators/bi_so_leader/bi_so_leader.py)
- Official source: [LeRobot v0.6.1 SO follower](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py)
- Official: [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Official: [XLeRobot documentation](https://xlerobot.readthedocs.io/en/latest/)
- Official: [Zero-WAM project](https://robbyant-research.github.io/Zero-WAM/)
- Official: [MuJoCo Menagerie SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
