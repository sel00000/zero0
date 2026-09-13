# Zero-WAM 구현 갭 분석

- 기준 시각: 2026-09-01 KST
- 감사 대상 코드 기준 리비전: `4273574d39fc6f2e6fd3ae56701cebad1baf64d1`
- 목적: 다음 구현·검증 작업의 우선순위 결정
- 범위: 공식 Zero-WAM 방법/공개 자산, 필수 주변 기술, 현재 로컬 코드와 실행 증거

## 결론

현재 저장소는 Zero-WAM의 핵심 아이디어 중 **causal future visual latent → inverse dynamics → action chunk** 분해를 작고 안전한 형태로 보존한, 잘 경계된 연구 프로토타입이다. 그러나 공식 방법의 인간 비디오/언어 입력, Wan-2.2 기반 Mixture-of-Transformers(MoT), flow-matching 학습, HumanGen과 task-balanced 데이터, 논문형 IFP의 유효성, 폐루프 미지 과제 성공 증거는 아직 없다. 따라서 현재 위치는 “공식 Zero-WAM 재현”이 아니라 **축소 구현과 실행·안전 검증용 발판**이다.

가장 먼저 해야 할 일은 더 큰 모델을 붙이는 것이 아니라, 로봇 없이도 가능한 **폐루프 held-out MuJoCo 과제 성공 벤치마크**, **프롬프트 의존성 반증 실험**, **인간 비디오/언어 입력 계약 분리**, **IFP 효과 검증 및 데이터 샘플링 ablation**이다. 현재 문서는 비공식 구현과 실제 SO-101 성공 부재를 분명히 밝히고 real-output을 fail closed하므로, 확인된 P0 오류나 과장 주장은 없다. 다만 이 경계가 약해지거나 offline/simulation 결과를 실제 성공으로 표현하면 즉시 P0가 된다.

## 범위와 현재 구현 기준선

### 공식 방법의 비교 기준

[Zero-WAM 논문](https://arxiv.org/pdf/2608.26103)과 [공식 프로젝트](https://robbyant-research.github.io/Zero-WAM/)가 설명하는 정책은 언어 또는 인간 시연 비디오를 task specification으로 받아 미래 로봇 비디오와 실행 action chunk를 함께 예측하는 causal video-action model이다. 인간 비디오 표현은 추론 때 한 번 계산해 prefix memory로 캐시하고, action Transformer는 인간 비디오를 직접 보지 않고 예측된 미래 로봇 비디오를 매개로 행동을 복원한다. 공식 모델은 Wan-2.2-TI2V-5B를 causal policy로 변환하고 비디오/행동 파라미터를 분리한 MoT와 flow-matching objective를 사용한다.

공식 IFP는 training-only auxiliary objective이며 `K=4`, stride 2를 사용하고 추론 때 모듈을 제거한다. 논문은 HumanGen 74.2K 인간-로봇 ICL pair/8.6K task와, 6K가 넘는 task에서 epoch당 약 400K trajectory를 task-balanced sampling한 학습 규모를 보고한다. 또한 pretraining 15,360 GPU-hours와 RoboTwin post-training 64 GPU/4,000 step을 보고하므로, 이 방법의 완전 재현과 workstation-scale prototype은 별개다.

공식 평가 역시 로컬 smoke와 직접 비교할 수 없다. 논문은 RoboTwin 2.0의 50개 중 7개 task를 hold out하고 Zero-WAM 46.95%, LingBot-VA 17.45%, WAN-Action 10.98%를 보고한다. 실물 평가는 SO-101이 아니라 bimanual Franka에서 task-family별 30 trial로 보고되며, 논문 자체도 stationary tabletop 중심이라는 한계를 밝힌다. 이 수치는 논문 보고값이지 이 저장소에서 재현된 결과가 아니다.

### 로컬 기준선

[README 1–15행](../README.md)은 이 저장소를 비공식 독립 구현으로 명시하고 실제 SO-101 성공을 주장하지 않는다. [아키텍처 18–30행](ARCHITECTURE.md)은 5B Wan/pixel generation 대신 compact latent model을 쓰며, 기본 `K=2` compatibility head와 별도 removable `K=0/2/4` fused ablation을 구분한다.

| 구성 요소 | 현재 책임과 근거 | 확인된 경계 |
| --- | --- | --- |
| Task prompt | 두 wrist RGB, 12축 proprioception, executed action을 가진 3–12초 robot episode; [README 6–15행](../README.md), [training data](../src/so101_wam/training_data.py) | 인간 비디오 전용 또는 언어 입력이 아님 |
| Causal context | prompt slot을 고정하고 live frame FIFO를 causal mask로 처리; [architecture](ARCHITECTURE.md), [model](../src/so101_wam/model.py) | pre-temporal prompt token은 fingerprint별 재사용하지만 full temporal attention은 매 호출 재계산 |
| Future prediction | shared small CNN과 one-layer Transformer의 dual-wrist latent prediction | Wan pixel/video generation, MoT, flow matching이 아님 |
| Action decoding | 예측 future latent, 현재 proprioception, action history로 action chunk 복원 | 공식 factorization의 compact analogue |
| IFP | 기본 compact `K=2` head와 별도 training-only `K=0/2/4` fused-module ablation, stride 2 | multi-layer summary를 쓰는 latent-MSE 축소이며 공식 token-level/flow-matching 구현과 다름 |
| Offline data | same-task/different-episode pair, task-id/label-disjoint split, checksum manifest, reviewed multi-pair human-video candidate와 offline held-out prompt controls | 공식 HumanGen, 독립 semantic review, task-balanced multi-dataset pretraining 없음 |
| MuJoCo G8 | vendored Menagerie SO-101, dual-arm actuation, collision preflight/rollback; [MuJoCo 문서](MUJOCO.md) | nominal simulation evidence일 뿐 sim-to-real 증거가 아님 |
| Robot-free path | deterministic synthetic 32×24 episodes, stage 1은 0 step, stage 2는 1 step, MuJoCo policy 1 step; [G8 CLI](../src/so101_wam/robot_free_cli.py)와 [multi-object semantic CLI](../src/so101_wam/robot_free_semantic_cli.py) | 두 validation task의 terminal object state를 측정하지만 작은 synthetic negative benchmark라 학습 성능·일반화를 입증하지 않음 |
| Evidence gates | candidate는 `trained=false`/`deployment_ready=false`; G6–G9와 certification 없이는 real output 거부; [certification](CERTIFICATION.md) | software gate는 실제 하드웨어 안전 인증이 아님 |
| Hardware path | doctor, read-only preflight, G6–G11 artifact 계약과 exact-candidate binding | 실제 장치 증거가 아직 없음 |

기준 리비전에서 전체 test suite가 통과했다. 별도의 robot-free smoke도 offline candidate, actuated MuJoCo G8 1 step, real-output guard rejection을 생성하고 정적 bundle verifier를 통과했다. 이는 소프트웨어·offline·simulation 계약의 실행 증거이며, 공식 checkpoint 호환성, RoboTwin 성능, 실제 SO-101 안전 또는 task success 증거가 아니다.

## 방법론 추적성 매트릭스

상태는 `일치`, `축소 구현`, `의미 있는 이탈`, `미구현`, `현재 검증 불가` 중 하나다. 우선순위는 현재 저장소가 주장하는 범위를 기준으로 판정했으며, 공식 재현을 주장할 경우 일부 P1/P2는 P0로 상승한다.

| 공식 요구 또는 비교 항목 | 직접 근거와 로컬 매핑 | 상태·신뢰도 | 영향·의존성 | 우선순위와 관찰 가능한 완료 조건 |
| --- | --- | --- | --- | --- |
| 언어/인간 비디오 통합 task interface | [논문](https://arxiv.org/pdf/2608.26103) §3.1; 로컬은 action-free RGB/timestamp artifact와 language/robot task spec이 동일 offline policy boundary를 통과하고 reviewed human-video candidate를 학습 | **축소 구현**, 높음 | task-disjoint offline 지표와 matched/wrong/null human-video MuJoCo object-state 경로는 있으나 language 학습, 독립 semantic mapping과 open-task 능력은 없음 | **P1 잔여** — 독립 검토된 여러 held-out task의 폐루프 prompt control과 tokenizer/encoder 근거를 검증 |
| 인간 비디오 prefix memory와 간접 action conditioning | 논문 §3.2/3.4; 로컬은 pre-temporal prompt token을 fingerprint별 재사용하고 action head에서 raw prompt를 제외 | **축소 구현**, 높음 | cached/non-cached parity는 있으나 공식 prefix representation/KV cache와 다르고 temporal attention은 전체 prefix를 재계산 | **P2 잔여** — temporal KV/prefix cache를 추가하고 non-cached parity와 반복 호출 latency를 함께 검증 |
| 미래 비디오 → action 분해 | 논문 §3.1; [local model](../src/so101_wam/model.py)은 future wrist latent 후 action chunk를 예측 | **축소 구현**, 높음 | 핵심 구조는 보존, pixel-space/large-video semantics는 없음 | **P2** — future-only, direct-action baseline과 held-out task 성능/실패 mode를 같은 protocol로 비교 |
| Wan-2.2-TI2V-5B causal conversion과 MoT | [논문](https://arxiv.org/pdf/2608.26103), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [TI2V-5B card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B); 로컬 dependency/모듈 없음 | **미구현**, 높음 | 공식 checkpoint/representation fidelity 없음; 고성능 GPU 및 공식 자산 의존 | **P2** — replaceable adapter에서 video/action parameter stream, causal mask, state-dict mapping을 검사하고 bounded feasibility run 성공 |
| Video/action flow-matching losses | 논문 §3.4; 로컬은 latent MSE와 normalized action loss | **미구현**, 높음 | 공식 학습 dynamics와 호환되지 않음; GPU/데이터 의존 | **P2** — objective별 unit test, overfit smoke, compact MSE 대비 동일 benchmark ablation 완성 |
| IFP training-only, `K=4`, stride 2 | 논문 §3.3/4.1; 로컬은 기본 compact `K=2`, 별도 fused `K=0/2/4` path, inference module 제거, ≥3 seed report 집계와 result-independent local plan을 제공 | **축소 구현**, 높음 | 동일 multi-case semantic object-state suite를 각 K에 연결하고 scored/execution failure를 보존할 수 있지만 외부 등록 시점, 실제 충분-budget 결과, 독립 task mapping과 shortcut 억제 효과는 미검증; robot 불필요 | **P1 잔여** — 사전 고정한 semantic suite와 ≥3 seed에서 충분-budget prompt sensitivity, future/action error, terminal success를 함께 비교 |
| HumanGen 인간-로봇 ICL pair | 논문 §2.1–2.3의 74.2K pair/8.6K task; 로컬은 checksum/provenance compatible manifest, pair/task-disjoint split, reviewed multi-pair candidate, task-balanced schedule, offline deltas와 3조건 MuJoCo task-spec 진단을 제공 | **축소 구현**, 높음 | immutable artifact와 local closed-loop 경로는 있으나 semantic matching 생성·독립 판정, 대규모 데이터와 여러 held-out task의 성능은 없음 | **P1 잔여** — 독립 review와 여러 semantic task의 matched/mismatched 폐루프 결과를 검증 |
| Task-balanced multi-dataset sampling | 논문 §2.2의 >6K task/~400K trajectory per epoch; 로컬은 task-first balanced window sampler와 task-disjoint split을 사용 | **축소 구현**, 높음 | per-task optimizer draw 편향은 통제하지만 공식 multi-dataset 규모·분포는 재현하지 않음 | **P1 잔여** — 공개 multi-dataset adapter에서 동일 histogram/cursor audit를 유지하고 task leakage test 통과 |
| Same-task/different-episode와 task-disjoint 평가 | [training data](../src/so101_wam/training_data.py)에서 episode fingerprint, task id/label leakage를 거부 | **일치**, 높음 | 좋은 local anti-leakage 기반; 규모는 작음 | **유지** — split manifest와 negative leakage tests를 모든 새 importer에 재사용 |
| 폐루프 unseen-task benchmark | 논문 §4.2는 7 held-out RoboTwin task 결과를 보고; 로컬은 joint proxy와 별도로 manifest-selected named body의 3D position criterion, seed별 object 초기값, artifact/CI, checkpoint-bound training task inventory와 prompt task identity, collision/safety/watchdog/policy failure taxonomy를 제공 | **축소 구현**, 높음 | schema는 unit/fake injection, 실제 6개 joint-limit rejection, production path의 실제 target-preflight collision과 stale-action watchdog으로 검증했다. production suite는 물성이 다른 `task_block`/`task_cylinder`에 deterministic offline checkpoint를 각 3개 seed로 실행하고 서로 다른 compiled profile hash와 negative outcome을 보존한다. dataset→object mapping은 둘 다 `unverified`이며 독립 검토와 learned semantic success는 아직 없다 | **P1 잔여** — 독립 task mapping과 의미·물리가 더 다양한 held-out object task의 실제 폐루프 결과 |
| 현실에서의 실행 가능성 | [RoboWM-Bench](https://robowm-bench.github.io/RoboWM-Bench/)와 [EVA](https://eva-project-page.github.io/)는 시각적 일관성과 물리 실행 가능성의 간극을 지적; 로컬은 decoded/servo/executed action stream hash, safety 판정, normalized joint-limit margin, bounded contact progression, time-aligned future latent MSE와 terminal outcome을 같은 rollout id로 결합 | **축소 구현**, 높음 | paired report schema 4는 latent MSE 개선과 terminal success 퇴행을 자동 분류하고 child report SHA를 통해 trial/trace SHA에 결합하지만 pixel-space future video metric은 없음 | **P2 잔여** — 동일 rollout의 pixel-space future visual metric과 latent-proxy 판정 차이를 비교 |
| 공식 code/model/data와 checkpoint 호환성 | [공식 저장소](https://github.com/robbyant-research/Zero-WAM)는 기준일에 paper만 공개하고 code/model/data를 2026-09-15 전으로 예고; tag/release 없음 | **현재 검증 불가**, 높음 | 공식 state dict, config, data schema가 없어 compatibility 주장 불가; 외부 release 의존 | **P1(공개 후)** — 공식 release/tag를 고정하고 license/hash/schema/forward semantics diff와 최소 inference parity 보고서 완성 |
| LeRobotDataset v3 데이터 호환 | [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)는 Parquet state/action/timestamp와 MP4 camera를 표준화; 로컬은 이미 변환된 [NPZ/JSON](../src/so101_wam/dataset.py)을 compatible pair manifest에 연결 | **의미 있는 이탈**, 높음 | manifest round-trip은 있으나 Parquet/MP4 shard reader와 metadata offset 검증이 없음 | **P2** — lossless dataset importer/exporter, timestamp/camera/action key validation, checksum parity test 통과 |
| SO-101 calibration/identity 운영 계약 | [SO-101](https://huggingface.co/docs/lerobot/so101)과 [LeRobot 실물 workflow](https://huggingface.co/docs/lerobot/il_robots)는 calibration과 stable robot/teleoperator ID를 요구; 로컬 G6/G7 gate와 대응 | **일치**, 높음 | 올바른 하드웨어 전제이나 아직 실측 증거 없음; robot 의존 | **P1, 2단계** — 두 arm/camera의 고유 ID, calibration artifact, home repeatability, sign/unit bench가 실제 장치에서 통과 |
| Menagerie 모델과 sim-to-real dynamics | [Menagerie](https://github.com/google-deepmind/mujoco_menagerie)는 curated model을 제공하지만 [MuJoCo XML](https://mujoco.readthedocs.io/en/latest/XMLreference.html)의 transmission/control range/gain/damping semantics가 여전히 필요; 로컬은 upstream commit을 그대로 vendoring | **현재 검증 불가**, 높음 | provenance는 좋지만 실측 geometry/dynamics가 없음; robot 의존 | **P1, 2단계** — measured geometry/extrinsics/joint signs/offsets/gripper/latency를 반영하고 slow-jog trajectory 오차 기준 통과 |
| ACT/MACT action chunk 근거 | [ACT](https://arxiv.org/pdf/2304.13705)는 chunking으로 effective horizon을 줄임; [MACT](https://arxiv.org/pdf/2411.04050)는 tissue-scanning 전용 결과 | **현재 검증 불가**, 높음 | action chunk 설계 동기는 있으나 MACT 성능을 dual-SO-101에 전이할 근거 없음 | **P2** — README에서 “설계 영감”으로 한정하고 direct action/ACT-like/future-mediated head를 동일 benchmark로 비교 |
| Fail-closed evidence와 claim boundary | [README 140–167행](../README.md), [certification](CERTIFICATION.md), [MuJoCo collision gate](MUJOCO.md)는 candidate/real evidence를 분리 | **일치**, 높음 | 현재 가장 강한 로컬 확장; 유지 실패 시 안전·연구 타당성 훼손 | **유지/P0 보호선** — negative tests가 모든 missing/stale/mismatched evidence와 offline candidate의 real output을 계속 거부 |
| 실제 SO-101 task success | 로컬 문서가 성공 부재를 명시하고 G6/G7/G9/G10/G11은 미실행 | **현재 검증 불가**, 높음 | 실물 성능과 안전을 주장할 수 없음; robot/운영 승인 의존 | **P1, 2단계** — preregistered G11 trial, 실패 포함 raw evidence, seed/config별 성공률과 prompt control 결과 공개 |

## 확인된 일치점과 강점

1. **주장 경계가 정직하다.** 저장소 첫 문단부터 비공식 구현과 실물 성공 부재를 밝히며, robot-free 결과에도 offline/simulation label을 유지한다. 이 때문에 큰 방법론 공백이 곧바로 허위 재현 주장으로 이어지지 않는다.
2. **핵심 factorization의 의미는 남아 있다.** [InverseDynamicsActionHead](../src/so101_wam/model.py)는 raw prompt tensor 대신 예측 future latent, robot history, proprioception을 사용한다. 이름만 차용한 것이 아니라 “미래 시각 상태가 action decoding을 매개한다”는 구조를 실제 코드로 보존한다.
3. **데이터 누출 방지가 구체적이다.** [training_data](../src/so101_wam/training_data.py)는 same-task/different-episode pair를 만들고 episode fingerprint, task id, task label의 train/validation 중복을 거부한다.
4. **simulation provenance와 실패 처리가 강하다.** Menagerie SO-101의 정확한 upstream commit을 기록하고, actuator step 전에 collision preflight를 수행하며 중간 충돌 시 integration state를 rollback한다.
5. **artifact와 real-output 결합이 fail closed다.** candidate checkpoint, G6–G9 evidence, certification, prompt, config, hash가 맞지 않으면 robot object 생성 전에 거부한다. `trained=false`를 boolean 하나로 뒤집어 배포할 수 없도록 artifact kind를 별도로 검사한다.
6. **실행 가능한 regression 기반이 있다.** 전체 test suite와 robot-free bundle verifier가 software/offline/simulation 계약을 반복 검증한다. 이 증거의 범위를 문서가 과장하지 않는 점도 강점이다.

## 중요한 이탈·누락·과장 위험

### 1. 기본 deployment prompt는 인간 비디오가 아니라 로봇 시연이다

기본 deployment prompt에는 두 wrist RGB뿐 아니라 12축 proprioception과 executed action이 들어간다. 별도 paired 경로는 action-free human-video task-spec으로 학습하고 MuJoCo semantic rollout에도 직접 연결하지만 real deployment 계약은 바꾸지 않는다. 공식 Zero-WAM의 인간 시연 비디오는 로봇 embodiment state/action 없이 task semantics를 전달한다. 따라서 기존 physical-prompt 결과를 “인간 영상에서 robot action을 추론했다”고 해석할 수 없다. paired 경로도 작은 local model과 선언된 review에 한정되므로 공식 연구 질문을 완전히 재현하지 않는다.

### 2. 프롬프트를 실제로 사용하는지 증명되지 않았다

Same-task pair와 IFP가 있어도 모델이 background, target episode history, proprioception 같은 shortcut을 사용할 수 있다. matched/mismatched/shuffled/null/counterfactual prompt를 바꿨을 때 held-out terminal success가 예상 방향으로 변하는지 보여야 한다. 단순 latent MSE 감소나 prompt tensor가 graph에 연결됐다는 test는 인과적 사용 증거가 아니다.

### 3. 생성·행동 학습 objective가 공식 방법과 다르다

작은 CNN/Transformer, latent MSE, normalized action loss는 빠른 local iteration에는 합리적이다. 그러나 Wan-2.2 causal conversion, video/action MoT, flow matching을 제거했기 때문에 official checkpoint나 representation behavior와의 등가성을 주장할 수 없다. 이 차이는 문서에 이미 공개되어 있으므로 현재는 P2 fidelity gap이지만, 공식 재현을 주장하는 순간 P0 claim error가 된다.

### 4. IFP 구조는 확장됐지만 효과는 미검증이다

기본 compatibility path는 `K=2` linear head를 유지한다. 별도 ablation path는 `K=0/2/4`, stride 2, main temporal layer에서 초기화한 parallel module, multi-layer fused summary, inference module removal을 구현하고 같은 initialization/split/schedule/budget을 검사한다. closed-loop는 단일 joint proxy와 기존 multi-case semantic object-state suite 중 하나를 선택하며, semantic 경로는 scored/execution-failure 분리와 failure taxonomy를 보존한다. 별도 집계기는 ≥3개 고유 seed report의 split, optimizer, metric schema, closed-loop protocol을 대조하고 scored trial 기준 object error와 전체 trial 기준 success rate를 집계하지만 winner를 고르지 않는다. 실제 충분-budget multi-seed 결과는 아직 없다. official token-level representation과 flow matching이 아닌 compact latent MSE이며, local mapping의 semantic suite도 prompt shortcut 감소나 held-out 일반화를 증명하지 않는다. 의미 있는 결론에는 충분한 budget의 여러 seed, 독립 task mapping과 task-level prompt control이 필요하다.

### 5. 데이터 다양성과 sampling이 핵심 병목이다

공식 결과는 단순 trajectory 수가 아니라 task diversity와 human-robot pair 규모에 기대고 있다. 현재 deterministic synthetic episode와 소수 local episode는 pipeline smoke에는 충분하지만 open-ended generalization을 지지하지 않는다. 공식 자산 공개 전에는 HumanGen “재현”을 주장하지 말고, provenance가 있는 paired schema, task-balanced sampler, 작은 공개 subset importer를 먼저 검증해야 한다.

### 6. G8 pass는 task success가 아니다

현재 robot-free 기본은 stage 1 학습 0 step, stage 2 학습 1 step, MuJoCo policy 1 step이다. 별도 local benchmark는 manifest-selected named body의 최종 3D 위치를 자동 판정하고 checkpoint training inventory와 prompt task identity를 결합한다. 같은 object-state criterion의 7조건 prompt-control도 있다. multi-case suite는 서로 다른 dataset task와 task-specific prompt를 별도 단일-task manifest로 실행하고 local mapping artifact와 child report를 hash로 묶는다. mapping schema 3은 body, 초기/목표 object position과 tolerance signature를 검증한다. suite report schema 5는 이름을 제외한 compiled profile hash로 `task_block`과 `task_cylinder`의 simulator 물리 다양성을 별도 기록한다. schema-3 trial은 collision/safety/watchdog/policy failure를 구조화하고 production checkpoint/session 통합 test는 실제 target-preflight collision과 stale-action watchdog도 관측한다. robot-free semantic CLI는 합성 학습 checkpoint와 두 validation task의 6개 trial을 한 bundle로 발행한다. 다만 mapping은 독립 검증되지 않았고 budget은 한 optimizer step이며 현재 outcome은 작은 synthetic negative evidence다. 시각적으로 그럴듯한 rollout도 물리적으로 실행 불가능할 수 있다는 [RoboWM-Bench](https://robowm-bench.github.io/)와 [EVA](https://eva-project-page.github.io/)의 지적 때문에, 다음 evidence 단위는 독립 검토된 task mapping과 의미·물리가 다양한 object task의 실제 폐루프 outcome이어야 한다.

### 7. 공식 자산 공개 전 호환성은 검증 불가다

기준일의 [공식 저장소](https://github.com/robbyant-research/Zero-WAM)는 code/model/data 공개를 2026-09-15 전으로 예고하지만 아직 문서와 license만 있고 tag/release가 없다. 공개 Wan2.2-TI2V-5B는 upstream video model이지 Zero-WAM causal/MoT/action conversion이나 checkpoint가 아니다. 따라서 “official-ready boundary”는 설계 의도일 뿐 현재 검증된 호환성이 아니다.

### 8. Franka 결과와 SO-101 결과는 분리해야 한다

공식 실물 증거는 bimanual Franka이고, 로컬 embodiment는 dual SO-101이다. joint topology, payload, backlash, control bandwidth, camera mount, calibration과 workspace가 다르므로 공식 성공률을 baseline처럼 직접 전이하면 안 된다. SO-101은 별도의 protocol과 confidence interval을 가져야 한다.

## P0-P3 우선순위

### P0 — 현재 확인된 항목 없음

현재 문서와 runtime은 비공식·offline·simulation·real evidence를 명확히 분리한다. 다음 상황은 즉시 P0로 취급한다.

- official code/checkpoint 호환성 검증 없이 “공식 구현” 또는 “재현”이라고 표기;
- G8 smoke나 test pass를 real SO-101 task success로 표현;
- `trained=false`, artifact kind, G6–G9, certification, hash binding 중 하나를 우회해 real output 허용;
- train/validation task 또는 episode leakage를 허용한 채 zero-shot 결과 보고.

### P1 — 다음 실험을 막는 핵심 공백

| 순서 | 작업 | 이유 | 완료 증거 |
| ---: | --- | --- | --- |
| 1 | 폐루프 held-out MuJoCo mapping·task 확장 | 한 명령이 synthetic 학습 checkpoint와 두 named body의 6개 trial/compiled profile을 묶지만 mapping이 `unverified`이고 한-step negative evidence에 한정됨 | 독립 task mapping, 의미·물리가 더 다양한 여러 task, 충분한 학습 budget, ≥3 seed, rollout bundle, 성공률/CI |
| 2 | 프롬프트 인과성 control suite 잔여 | 7개 조건은 joint proxy와 semantic object-state에 모두 연결했고 둘 다 local hash-bound 방향성 plan을 지원하지만 외부 사전등록은 없음 | 독립 mapping의 여러 semantic task에서 외부 timestamp가 있는 사전등록 판정 |
| 3 | human-video/language conditioned checkpoint 잔여 | reviewed multi-pair checkpoint와 matched/wrong/null human-video semantic closed loop는 구현됐지만 local 단일 task·선언 review이며 language 학습은 없음 | 독립 검토된 여러 held-out task의 폐루프 평가, tokenizer/encoder 선택 근거 |
| 4 | IFP K=0/2/4 효과 ablation | 구조·비교·≥3 seed 집계와 local plan은 있으나 실제 충분-budget 결과와 shortcut/task 효과는 불명 | ≥3 seed, 충분한 학습 budget, task-level closed loop와 prompt control을 결합한 외부 시점 고정 결과 |
| 5 | official release intake | 현재 호환성 검증 자체가 불가능 | release/tag/hash 고정, license 검토, schema/state-dict/forward semantic diff |
| 6 | 실물 G6–G11 검증 | SO-101 성능·안전은 현재 미확인 | 2단계의 preregistered hardware evidence chain |

### P2 — fidelity·재현성 개선

- temporal KV/prefix cache와 cached/non-cached parity·반복 호출 latency;
- replaceable Wan-2.2/MoT/flow-matching adapter와 bounded GPU feasibility smoke;
- LeRobotDataset v3 import/export와 metadata round-trip;
- direct action, ACT-like, future-mediated head의 동일 benchmark 비교;
- MACT를 일반 성능 근거가 아니라 tissue-scanning에서 가져온 설계 영감으로 문서화;
- simulation parameter/version/report schema 고정과 richer executability diagnostics.

### P3 — 핵심 증거 후의 장기 항목

- mobile, dynamic, unstructured, substantially longer-horizon setting;
- head camera와 추가 modality ablation;
- 대규모 throughput/latency 최적화와 distributed training;
- 새로운 embodiment 확장과 task generator 자동화.

## 1단계: 로봇 없이 진행할 작업

아래 순서는 실제 robot 없이 수행할 수 있다. GPU, 외부 데이터, 공식 release 의존성은 별도로 표시했다.

### 1. 폐루프 benchmark를 먼저 만든다

- 최소 pilot protocol을 사전에 고정한다: train/held-out task 분리, 각 held-out task 3개 이상 seed, 자동 terminal success checker, timeout/failure reason, full rollout artifact.
- 처음부터 공식 7 task × 100 rollout을 흉내 내기보다, 계산 가능한 pilot 규모를 “local benchmark”로 명명하고 protocol이 안정된 뒤 확장한다.
- `future-mediated`, direct-action, null-policy baseline을 동일 task/seed에서 비교한다.
- 완료 조건: 한 명령으로 재현 가능한 report가 task/seed별 성공, collision, timeout, action rejection, latency와 confidence interval을 낸다.

### 2. 프롬프트 사용 여부를 반증 가능하게 만든다

- matched, same-task-wrong-episode, wrong-task, temporal shuffle, frame shuffle, null, counterfactual prompt를 생성한다.
- 현재 robot-free 진단은 이 7개 조건을 immutable episode로 저장하고 동일 checkpoint를 MuJoCo joint-target과 object-state task/seed에 각각 실행해 success-rate와 terminal-error 차이를 남긴다.
- target history와 배경 shortcut을 분리하도록 object/texture/camera randomization과 controlled pair를 사용한다.
- latent/action loss뿐 아니라 terminal task success와 action sequence divergence를 함께 본다.
- 선택적 로컬 plan은 입력 hash, 최소 trial, 비열등성과 negative margin을 첫 rollout 전에 고정하고 negative result도 그대로 남긴다.
- object-state 경로도 별도 claim scope
  `preregistered_mujoco_object_state_directionality_only`를 쓰며
  `success_rate`와 `object_position_error_mean_m`만 판정한다. dataset→object 의미
  대응은 manifest 선언이고 외부 등록 timestamp를 검증하지 않으므로 인과성 완료
  증거가 아니다.
- 완료 조건: 사전 정의된 기대 순서를 만족하거나, 만족하지 못했다는 negative result를 그대로 artifact로 남긴다.

### 3. 공식 task-spec interface를 별도 경로로 추가한다

- `HumanVideoPrompt`: RGB frame, timestamp, optional text metadata만 허용하고 joint state/executed action을 schema 수준에서 금지한다.
- `LanguagePrompt`: text/token representation을 기존 robot context와 분리하고 같은 task-spec boundary에서 선택한다.
- 기존 sensorimotor prompt는 `RobotEpisodePrompt`로 명시해 baseline으로 유지한다.
- 완료 조건: 세 prompt type의 shape/provenance/no-leakage test와 matched/mismatched inference smoke가 통과한다.

### 4. IFP를 paper-inspired structural ablation으로 확장한다

- 구현 상태: `K=0/2/4`, stride 2, separate future modules/fused summary,
  동일 initialization/split/schedule/optimizer budget 검사와 inference module
  removal은 [IFP ablation](IFP_ABLATION.md)에 구현됐다. ≥3개 고유 seed의
  split/optimizer/metric/evaluation protocol을 검사하고 source hash와 기술 통계를
  남기는 immutable 집계기도 제공한다. 선택적 strict local plan은 result hash/metric을
  금지하고 exact seed/split/optimizer/scope와 K4-vs-K0 endpoint margin을 결속한다.
- 남은 과학적 완료 조건: 충분한 학습 budget과 ≥3 seed를 사전 고정한 뒤,
  task-level prompt sensitivity, future error, action error, closed-loop success를
  함께 비교한다. local plan은 외부 등록 시점을 증명하지 않고 집계기 존재도 실험
  결과가 아니다. multi-case semantic suite 연결도 local mapping 기반 simulation
  진단이며 독립 semantic review나 held-out 일반화 증거가 아니다.

### 5. 데이터 계층을 확장한다

- 현재 checksum과 task-disjoint validation을 유지하면서 LeRobotDataset v3 reader/writer 경계를 추가한다.
- task-balanced sampler는 task별 draw count, schedule hash, sampler cursor를 training report에 남긴다. 이 cursor는 optimizer checkpoint가 아니다.
- HumanGen-like pair는 “공식 HumanGen”이 아닌 provenance/checksum 기반 compatible manifest와 task-disjoint split으로 시작했다.
- `human_reviewed` pair는 action-free RGB prompt와 train-mean neutral axis로
  task-balanced multi-pair 학습, immutable checkpoint/report, task-disjoint
  matched/null/mismatched offline 지표까지 연결된다. 이 delta는 semantic task success나
  prompt causality가 아니다.
- paired checkpoint는 validation pair inventory를 다시 결합한 뒤 action-free prompt를
  matched/wrong/null 3조건의 MuJoCo object-state trial에 직접 전달한다. runtime physical
  prompt는 context 자리표시자이며 모델 입력이 아니다. 이 local closed loop도 독립
  semantic review나 성공 주장이 아니다.
- 남은 완료 조건: LeRobot v3 import→local→export round-trip, timestamp/camera/action
  semantics, 공개 multi-dataset sampling histogram test가 통과한다.

### 6. 실행 가능성 지표를 연결한다

- 구현 상태: checkpoint semantic trial은 동일 rollout id에 bounded decoded action hash,
  전체 policy/servo/executed action stream hash, sent/shadow 수, safety
  accept/reject/clip 수와 reason, normalized joint-limit margin, bounded contact progression,
  collision evidence와 terminal outcome을 결합한다. trace는 별도 immutable artifact이며
  safety/collision 중단에도 partial evidence를 보존한다. opt-in policy telemetry는
  predicted future latent를 같은 rollout의 후속 policy observation live-encoder latent와
  `+1…+N`으로 정렬하고 raw latent 대신 producer stream hash, aligned/censored count,
  offset별 MSE를 trace schema 3과 semantic report schema 5에 결합한다. paired 상위 report도
  child summary를 신뢰하지 않고 trace artifact에서 이를 재집계한다. paired report schema
  4는 matched 대비 latent MSE와 terminal success rate가 모두 엄격히 낮은 조건을 별도
  mismatch로 자동 분류한다. condition의 child report SHA가 해당 trial/trace SHA를
  결합하며 정렬 pair 부재와 동률은 분류하지 않는다. validator는 index hash만 재계산하며
  prediction/observation hash는 원본 latent가 없어 형식만 검사한다.
- 남은 범위: pixel-space future visual metric은 없고 runtime latent success marker도 계속
  `false`다. 따라서 현재 분류는 compact latent proxy와 simulation outcome의 불일치일 뿐
  video quality와 실행 가능성의 완전한 비교가 아니다.
- 완료 조건: 같은 rollout의 pixel-space future visual metric을 추가하고 latent proxy와
  실제 simulation execution 판정의 차이를 재현 가능한 artifact로 비교한다.

### 7. 공식 release를 감시하고 한 번에 diff한다

- 2026-09-15 이후라고 자동으로 공개됐다고 가정하지 말고 공식 repo의 tag/release/root를 다시 확인한다.
- 공개되면 commit/tag/hash와 license를 고정하고 config, tokenizer, video/action stream, IFP, data schema, checkpoint key를 비교한다.
- 완료 조건: `compatible`, `adaptable`, `incompatible`, `unknown`을 항목별로 판정한 versioned compatibility report가 있다.

### 8. 큰 backbone은 마지막에 붙인다

[Wan2.2-TI2V-5B model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)는 documented offload command에 최소 24 GB VRAM을 제시한다. 이는 Zero-WAM checkpoint가 아니라 feasibility input이다. 위 benchmark와 interface가 준비된 뒤, 작은 clip 하나의 memory/runtime profile부터 수행한다.

- 의존성: 적합한 GPU와 model license/storage; robot은 불필요.
- 완료 조건: OOM 여부, peak VRAM, latency, resolution/frame limit과 output schema를 기록한 bounded smoke report. 성공해도 Zero-WAM fidelity로 표현하지 않는다.

## 2단계: 하드웨어 확보 후 진행할 작업

### 1. G6/G7 관측·identity·calibration부터 고정한다

- read-only doctor와 camera bench를 먼저 실행하고 좌/우 camera identity, FPS, skew, age, observation latency를 측정한다.
- 각 arm의 stable hardware/calibration ID, port, joint order, sign, unit, home repeatability, gripper endpoint를 기록한다.
- 완료 조건: 재부팅/재연결 후 같은 identity와 calibration을 로드하고, 허용 오차 내 반복성을 보이는 signed evidence가 있다.

### 2. 실측값으로 MuJoCo를 교정한다

- `torso_T_arm_base`, shoulder spacing, link/cable clearance, wrist camera extrinsic, table pose를 측정한다.
- low-speed single-joint trajectory로 delay, backlash, tracking error와 gripper mapping을 추정한다.
- 완료 조건: simulation과 real trajectory의 축별 오차가 사전 정의된 bound 안에 들고, 벗어난 축은 명시적으로 block된다.

### 3. G9 prompt와 human-video pair를 수집한다

- 현재 kinesthetic robot prompt와 별도로 동일 task의 human demonstration video를 수집한다.
- object/configuration/task-family/embodiment metadata와 촬영 provenance를 결합한다.
- 완료 조건: matched/mismatched pair audit와 camera/time integrity를 통과한 최소 task-family dataset이 있다.

### 4. G10을 shadow에서 bounded low-speed로 승격한다

- shadow replay → torque-off observation → low-speed/limited-delta actuation 순으로 진행한다.
- exact candidate, prompt, config, G6–G9 evidence와 수동 signoff를 certification에 묶고 manual abort를 유지한다.
- 완료 조건: stale/mismatched evidence가 모두 거부되고, bounded rollout이 사전 정의한 stop/abort/recovery 기준을 지킨다.

### 5. G11 held-out task success를 독립 측정한다

- train과 held-out object/configuration/task를 preregister하고 trial denominator, retry, human intervention, failure coding을 고정한다.
- 최소 3 seed/configuration에서 matched/mismatched/null prompt control을 함께 실행한다.
- raw video, observation/action log, safety events, success checker 결과를 trial별로 보존한다.
- 완료 조건: 성공률과 confidence interval, 모든 실패/개입, prompt control 효과가 재감사 가능한 report로 남는다.

SO-101 결과는 공식 Franka 결과와 독립적으로 보고한다. embodiment와 protocol이 다르므로 “46.95% 재현” 같은 직접 등치 표현은 공식 RoboTwin protocol과 asset을 실제로 맞춘 경우가 아니면 사용하지 않는다.

## 반박되거나 확인되지 않은 주장

### 선별된 25개 주장

세 독립 verifier가 각 25개 claim을 판정해 총 75개 유효 표를 만들었다. 결과는 **확인 75표, 반박 0표, 미확인 0표**였으며, 집계 결과는 **confirmed 25, refuted 0, unverified 0**이다. 여기서 “confirmed”는 주장에 적힌 좁은 범위가 직접 근거와 일치한다는 뜻이지, 공식 구현이나 결과가 이 저장소에서 재현됐다는 뜻이 아니다.

### 발견 단계에서 반박 또는 제외한 해석

- `jiaming-zhou/Zero-WAM` 검색 결과를 canonical official repository로 보는 해석은 GitHub API 404와 현재 공식 링크 때문에 제외했다. canonical repo는 [robbyant-research/Zero-WAM](https://github.com/robbyant-research/Zero-WAM)이다.
- 공식 project page의 `Data` 링크를 공개 dataset으로 보는 해석은 반박됐다. 기준일에는 GitHub `#release-plan`으로 연결될 뿐 dataset artifact가 아니다.
- Wan2.2-TI2V-5B 공개를 Zero-WAM code/model 공개로 보는 해석은 반박됐다. 전자는 upstream video backbone이고 후자의 causal/MoT/action conversion과 checkpoint는 기준일에 미공개다.

### 현재 확인할 수 없는 핵심 항목

- 로컬 state dict/config가 미래 공식 checkpoint와 호환되는지;
- HumanGen 또는 official Task-diverse VA data schema와 로컬 importer가 호환되는지;
- compact latent model이 official Zero-WAM과 같은 prompt-use behavior를 보이는지;
- 한-step compact candidate가 더 다양한 held-out object task에서 성공하는지;
- 실제 dual-SO-101에서 G6–G11과 task success가 통과하는지;
- tissue-scanning MACT의 성능 효과가 이 embodiment/task에 전이되는지.

## 한계와 열린 질문

1. Zero-WAM은 기준일 현재 2026-08 arXiv preprint이며, 공식 code/model/data 공개 예정일 이전이다. 공개 후 방법 세부와 자산 상태가 바뀔 수 있다.
2. 본 감사는 공개 논문·공식 문서·현재 로컬 코드에 대한 traceability다. paper-scale 학습, official checkpoint inference, RoboTwin reproduction, 실물 robot experiment는 수행하지 않았다.
3. 전체 test suite와 robot-free smoke는 소프트웨어 증거다. hardware collision, payload, cable snag, calibration drift, thermal/fault behavior는 측정하지 않았다.
4. 두 validation object task와 success checker는 정의됐지만 task 수·난이도·학습 budget이 너무 작아 local model의 유효성을 일반화할 수 없다. 독립 mapping과 확장 benchmark가 P1인 이유다.
5. HumanGen-compatible data를 어떤 license/provenance로 확보할지, language encoder를 어떤 공식 component에 맞출지, Wan integration을 어느 GPU budget에서 할지는 official release 확인 후 결정해야 한다.
6. 장기적으로는 논문 자체의 한계인 mobile, dynamic, unstructured, longer-horizon setting을 다뤄야 하지만, 현재 tabletop closed-loop evidence보다 앞설 이유는 없다.

## 방법론 집계

이번 조사는 사전 승인된 고정 예산을 따랐다.

| 항목 | 결과 |
| --- | ---: |
| 독립 검색 각도 | 5 |
| 검토한 discovery candidate | 30 (각도당 6) |
| 직접 검토한 canonical source | 15 |
| 중복·보조 URL variant 통합/제외 | 9 |
| source별 추출한 반증 가능 claim | 42 |
| 적대적 검증 대상으로 선별한 claim | 25 |
| 독립 verifier | 3 |
| 유효 verdict | 75/75 |
| 집계 confirmed / refuted / unverified | 25 / 0 / 0 |
| 로컬 test | 213 collected, full suite pass |

검색 각도는 공식 방법, 공개 구현·재현 자산, 주변 기술 계약, 경험적 근거·한계, 구현 가능성·의존 순서였다. 15개 canonical source는 다음 source family로 구성했다.

- Zero-WAM: [논문](https://arxiv.org/pdf/2608.26103), [공식 저장소](https://github.com/robbyant-research/Zero-WAM), [프로젝트 페이지](https://robbyant-research.github.io/Zero-WAM/)
- Wan: [Wan2.2 repo](https://github.com/Wan-Video/Wan2.2), [TI2V-5B card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), [Wan 논문](https://arxiv.org/abs/2503.20314)
- LeRobot: [SO-101](https://huggingface.co/docs/lerobot/so101), [real-robot imitation workflow](https://huggingface.co/docs/lerobot/il_robots), [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Simulation: [MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- Action chunking: [ACT](https://arxiv.org/pdf/2304.13705), [MACT](https://arxiv.org/pdf/2411.04050)
- Executability: [RoboWM-Bench](https://robowm-bench.github.io/RoboWM-Bench/), [EVA](https://eva-project-page.github.io/)

NVIDIA secondary article, Hugging Face paper index, author homepage, stale non-canonical GitHub result와 검색 snippet은 중앙 근거로 사용하지 않았다. arXiv PDF/HTML/abstract, GitHub repo/raw/API처럼 같은 canonical project의 variant는 독립 출처나 독립 확인으로 중복 계산하지 않았다. 16번째 예외 출처는 사용하지 않았다.
