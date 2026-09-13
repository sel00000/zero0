# 7개 요청 작업 검증 프로토콜

요청한 7개 작업은 현재 **미실행 / 검증 환경 미구비**다. 기본 장면에는 블록과
원기둥만 있고, 해당 작업의 시연 데이터와 성공 판정기가 없다.
[현재 환경·체크포인트 조사](verification/task-readiness.json)에 확인한 내용을 기록했다.
미실행 항목을 모델의 실패나 성공률 0%로 계산하지 않는다.

zero01은 GEN-1.5 공식 weight나 공식 평가 환경이 아니다. GEN-1.5의 one-shot
physical prompting, 3-12초 prompt, 약 30초 memory, 100 Hz action trajectory 설명은
공식 소개 글과 영상의 연구 배경으로만 사용한다.[^gen-blog][^gen-video] 아래 기준은
GEN-1.5가 공개한 성공률이나 benchmark가 아니라 이 저장소의 시험 설계 초안이다.
환경과 물체 크기를 정한 뒤 기준을 확정하고, 실행 결과를 본 뒤 기준을 완화하지 않는다.

## 검증 전에 고정할 것

모든 작업은 실행 전에 같은 항목을 고정한다.

| 항목 | 필수 조건 |
| --- | --- |
| Scene fidelity | 물, 의류, 사람 손, 컵 collision을 어떤 수준으로 모델링했는지 명시한다. MuJoCo 기본 fluid force는 주변 매질 힘 모델이며 자유표면 물 붓기 검증으로 쓰지 않는다.[^mujoco-fluid] 의류는 `flexcomp` 또는 mesh 기반 proxy로만 시작한다.[^mujoco-flex] |
| Correct prompt | 작업별 3-12초 matched prompt를 별도 파일 hash로 고정한다. prompt는 평가 seed와 다른 episode에서 온다. |
| Compatible checkpoint | checkpoint hash, training report hash, normalization, action width, camera contract, MuJoCo MJB hash를 trial 전에 결속한다. 공식 GEN-1.5 weight로 주장하지 않는다. |
| Frozen vs adaptation | 기본 평가는 frozen checkpoint + in-context prompt만 허용한다. gradient update를 쓰면 few-shot adaptation 결과로 따로 기록한다.[^gen-blog] |
| Seeds / held-out | 정량 주장은 작업당 최소 20개 held-out seed를 사용한다. 3개 seed smoke는 실행 경로 확인만 허용한다. |
| Controls | 같은 목표·초기 상태에서 `matched`, `wrong_task`, `empty_prompt`, `shuffled_prompt`를 실행한다. `wrong_task`에서는 다른 작업의 시연만 넣고 평가 목표를 유지한다. |
| Completion | 작업별 success predicate, failure reason, timeout, terminal state를 trial artifact에 남긴다. 실패도 분모에 포함한다. |
| Stability | 성공 판정 뒤 최소 2-3초 동안 object/cloth/contact 상태가 기준 안에 남아야 한다. |
| Safety | joint limit, forbidden collision, stale action, excessive contact force, dropped object를 success보다 먼저 fail로 판정한다. |

방향성 통과 기준은 로컬 기준으로 고정한다. `matched` 성공률이 60% 이상이고,
`wrong_task`와 `empty_prompt`가 각각 20% 이하이며, `matched - max(control)` 차이가
40 percentage point 이상을 목표 기준으로 둔다. 기준을 수정하면 기존 실행 기록을
보존하고 새 protocol version으로 다시 평가한다. 이 수치 자체가 기술의 타당성을
보장하지는 않으며, 신뢰구간과 대조군 결과를 함께 읽는다.

## 작업별 성공 기준

| 작업 | Fidelity 요구 | 로컬 success predicate | 현재 상태 |
| --- | --- | --- | --- |
| 물 따르기 | 실제 물이면 자유표면 유체가 필요하다. MuJoCo 기본 fluid force만으로는 물 붓기라고 부르지 않는다.[^mujoco-fluid] 초기 검증은 bead/bolt 같은 작은 rigid particle proxy로만 허용한다. | proxy 기준: 시작 container의 particle 중 80% 이상이 목표 컵 내부, workspace 밖 spill 10% 이하, 컵 전도 없음, terminal 2초 안정. 실제 물 기준은 hardware 또는 별도 유체 simulator 없이는 미정. | 미실행 / 검증 환경 미구비 |
| 티셔츠 접기 | deformable shirt mesh 또는 shirt-shaped `flexcomp` proxy가 필요하다.[^mujoco-flex] 단순 사각 천이면 “셔츠 접기”가 아니라 cloth-fold proxy로 표시한다. | collar, hem, 양쪽 sleeve keypoint가 목표 위치 4 cm 이내, fold-line 각도 15도 이내, self-intersection fail 없음, terminal 3초 안정. | 미실행 / 검증 환경 미구비 |
| 바지 접기 | pants-shaped cloth mesh와 허리/밑단/무릎 keypoint가 필요하다. 사각 천 두 장이면 pants-fold proxy로 표시한다. | waist와 양쪽 cuff keypoint가 목표 위치 5 cm 이내, 좌우 leg overlap IoU 0.65 이상, fold-line 각도 15도 이내, terminal 3초 안정. | 미실행 / 검증 환경 미구비 |
| 물건 정리 | rigid object, bin/zone, object identity가 scene에 선언되어야 한다. GEN-1.5 공식 자료의 물체 정리·도구 사용 장면은 연구 배경일 뿐 zero01 성능 근거가 아니다.[^gen-video] | 6개 중 5개 이상이 지정 zone 또는 container 안에 있고, 각 object 중심이 zone 경계 안쪽 2 cm 이상, 떨어뜨림 없음, terminal 2초 안정. | 미실행 / 검증 환경 미구비 |
| 책상 정리 | 실제 책상 정리가 아니라 tabletop rigid-object tidying abstraction으로 시작한다. 서랍, 케이블, 종이 더미는 별도 fidelity 단계다. | 지정 clear area의 object 점유율 10% 이하, 정리 대상의 80% 이상이 tray/bin/zone 안, 금지 영역 침범 없음, terminal 2초 안정. | 미실행 / 검증 환경 미구비 |
| 악수 | 사람 손과 안전을 검증하려면 human model 또는 실제 하드웨어 안전 protocol이 필요하다. MuJoCo dummy hand는 contact-gesture proxy일 뿐이다. | dummy hand target site와 접촉 1.5초 이상, 접촉력 2-15 N, 상대 속도 0.15 m/s 이하, 손/팔 금지 collision 없음. 실제 사람 악수 성공으로 표기하지 않는다. | 미실행 / 검증 환경 미구비 |
| 색상 컵 3개 쌓기 | 컵은 단순 cylinder가 아니라 내부 공간과 rim collision이 있는 rigid geometry여야 한다. 단순 frustum이면 cup-stack proxy로 표시한다. | 아래부터 지정 색상 순서 일치, 각 컵 중심 수평 offset 2 cm 이하, upright angle 10도 이하, 낙하/전도 없음, terminal 3초 안정. | 미실행 / 검증 환경 미구비 |

## 실행 절차

1. 작업별 scene, prompt, checkpoint, config, metric 파일을 freeze한다.
2. 각 task에 대해 held-out 20 seed 이상을 만든다. 개발 smoke 3 seed는 공개 성공률로 쓰지 않는다.
3. 같은 seed schedule로 `matched`, `wrong_task`, `empty_prompt`, `shuffled_prompt`를 실행한다.
4. 각 trial은 success/failure, safety reason, terminal metric, artifact hash를 남긴다.
5. 성공률은 Wilson 95% interval과 함께 보고한다. control 대비 차이가 없으면 prompt-following 근거로 쓰지 않는다.
6. frozen run과 gradient-adapted run을 같은 표에 섞지 않는다. gradient update가 있으면 “few-shot adaptation”으로 별도 표기한다.
7. 영상/GIF는 해당 trial artifact hash와 연결한다. scripted 또는 reference-nudge 영상이면 learned policy success로 쓰지 않는다.

## 현재 실행 가능한 진단

[체크포인트 검증 영상과 기록](DEMO.md)은 학습에 사용한 기본 블록 이동을
정해진 기준 동작·정지 대조군·학습 정책으로 비교한다. 이 진단은 위 7개 작업을
대신하지 않으며, 새로운 작업에 대한 일반화 검증도 아니다.

## 인용

[^gen-blog]: Generalist, “GEN-1.5: A one-shot learner for robots”, 2026-08-19. https://generalistai.com/blog/gen-1.5
[^gen-video]: Generalist 공식 영상, “Introducing GEN-1.5, a one-shot learner”, 2026-08-19. https://www.youtube.com/watch?v=1cllCVK-9lo
[^mujoco-fluid]: MuJoCo documentation, Fluid interaction model. 확인일: 2026-09-13. https://mujoco.readthedocs.io/en/latest/computation/fluid.html
[^mujoco-flex]: MuJoCo XML reference, `body/flexcomp`. 확인일: 2026-09-13. https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-flexcomp
