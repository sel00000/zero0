# GEN-1.5 영상 근거 추적

이 문서는 사용자가 제공한 두 영상을 실제로 전사·키프레임화한 결과를
zero01 요구사항에 연결한다. 영상에서 직접 확인되는 사실과 이 저장소의 설계
추론을 섞지 않는다.

## A. Generalist 공식 GEN-1.5 영상

- URL: `https://youtu.be/1cllCVK-9lo?si=HgmyTSnIsP27-PZ0`
- 제목/채널/길이: *Introducing GEN-1.5, a one-shot learner* / Generalist / 3:33
- 게시 메타데이터: 2026-08-19
- 전사·키프레임은 로컬 분석 자료로 보관한다. 아래 링크는 원본 영상의 해당 시각을 가리킨다.

### 영상에 명시되거나 직접 보이는 것

| 시각 | 근거 | zero01에 주는 의미 |
| --- | --- | --- |
| 00:15–00:26 | one-shot learner이며 수초 내 새 작업을 context prompt로 학습한다고 설명 | 새 작업마다 재학습하는 정책이 아니라 deployment-time context 적응을 핵심 계약으로 둔다. |
| 00:26–00:30 | 여러 prompt를 조합해 long-horizon 작업을 수행한다고 설명 | 단일 prompt MVP 뒤에 prompt composition 평가를 별도 단계로 둔다. |
| 00:30–00:35 | simulator prompt를 real robot behavior로 전이한다고 설명 | fake/offline 다음 simulation gate를 두는 근거다. 영상은 SO-101 sim-to-real 성공을 증명하지 않는다. |
| 00:35–00:42 | 사람 행동을 보고 현장에서 모방하는 경우를 소개 | human-to-robot prompt는 최종 목표지만 초기 MVP는 동형 robot prompt로 시작한다. |
| 00:42–00:50 | few-shot은 1–10 gradient step, 1–5분 데이터라고 별도 설명 | zero-training physical prompt와 few-shot weight update를 같은 평가로 섞지 않는다. |
| 01:01–01:17 | 새 작업 훈련 없이 수초 시연을 context에 넣는 방식을 physical prompting이라 정의 | `PhysicalPrompt`는 영상만이 아니라 실행 action과 proprioception이 정렬된 고정 prefix다. |
| 01:27–01:43 | 현재 작업은 단순하고 성공률이 높지 않으며, 능력이 학습 recipe에서 emergent했다고 명시 | 데모 영상만으로 로봇 성공을 주장하지 않고 모든 실패를 trial log에 남긴다. |
| 01:55–02:26 | brush→banana/dustpan 대체, 양손 협업, 손 전환을 시연 | bimanual context와 양팔 action history가 필요한 이유다. |
| 02:32–02:57 | 손에 붙은 블록 제거, 가림막 제거, 한손 작업의 양손 전환을 시연 | 장애물/실패 회복은 open-loop 재생보다 rolling observation이 필요한 평가 항목이다. |
| 02:59–03:20 | human-to-robot in-context imitation을 보여주면서 아직 매우 초기라고 설명 | human prompt는 별도 고난도 gate로 유지한다. |

형태와 센서에 대한 시각 근거도 있다.

- [00:00](https://www.youtube.com/watch?v=1cllCVK-9lo&t=0s)은
  작업대 앞 고정형 좌우 양팔 구조를 보여준다.
- [00:47.8](https://www.youtube.com/watch?v=1cllCVK-9lo&t=47s)은
  중앙 기둥/가로 빔과 두 gripper 주변의 검은 센서형 모듈을 정면에서 보여준다.
- [02:01.1](https://www.youtube.com/watch?v=1cllCVK-9lo&t=121s)부터
  [02:24.1](https://www.youtube.com/watch?v=1cllCVK-9lo&t=144s)까지는
  gripper가 화면에 보이는 hand-centric 시점을 포함한다.
- [03:01.2](https://www.youtube.com/watch?v=1cllCVK-9lo&t=181s)부터
  [03:23.2](https://www.youtube.com/watch?v=1cllCVK-9lo&t=203s)까지는
  사람 손 시연 뒤 같은 테이블에서 로봇 양팔이 물체를 다루는 비교 장면이다.

검은 장치를 카메라형 손목 센서로 보는 것은 강한 시각적 추론이지만, 영상은 센서
모델명·해상도·총 카메라 수·정확한 policy feature를 말하지 않는다. 중앙 검은
블록도 head camera라고 단정할 수 없다.

## B. 한국어 해설 영상

- URL: `https://youtu.be/GgEPzq3D328?si=FRjcMWdIC5Kso2zT`
- 전사·접촉시트는 로컬 분석 자료이며 공개 저장소에는 포함하지 않는다.

| 시각 | 확인 내용 |
| --- | --- |
| 00:25:14–00:25:20 | 시연 3–12초를 context에 넣는다고 설명한다. [해당 장면](https://www.youtube.com/watch?v=GgEPzq3D328&t=1514s) |
| 00:25:31–00:25:35 | 사람 시연을 본 로봇의 즉시 모방을 설명한다. |
| 00:27:49–00:30:00 | `Prompt Frames`, `Context Editor`, `Prime Model`이 있는 Physical Prompt Engineering UI를 보여준다. [해당 장면](https://www.youtube.com/watch?v=GgEPzq3D328&t=1669s), [해당 장면](https://www.youtube.com/watch?v=GgEPzq3D328&t=1780s) |
| 00:30:04–00:30:18 | 30초 memory, sensor/language/proprioceptive prompt, 100 Hz action trajectory를 설명한다. |
| 00:43:09–00:43:57 | 양손 glove/360 camera 장치와 위치 추정 가능성을 논의한다. [해당 장면](https://www.youtube.com/watch?v=GgEPzq3D328&t=2589s), [해당 장면](https://www.youtube.com/watch?v=GgEPzq3D328&t=2637s) |

3–12초 prompt, 30초 context와 원 연구의 실행 설정은
[Generalist 공식 GEN-1.5 소개](https://generalistai.com/blog/gen-1.5)에서도 확인할 수 있다.
로컬 제어 주기는 [설정](../configs/mujoco.toml)에 따르며 원 모델의 성능과 구분한다.

## 설계로 채택한 것과 채택하지 않은 것

| 구분 | 결정 |
| --- | --- |
| 채택 | 3–12초 sensorimotor prompt, 30초 causal rolling context, no-gradient core rollout |
| 채택 | 고정형 토르소의 좌우 양팔, two-view wrist-centric RGB, 12축 proprio/action history |
| 축소 | 영상/원문 목표는 100 Hz action trajectory지만 MVP servo는 측정 전 50 Hz로 제한하고 설정 가능하게 유지 |
| 분리 | few-shot gradient update와 zero-training physical prompting을 별도 protocol로 평가 |
| 후순위 | prompt composition, sim-to-real, human-to-robot, tool improvisation, recovery benchmark |
| 선택 | head/global camera는 진단·로그·ablation만 허용하고 core score 입력에서 제외 |

정확히 두 대의 손목 RGB만으로 모든 GEN-1.5 작업이 가능하다는 주장은 두 영상에
없다. 이는 사용자의 하드웨어 제약에 맞춘 최소 관측 가설이며, head-camera 없는
held-out trial로 검증해야 한다. SO-101, arms-down home pose, 실제 모터 안전,
충돌 여유, bus rate, task 성공률도 영상이 증명하지 않는다. 로컬 형태 목표는 [하드웨어 설계](HARDWARE.md)의 고정형 양팔 구성에 따른다.

Zero-WAM의 future visual state → inverse-dynamics action 분해는 영상에서 보이는
구조가 아니라 [Zero-WAM 논문](https://arxiv.org/html/2608.26103)의 논문
가설이다. 코드 추적점은 `contracts.py` → `context.py` → `tensorizer.py` →
`model.py` → `policy.py` → `runtime.py` 순서다.
