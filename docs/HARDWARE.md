# zero01 hardware

## 운용 기준

zero01의 확인된 MVP 하드웨어는 고정형 휴머노이드 상체와 좌우 대칭
SO-101 follower arm 두 개다. 기본 홈 포즈는 양팔을 아래로 내린 중립 자세로
정의한다. 이 자세는 전원 인가, 캘리브레이션 확인, 프롬프트 기록, 롤아웃 시작
전에 매번 재현되어야 한다.

핵심 정책 입력은 정확히 두 개의 RGB 카메라다.

| 입력 | 상태 | 정책 계약 |
| --- | --- | --- |
| `left_wrist` RGB | 필수 | 왼쪽 손목 시점 |
| `right_wrist` RGB | 필수 | 오른쪽 손목 시점 |
| `head_optional` 또는 head depth/RGB | 선택 | 진단, 로깅, ablation 전용. 핵심 점수의 필수 입력 금지 |

`docs/ARCHITECTURE.md`와 코드 계약은 손목 두 시점만 핵심 입력으로 노출한다.
`src/so101_wam/constants.py`의 `PRIMARY_CAMERA_KEYS`는
`("left_wrist", "right_wrist")`이며, `RuntimeConfig`는 다른 필수 카메라 구성을
거부한다.

## 기계 구성

| 항목 | 확정값 | 메모 |
| --- | --- | --- |
| 베이스 | 고정형 상체/토르소 | 모바일 베이스는 MVP 필수 조건이 아니다. |
| 팔 | mirrored dual SO-101 follower arms | 좌우 모두 6축: shoulder pan/lift, elbow flex, wrist flex/roll, gripper |
| 홈 포즈 | arms-down neutral | 실측 엔코더 값으로 저장해야 하며 문서에서 각도를 추정하지 않는다. |
| 손목 카메라 | RGB 2대 | 좌/우 모두 같은 해상도, 30 FPS 목표 |
| 헤드 카메라 | 선택 | 진단 전용. 정책 필수 feature로 승격하지 않는다. |

치수, 기준 좌표계, 카메라 extrinsic, 토르소-팔 transform은 아직 문서값으로
확정하지 않는다. 다음 필드는 실측 캘리브레이션 산출물로만 채운다.

| 필드 | 단위 | 채우는 시점 |
| --- | ---: | --- |
| `torso_T_left_arm_base` | m/rad 또는 SE(3) | 좌측 베이스 체결 후 측정 |
| `torso_T_right_arm_base` | m/rad 또는 SE(3) | 우측 베이스 체결 후 측정 |
| `left_wrist_T_camera` | m/rad 또는 SE(3) | 좌측 손목 카메라 고정 후 측정 |
| `right_wrist_T_camera` | m/rad 또는 SE(3) | 우측 손목 카메라 고정 후 측정 |
| `home_joint_position` | LeRobot `.pos` 12개 | arms-down neutral 자세에서 측정 |
| `home_joint_tolerance` | 양수 LeRobot `.pos` 12개 | 홈 자세 반복 측정 오차와 고정 지그 오차로 결정 |
| `joint_lower` / `joint_upper` | LeRobot `.pos` 12개 | 충돌 여유를 포함한 작업공간 측정 |
| `max_delta_per_servo_tick` | LeRobot `.pos`/tick | 부하 상태에서 안정 속도 측정 |

`configs/fake.toml`의 범위는 fake-device 정규화 값이다. 실제 하드웨어
캘리브레이션이나 충돌 한계로 복사하면 안 된다.

실제 구성의 시작점은 `configs/lerobot_hardware.template.toml`이다. 템플릿은
`backend="lerobot"`, `actuation_enabled=false`, `calibrated=false`이며 포트와
두 식별자가 비어 있다. 출력 활성화 전 모두 실측값으로 채워야 한다.

| 필드 | 의미 |
| --- | --- |
| `left_port`, `right_port` | 서로 다른 실측 모터 버스 포트 |
| `left_wrist_camera`, `right_wrist_camera` | 서로 다른 `/dev/video*` 경로 또는 OpenCV integer index |
| `camera_width`, `camera_height`, `camera_fps` | 양쪽 공통 capture 형식; `camera_fps`는 runtime `camera_hz`와 같아야 함 |
| `calibration_dir` | LeRobot `BiSOFollowerConfig.calibration_dir`에 전달할 calibration directory |
| `hardware_id` | 고정 토르소/양팔 조립체 식별자 |
| `calibration_id` | 12축 범위·방향·홈 포즈 캘리브레이션 식별자 |
| `home_joint_position` | 충돌 없는 arms-down neutral에서 실측한 `.pos` 12개 |
| `home_joint_tolerance` | 같은 홈 자세를 반복 측정해 정한 양수 허용 오차 12개 |
| `safety.calibrated` | placeholder 36개가 실측값으로 교체됐음을 표시하는 gate |

식별자는 감사 추적용이며 물리 시험 자체를 대신하지 않는다. runtime은
`fake`/`mujoco`/`lerobot` backend와 adapter가 다르면 rollout 생성 단계에서 거부한다.

## LeRobot 0.6.1 경로

이 저장소는 LeRobot `0.6.1`의 bimanual SO 경로를 기준으로 계약을 맞춘다.
공식 LeRobot v0.6.1 소스는 `bi_so_follower`와 `bi_so_leader`를 등록하고,
각 단일 SO arm feature에 `left_` / `right_` prefix를 붙여 bimanual feature를
만든다.

공식 절차는 다음 순서다. 실제 명령은 포트, 카메라 index, 캘리브레이션 id를
하드웨어별로 바꿔 실행해야 하므로 이 문서에서는 성공 실행을 주장하지 않는다.

1. `lerobot==0.6.1` 및 hardware extra 설치
2. `lerobot-setup-motors`로 leader/follower 모터 id 설정
3. `lerobot-calibrate`로 `bi_so_leader`와 `bi_so_follower` 각각 캘리브레이션
4. `lerobot-teleoperate`로 torque, 방향, 좌우 매핑, 손목 카메라 확인
5. `so101_wam.prompt_recorder`로 3-12초 observation-only kinesthetic prompt 기록

zero01 episode가 30 FPS이면 `physical_prompt_from_episode()`가 시작/끝과
executed action을 보존하면서 10 Hz 정책 prompt로 nearest-timestamp downsample한다.
recorder의 `action`은 motor bus로 전송된 target이 아니라 각 frame에서 측정한
`Present_Position`의 키네스테틱 target proxy다. manifest는 이를
`action_source="measured_present_position_no_goal_write"`로 표시하며
`goal_position_commands_sent=0`을 함께 기록한다.
실제 연결·shadow·출력 절차는 [BRINGUP.md](BRINGUP.md)에 고정한다.

검증된 feature 이름은 다음 순서를 따른다.

| zero01 축 순서 | LeRobot feature |
| ---: | --- |
| 0 | `left_shoulder_pan.pos` |
| 1 | `left_shoulder_lift.pos` |
| 2 | `left_elbow_flex.pos` |
| 3 | `left_wrist_flex.pos` |
| 4 | `left_wrist_roll.pos` |
| 5 | `left_gripper.pos` |
| 6 | `right_shoulder_pan.pos` |
| 7 | `right_shoulder_lift.pos` |
| 8 | `right_elbow_flex.pos` |
| 9 | `right_wrist_flex.pos` |
| 10 | `right_wrist_roll.pos` |
| 11 | `right_gripper.pos` |

카메라를 각 arm config에서 `wrist`로 선언하면 LeRobot `BiSOFollower` 관측에서
`left_wrist`, `right_wrist`로 prefix가 붙는다. zero01의 핵심 dataset feature는
`observation.images.left_wrist`, `observation.images.right_wrist`,
`observation.state`, `action`이다.

## XLeRobot 참고 자산

XLeRobot은 MVP의 필수 구현이 아니라 기계/전장 참고 자산이다. TheRobotStudio
SO-ARM100 README는 XLeRobot을 2x SO101 arm, 2x wrist RGB camera, 1x head depth
camera와 2-DOF neck을 갖는 dual-arm mobile robot add-on으로 설명하고, BOM,
3D printing model, assembly guide, simulation, teleop guide를 연결한다.

zero01 MVP에는 다음만 가져온다.

- SO101 arm BOM, CAD/STL/STEP, wiring, wrist camera mount 참고
- dual-arm routing과 cable strain relief 아이디어
- simulation/teleop 절차 참고

다음은 MVP 요구사항으로 간주하지 않는다.

- Lekiwi/mobile base
- 300 Wh battery
- head depth camera 또는 2-DOF neck
- XLeRobot 전체 형상 치수

## 물리 안전 체크

실제 모터 enable 전 통과 조건:

- `halt()`가 terminal 상태로 전환되고 내부 fault latch가 명시적 reset 전까지 유지된다.
- 좌/우 USB, 전원, 카메라 케이블에 full-range 동작 여유와 strain relief가 있다.
- 손목 카메라 케이블이 wrist roll에 감기지 않는다.
- 좌우 팔 사이, 팔-토르소, 팔-테이블, 팔-카메라 mount 충돌 금지 부피가 측정되어 있다.
- arms-down neutral 홈에서 양팔이 서로 닿지 않고, gripper가 테이블/토르소에 닿지 않는다.
- 모든 action chunk는 안전 supervisor를 통과한 뒤에만 bus로 전송된다.
- multi-row chunk는 bus로 직접 전송되지 않고 50 Hz에서 한 행씩 검증된다.
- 원래 policy chunk timestamp를 유지해 100 ms watchdog을 넘긴 나머지 행은 폐기한다.
- 실제 출력은 LeRobot `is_calibrated=true`이고 최초 실측 12축이 home tolerance 안일 때만 시작한다.
- 실제 출력 adapter는 LeRobot `OpenCVCamera.frame_lock` 안에서 각 wrist의
  `latest_frame`과 `latest_timestamp`를 한 쌍으로 복사한다. 둘 중 하나라도 없거나
  frame age가 100 ms를 넘거나 손목 카메라 skew가 17 ms를 넘으면 command를
  전송하지 않고 managed path가 fault latch 후 연결을 해제한다. supervisor의 hold
  target은 진단용이다.

정책 없이 camera timing과 arms-down home 반복성을 먼저 측정하는 명령은
[BRINGUP.md](BRINGUP.md)의 `so101_wam.hardware_preflight`다. 자동 계측 통과는
좌우 영상 identity, 실제 축 방향, 케이블 간섭 확인을 대신하지 않는다.

현재 소프트웨어에는 [nominal MuJoCo collision gate](MUJOCO.md)가 있다. 공식
Menagerie SO-101 link geometry에서 팔-팔, 팔-토르소, 팔-테이블, 팔-바닥과 self
collision을 target/transition 단계에서 차단한다. 그러나 local torso/어깨/table
transform은 설계값이므로 실제 모터 enable 전에 실측 transform으로 scene을
갱신하고 real slow-jog로 별도 검증해야 한다.

## 근거

- Local: `docs/ARCHITECTURE.md`
- Local: `docs/VIDEO_REFERENCE.md`, `configs/lerobot_hardware.template.toml`
- Local: `src/so101_wam/constants.py`, `src/so101_wam/config.py`,
  `src/so101_wam/contracts.py`, `src/so101_wam/dataset.py`
- Paper: [Zero-WAM](https://arxiv.org/abs/2608.26103) p.9, test-time prompt cache와 next-video/action chunk
  decoding; p.11, RoboTwin/real-world 평가는 원 논문 결과이며 zero01 결과가 아니다.
- Paper: [MACT](https://arxiv.org/abs/2411.04050) p.3-p.5,
  history image와 near-future action chunking 동기.
- Official: [LeRobot repository](https://github.com/huggingface/lerobot)
- Official source: [LeRobot v0.6.1 `BiSOFollower`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/bi_so_follower/bi_so_follower.py)
- Official source: [LeRobot v0.6.1 `BiSOLeader`](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/teleoperators/bi_so_leader/bi_so_leader.py)
- Official: [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Official: [XLeRobot documentation](https://xlerobot.readthedocs.io/en/latest/)
