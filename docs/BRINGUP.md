# zero01 real bring-up

## 현재 도달점

실제 LeRobot `BiSOFollower`를 구성하고 연결·해제하며, 저장된 CompactWAM
checkpoint와 3–12초 episode prompt를 이용해 10 Hz policy / 50 Hz servo loop를
실행하는 코드 경로가 있다. 현재 개발 환경에는 `lerobot`과 실제 SO-101이 없고,
학습된 checkpoint도 포함되어 있지 않으므로 이 문서는 실제 작업 성공을 주장하지
않는다.

## 준비물

- Python 3.12 환경의 공식 `lerobot==0.6.1` hardware/Feetech 구성
- `mujoco>=3.12.0,<4`와 `configs/mujoco.toml`에 지정한 동작 가능한 GL backend
  (G10 실제 출력 직전 exact G8 재실행에 사용)
- 좌우 SO-101 follower port
- 서로 다른 좌우 wrist-camera device path 또는 OpenCV index
- LeRobot calibration directory
- 실측한 12축 joint limits, per-servo-tick delta, arms-down home position/tolerance
- zero01 schema의 3–12초 `.npz` episode와 `.json` manifest
- 동일 `action_horizon`으로 학습된 versioned CompactWAM checkpoint

공식 v0.6.1은 `Python >=3.12`, `torch>=2.7,<2.12.0`이며 SO-101 CLI에는
`core_scripts`와 `feetech` extra가 필요하다. 기존 Python 환경을 바꾸지 않도록 별도
가상환경에서 이 저장소의 pinned extra를 설치한다.

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[hardware,sim]'
```

## 접속 전 read-only doctor

machine-local config를 작성한 뒤 장치에 접속하기 전에 다음을 실행한다.

```shell
so101-wam-doctor --config configs/lerobot_hardware.local.toml
```

doctor는 filesystem과 설치 metadata만 읽는다. `lerobot-find-port`,
`lerobot-find-cameras`, calibration, motor setup을 대신 실행하지 않고 `/dev/tty*`나
`/dev/video*`도 열지 않는다. 다음을 JSON으로 보고하며 blocker가 있으면 exit code
2를 반환한다.

- Python, `lerobot==0.6.1`, v0.6.1-compatible torch와 공식 CLI 설치 여부
- `/dev/serial/by-id` 우선의 서로 다른 serial 후보 두 개
- `/dev/v4l/by-id` 우선의 서로 다른 V4L2 camera 후보 두 개
- local config의 device path 권한, calibration JSON, hardware/calibration identity,
  실측 safety/home profile
- WSL2 여부와 아직 전달되지 않은 USB/video device

doctor는 좌/우를 추측하지 않는다. 공식 `lerobot-find-port`는 각 USB를 실제로
뺐다 꽂는 절차로 확인하고, `lerobot-find-cameras opencv` 결과는 재부팅/재연결 뒤
바뀔 수 있으므로 매 세션 검증한다. `lerobot-setup-motors`는 EEPROM의 motor ID와
baudrate를 쓰며 한 번에 정확히 한 motor를 연결하는 물리 절차이므로 자동화하지
않는다.

LeRobot 0.6.1의 `lerobot-find-port`는 `--help`도 도움말로 처리하지 않고 즉시
대화형 분리/재연결 탐색을 시작한다. 자동 진단에서는 이 명령을 호출하지 말고 위
read-only doctor를 사용한다.

WSL2에서는 USB 장치가 기본 전달되지 않는다. Windows 관리자 PowerShell에서
doctor가 `wsl_usbipd_not_available`을 보고하면 먼저 Windows에 `usbipd-win`을
설치하고 WSL 셸을 다시 연다. `usbipd list`, 최초 1회
`usbipd bind --busid <BUSID>`, 이후
`usbipd attach --wsl --busid <BUSID>`를 실행하고 WSL의 `lsusb`와 `/dev/ttyACM*`로
확인한다. webcam은 attach 성공만으로 `/dev/video*`가 생긴다고 가정하지 않는다.
두 V4L2 node와 `lerobot-find-cameras opencv`가 실제로 확인되지 않으면 native Linux
host를 사용하거나 WSL camera 전달 문제를 먼저 해결한다.

doctor는 현재 WSL 세션의 PATH가 아직 갱신되지 않았더라도 표준 설치 위치인
`C:\Program Files\usbipd-win\usbipd.exe`를 읽기 전용으로 확인한다.

근거는 tagged 공식 문서인
[LeRobot v0.6.1 installation](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/docs/source/installation.mdx),
[SO-101 bring-up](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/docs/source/so101.mdx),
[camera discovery](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/docs/source/cameras.mdx)와
[Microsoft WSL USB 연결](https://learn.microsoft.com/windows/wsl/connect-usb)이다.

공식 v0.6.1 factory는 각 `SOFollowerConfig`에
`cameras={"wrist": OpenCVCameraConfig(...)}`를 넣는다. `BiSOFollower`가 이를
`left_wrist`와 `right_wrist`로 prefix한다. 두 카메라를 top-level bimanual camera
mapping에 넣지 않는다.

## 구성

`configs/lerobot_hardware.template.toml`을 machine-local 파일로 복사한 다음 다음
필드를 실제 값으로 바꾼다.

```toml
[runtime]
backend = "lerobot"
actuation_enabled = false

[safety]
calibrated = false
# joint_lower, joint_upper, max_delta_per_servo_tick를 실측값으로 교체

[lerobot]
left_port = "/dev/ttyACM0"
right_port = "/dev/ttyACM1"
left_wrist_camera = "/dev/video0"
right_wrist_camera = "/dev/video2"
calibration_dir = "/absolute/path/to/lerobot/calibration"
hardware_id = "fixed-torso-a"
calibration_id = "cal-2026-08-31-a"
home_joint_position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 반드시 실측값으로 교체
home_joint_tolerance = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 실측 반복 오차로 교체
```

위의 0/1 값은 형식 예시일 뿐 실제 홈 자세나 허용 오차가 아니다.

## 관측-only G6/G7 preflight

정책이나 checkpoint를 연결하기 전에 다음 명령으로 실제 bus, 두 wrist camera,
12축 현재 자세만 계측한다.

```shell
PYTHONPATH=src python -m so101_wam.hardware_preflight \
  --config configs/lerobot_hardware.local.toml \
  --seconds 10 \
  --report reports/g6_g7_preflight.json
```

이 경로는 `BiSOFollower.connect()`, `SOFollower.configure()`, `send_action()`을
호출하지 않으며 `Goal_Position`을 쓰지 않는다. 각 motor bus를 handshake하고 기존
LeRobot calibration이 유효한지 확인한 뒤 wrist camera를 연결해 관측만 읽는다.
종료 시 각 arm의 `disable_torque_on_disconnect` 설정을 그대로 적용하므로 토크가
풀려도 간섭하지 않는 arms-down 자세에서 실행해야 한다. 새 calibration이 필요하면
이 명령이 대신 수행하지 않으며 공식 `lerobot-calibrate`를 먼저 사용한다.

보고서는 다음 자동 판정을 포함한다.

- 요청 해상도와 정확히 두 camera key
- 설정 FPS의 95% 이상인 sampling 및 camera timestamp 기반 FPS
- camera buffer age p95와 observation assembly latency p95가
  `max_observation_age_s` 이하
- 좌우 camera timestamp skew p95가 `max_camera_skew_s` 이하
- 설정된 arms-down home/tolerance에 대한 전 sample 최대 오차
- median home candidate와 축별 최대 반복 오차

LeRobot `0.6.1`의 공개 observation은 camera timestamp를 반환하지 않는다. 따라서
preflight와 실제-output adapter는 version-pinned
`OpenCVCamera.frame_lock` 안에서 `latest_frame`과 `latest_timestamp`를 한 쌍으로
복사한다. 반환 영상을 읽은 뒤 시각을 별도로 조회하지 않으므로 background camera
thread가 다음 frame을 갱신하는 경쟁을 피한다. 시각이 없거나 감소하거나, frame age가
`max_observation_age_s`를 넘으면 fail closed한다.
모든 자동 항목이 통과해도 좌우 영상 identity, 실제 축 방향, 케이블/충돌 여유는
사람이 확인해야 하므로 report의 최종 `result`는 `partial`이다. 자동 항목 실패는
`fail`이며 누락 원인이 `failures`에 기록된다. 기존 report는 보호되며 같은 경로를
다시 쓸 때만 `--overwrite`를 명시한다. 실패 보고서도 보존·출력하지만 CLI는 exit
code 2를 반환하므로 자동화가 이를 통과로 오인하면 안 된다.

## G9 키네스테틱 prompt 기록

리더 팔로 팔로워를 구동하며 실제 전송 명령을 수집하려면
[리더 팔 시연 기록](TELEOP_RECORDING.md)의 `so101-wam-record-demo`를 사용한다.

G6/G7 자동 계측과 좌우 wrist 영상 identity를 확인한 뒤, 위치 명령 없이 수동
시연을 기록한다.

```shell
PYTHONPATH=src python -m so101_wam.prompt_recorder \
  --config configs/lerobot_hardware.local.toml \
  --output-dir data/prompts \
  --task "place block" \
  --duration 3.0 \
  --episode-index 1
```

동일 명령은 설치 후 `so101-wam-record-prompt` console script로도 실행할 수 있다.
출력은 `data/prompts/prompt_000001.npz`와 같은 stem의 JSON manifest다. 기존
artifact는 불변으로 보호되므로 다시 기록할 때는 새 `--episode-index`를 쓴다.

recorder는 `BiSOFollower.connect()`, `SOFollower.configure()`, `send_action()`과
`Goal_Position` write를 호출하지 않는다. 각 bus/camera를 직접 연결하고 기존
LeRobot calibration을 확인한다. 두 bus가 모두 연결된 뒤 camera를 열거나 토크를
풀기 전에 12축 `Present_Position`을 읽고, `safety.calibrated=true`인 실측 limit와
`home_joint_position`/`home_joint_tolerance`를 모두 통과해야 양쪽
`disable_torque()`를 호출한다. 한쪽 torque-off capability가 없거나 자세가 범위를
벗어나면 양쪽 모두 torque-disable 0회로 cleanup한다. 그래도 토크가 풀렸을 때
중력으로 링크가 떨어지지 않도록 기구 배치와 수동 지지를 먼저 확보해야 한다.

사람이 양팔을 움직이는 동안 각 frame의 12축 `Present_Position`을 dataset의
`action`에 복사한다. 이는 키네스테틱 target proxy이며 실제 전송 action이 아니다.
manifest는 다음 의미 필드를 고정한다.

- `capture_mode="observation_only_kinesthetic"`
- `action_source="measured_present_position_no_goal_write"`
- `goal_position_commands_sent=0`
- `torque_disabled_for_capture=true`

30 Hz episode timestamp는 fixed capture grid를 사용한다. 별도로 실제 monotonic
capture-start offset을 모두 manifest에 남기고 frame 간 오차가 expected dt의 5%를
넘으면 저장하지 않는다. 두 wrist frame/timestamp는 같은 camera lock에서 읽으며,
각 camera timestamp 간격도 expected dt의 5%를 벗어나거나 timestamp가 증가하지
않으면 거부한다. 설정 해상도/FPS, 100 ms age/assembly latency,
17 ms skew 기준을 어기면 즉시 cleanup하고 최종 `.npz`/`.json`을 발행하지 않는다.
완성된 pair는 checksum load와 10 Hz physical-prompt fingerprint replay까지 성공한
뒤에만 결과 JSON을 출력한다.

기본 640×480 RGB 두 시점의 12초 raw pixel payload만 약 635 MiB이며 현재
`EpisodeBuffer`의 stack 단계에서는 순간 메모리가 더 필요하다. 첫 bring-up은 3초로
시작하고, 12초 기록 전 host memory headroom을 확인한다. 실제 G9 통과 주장은 이
명령으로 생성한 artifact와 시연 영상/작업 기록을 함께 보존한 뒤에만 가능하다.

## Shadow bring-up

`so101_wam.train_cli`에서 나온 offline candidate도 shadow에는 사용할 수 있다.
그 artifact는 `offline_trained=true`, `trained=false`이므로 아래 명령에 실제 출력
플래그를 추가하면 checkpoint gate에서 거부된다. 데이터 및 학습 계약은
[TRAINING.md](TRAINING.md)에 있다.

다음 명령은 robot/camera를 연결하고 정책을 계산하지만 action을 전송하지 않는다.
구성 파일에 `actuation_enabled=true`가 있더라도 CLI 출력 플래그가 없으면 shadow로
강제된다.

```shell
PYTHONPATH=src python -m so101_wam.hardware_cli \
  --config configs/lerobot_hardware.local.toml \
  --checkpoint checkpoints/compact_wam.pt \
  --prompt data/prompts/prompt_000001.npz \
  --manifest data/prompts/prompt_000001.json \
  --device cuda:0 \
  --steps 20
```

실행 순서는 checkpoint strict load → episode checksum 검증 → 30→10 Hz prompt
변환 → LeRobot `0.6.1` 확인 → robot connect → measured history prime → rollout →
pending horizon 폐기 → disconnect다. 예외가 발생해도 disconnect를 시도하고 runtime은
terminal fault 상태로 들어간다.

## 실제 출력 gate

실제 출력에는 다음 조건이 동시에 필요하다.

1. config의 `actuation_enabled=true`와 `safety.calibrated=true`
2. 측정된 safety arrays, arms-down home/tolerance, ports/cameras/calibration fields
3. 연결 직후 LeRobot의 `is_calibrated=true`
4. 두 wrist frame/timestamp가 원자적으로 결합되고 age/skew gate를 통과함
5. policy 출력 전에 모든 12축 실측 자세가 home tolerance 안에 있음
6. checkpoint가 offline/candidate 표식을 포함하지 않고
   `artifact_kind="compact_wam_deployment"`이며 metadata의 `trained=true`,
   `deployment_ready=true`, non-empty `checkpoint_id`, 64자 lowercase
   `training_evidence_sha256`와 별도의 `deployment_evidence_sha256`를 포함함
7. 외부 deployment certification JSON이 현재 checkpoint bytes SHA-256, 전체
   validated config SHA-256, `hardware_id`, `calibration_id`, G6-G9 pass 및 각 evidence
   digest, 허용 policy-step 수를 정확히 결합하고, runtime이 동일 원본 증거를 다시
   검증해 같은 digest와 정확한 G8 MuJoCo 재실행 결과를 얻음
8. CLI의 `--enable-real-output`
9. `--ack-hardware-id`가 config의 `hardware_id`와 정확히 일치

6번과 7번 artifact는 수동으로 metadata를 바꿔 만드는 것이 아니다.
`so101_wam.certify_cli`가 offline candidate, training report, preflight, exact-candidate
MuJoCo report, real prompt pair, 수동 hardware signoff를 모두 검증한 뒤 exact weights로
새 pair를 발행한다. signoff schema, hash 계산, 발급 명령은
[CERTIFICATION.md](CERTIFICATION.md)에 있다.

```shell
PYTHONPATH=src python -m so101_wam.hardware_cli \
  --config configs/lerobot_hardware.local.toml \
  --checkpoint checkpoints/compact_wam_deployment_001.pt \
  --prompt data/prompts/prompt_000001.npz \
  --manifest data/prompts/prompt_000001.json \
  --deployment-certification reports/compact_wam_deployment_001.certification.json \
  --source-candidate checkpoints/compact_wam_candidate_001.pt \
  --source-training-report reports/compact_wam_candidate_001.training.json \
  --source-preflight-report reports/g6_g7_preflight.json \
  --source-mujoco-config configs/mujoco.toml \
  --source-mujoco-report reports/g8_candidate_001.json \
  --source-manual-signoff reports/manual_signoff_001.json \
  --device cuda:0 \
  --steps 20 \
  --enable-real-output \
  --ack-hardware-id fixed-torso-a
```

`trained=true`와 `deployment_ready=true`를 수동으로 적는 것은 학습·검증 증거가
아니다. real-output gate는 offline/candidate 표식 자체를 먼저 거부하며 별도의
deployment artifact/evidence 계약을 요구한다. 후보 weights를 deployment metadata로
다시 저장해도 외부 certification 없이는 실패한다. certification은
`artifact_kind="so101_wam.deployment_certification"`, `evidence_level="real"`,
`result="pass"`, `authorization_scope="G10_dry_rollout"`과 G6-G9 evidence를 포함하며
checkpoint/config/hardware/calibration identity가 하나라도 다르면 실패한다.
실제 출력 경로는 인증서의 요약 boolean을 단독으로 신뢰하지 않는다. checkpoint와
현재 prompt를 전후 해시하고, 지정한 candidate/training/preflight/MuJoCo/signoff 원본을
다시 해석하며, exact candidate+prompt+MuJoCo config 세션을 CPU에서 재실행한 뒤에만
로드된 deployment의 모든 state key/tensor가 원본 candidate와 정확히 같은지 비교하고
robot object를 생성한다. 원본 경로가 하나라도 없거나 인증서 digest/weights와 다르면
연결 전에 실패한다.
로컬 trainer가 만드는 것은
의도적으로 real-output 불가인 offline candidate이며, 이 저장소에는 아직 인증된
deployment checkpoint가 없다. MuJoCo collision checker도 아직 실측
조립체 geometry로 검증되지 않았다. 따라서 첫 실제 연결은 shadow로만 수행하고,
[VALIDATION.md](VALIDATION.md)의 G6–G10
증거가 생기기 전 task-success 완료로 기록하지 않는다.

안전 supervisor가 stale/limit 위반을 감지하면 반환 객체에는 진단용 measured-hold
target이 포함되지만 실제 bus로는 전송하지 않는다. managed rollout은 fault를 latch하고
연결을 해제한다.

## 관련 코드

- `src/so101_wam/checkpoint.py`: weights-only strict checkpoint I/O
- `src/so101_wam/dataset.py`: episode checksum과 policy-rate prompt 변환
- `src/so101_wam/lerobot_factory.py`: 공식 v0.6.1 lazy factory
- `src/so101_wam/adapters/lerobot.py`: exact feature/action mapping과 lifecycle
- `src/so101_wam/rollout.py`: history prime, 10/50 Hz clock, guaranteed disconnect
- `src/so101_wam/hardware_doctor.py`: 무접속 host/dependency/device/config 진단
- `src/so101_wam/hardware_preflight.py`: 무정책 G6/G7 계측과 atomic JSON report
- `src/so101_wam/prompt_recorder.py`: torque-off G9 kinesthetic record/checksum replay
- `src/so101_wam/hardware_cli.py`: artifact와 이중 출력 gate 결합
- `src/so101_wam/deployment.py`: 외부 certification의 checkpoint/config/device 결합 검증
- `src/so101_wam/deployment_issuer.py`: G6-G9 source 검증과 immutable G10 pair 발급
- `src/so101_wam/certify_cli.py`: artifact-path-only certification CLI
