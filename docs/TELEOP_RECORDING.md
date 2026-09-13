# 리더 팔로 시연 기록하기

`so101-wam-record-demo`는 사람이 리더 팔을 움직이는 동안 팔로워 로봇에
명령을 보내고, 양쪽 손목 영상·측정 관절 위치·전송 결과를 함께 저장한다.
실물 구동 검증은 아직 하지 않았다. 가짜 장비 테스트로 데이터 대응과 오류 처리를 확인한다.

## 준비와 실행

양쪽 SO-101 팔로워, 양쪽 리더, 손목 카메라 두 대가 필요하다.
LeRobot 버전은 `0.6.1`이다. [하드웨어 준비](BRINGUP.md)에 따라
카메라 시각, 실측 홈 위치, 관절 한계, 양팔 보정값을 먼저 확인한다.

`configs/lerobot_hardware.template.toml`을 로컬 설정으로 복사하고
팔로워와 리더의 실제 장치 경로·보정 정보를 채운다. `[leader]` 설정은 다음과 같다.

```toml
[leader]
robot_id = "zero01_leader"
left_port = ""       # 왼쪽 리더의 실제 serial 경로
right_port = ""      # 오른쪽 리더의 실제 serial 경로
calibration_dir = "" # LeRobot 리더 보정 파일 디렉터리
calibration_id = ""  # 해당 보정 결과의 식별자
```

네 팔의 serial 경로는 모두 달라야 한다. 리더와 팔로워에는 각각의 보정 파일을
사용한다. 두 장치 모두 팔 관절은 degree 단위, 그리퍼는 LeRobot의 정규화 단위를 쓴다.
기록기가 자동 보정을 수행하지 않으므로 기존 보정 파일이 있어야 한다.

실측 설정을 준비한 뒤 로컬 파일의 `[runtime]`에서 `actuation_enabled = true`로
설정한다. 기록 시작 시 팔로워가 저장된 홈 위치에 있어야 하며, 리더도 팔로워와
같은 자세로 맞춰야 한다. 첫 명령 전에 두 조건을 확인한다.

먼저 3초 동안 데이터가 정상적으로 저장되는지 확인한다.

```bash
so101-wam-record-demo \
  --config configs/lerobot_hardware.local.toml \
  --output-dir data/teleop \
  --task "색상 컵 쌓기" \
  --task-index 10 \
  --episode-index 1 \
  --duration 3
```

소스에서 실행할 때는 `PYTHONPATH=src python -m so101_wam.demo_recorder`를 쓴다.
같은 작업은 같은 `task-index`로 기록하고, 다시 촬영할 때마다 `episode-index`를
바꾼다. 기존 파일은 덮어쓰지 않는다. 데이터 품질을 확인한 뒤 `--duration`을
작업 전체가 들어가는 길이로 늘린다. 기록 길이는 카메라 주기에 맞아야 한다.

## 저장되는 값

출력은 기존 로더로 읽을 수 있는 `demo_000001.npz`와 `demo_000001.json`이다.
각 행에는 명령을 보내기 **전** 관측과 그 관측에 이어 보낸 명령이 들어간다.
`--duration`은 첫 관측부터 마지막 관측까지의 시간이다. 마지막 행에도 명령을
보내지만 그 뒤의 상태는 이 에피소드에 없으므로 마지막 명령의 결과까지
검증한 기록으로 해석하면 안 된다.

| 값 | 의미 |
| --- | --- |
| NPZ `wrist_rgb` | 좌우 손목 RGB 영상 |
| NPZ `joint_state` | 팔로워에서 읽은 12축 현재 관절 위치 |
| NPZ `action` | LeRobot `send_action()`이 반환한 실제 전송 목표값 |
| NPZ `timestamp` | 첫 관측부터 지난 실제 시간 |
| JSON `metadata.requested_action` | 리더에서 읽은 조작 목표값 |
| JSON `metadata.commanded_action` | 속도·관절 제한을 적용해 드라이버에 전달한 목표값 |
| JSON `metadata.timing` | 관측·리더 읽기·전송의 시작과 종료 시각, 좌우 카메라 시각 |

요청값, 드라이버에 전달한 값, 드라이버가 반환한 값은 서로 다를 수 있다.
`action`에는 반환된 값만 넣는다. 반환값이 없으면 요청값이나 현재 관절 위치로
대신 채우지 않고 기록을 중단한다. 전송 목표값은 로봇이 그 위치에 도달했다는
측정값이 아니다. 도달 여부는 다음 관절 관측이나 작업별 평가로 확인해야 한다.

카메라의 영상과 시각은 LeRobot 내부 잠금 안에서 함께 복사한다.
관측·전송 시각은 `monotonic`, 카메라 버퍼 시각은 `perf_counter` 기준으로
원점과 함께 저장한다. 이는 호스트에서 얻은 시각이며 센서의 노출 시각까지
하드웨어로 동기화했다는 뜻은 아니다.

명령은 카메라 주기마다 한 번 보낸다. 기본 설정에서는 **30Hz 수집·30Hz 전송**이다.
각 명령에는 기존 `max_delta_per_servo_tick` 제한을 그대로 적용하며,
카메라 주기가 설정된 servo 주기보다 빠르면 실행을 거부한다.
학습기의 50Hz 보간을 실제 50Hz 측정이나 전송으로 해석하면 안 된다.

## 중단과 검증 범위

오래되거나 반복된 카메라 프레임, 좌우 시각 차이, 관절 한계 위반,
전송 실패, 잘못된 반환값, 기록 주기를 넘긴 지연이 발생하면 중단하고
연결된 장치를 정리한다. `Ctrl+C`도 장치를 정리한다. 중단된 기록은 정상
에피소드로 저장하지 않는다. 양팔 전송은 원자적이지 않으므로 한쪽 전송 후
다른 쪽에서 오류가 나면 이미 보낸 명령까지 취소되는 것은 아니다.
정상 종료 때도 연결을 해제하며, LeRobot 기본 설정에 따라 팔로워 토크가 꺼진다.
종료 시점에는 물체를 내려놓고 팔이 지지되는 자세가 되도록 시연을 구성한다.

저장 시 체크섬과 다시 읽은 데이터의 fingerprint를 검증한다.
`round_trip_verified`는 파일의 무결성을 뜻한다. 작업 성공은 별도 평가 대상이며
`task_success`는 `null`로 남긴다. 가짜 장비를 주입한 테스트는
`injected_devices_unverified`로 표시한다.

전체 에피소드는 12초를 넘겨 저장할 수 있다. 다만 현재 프롬프트 로더는
긴 시연을 자동으로 나누지 않으므로, physical prompt로 쓰려면 별도의
3–12초 에피소드를 준비해야 한다. 현재 기록기는 영상을 메모리에 보관한다.
640×480 RGB 두 대, 30Hz 기준 원시 영상만 약 53MiB/s이고 파일 변환 중
복사본도 생긴다. 장시간 기록에는 메모리 사용량이 제약이 된다.

기존 `so101-wam-record-prompt`는 명령을 보내지 않는 키네스테틱 기록용이다.
이 문서의 `record-demo`는 사람이 리더로 팔로워를 구동하는 별도 명령이다.

## 근거

- [ACT 논문 §IV](https://arxiv.org/html/2304.13705): 팔로워 관절 위치를 관측으로,
  리더 관절 위치를 행동 목표로 기록한다. 두 값의 차이를 보존해야 한다.
- [LeRobot 0.6.1 SOFollower](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py):
  `send_action()`은 제한을 적용한 `Goal_Position`을 쓰고 그 값을 반환한다.
- [LeRobot 0.6.1 BiSOLeader](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/teleoperators/bi_so_leader/bi_so_leader.py):
  좌우 리더 관절을 읽어 접두사가 붙은 행동 사전으로 반환한다.
