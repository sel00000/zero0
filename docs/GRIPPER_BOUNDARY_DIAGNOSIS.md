# 그리퍼 경계 진단 결과

후속 [경계 보완](GRIPPER_BOUNDARY_FIXES.md)은 활성 소스와 설정을 변경했다.
아래 해시와 검증 결과는 동결 진단 당시 기준이며, 현재 소스와의 일치를
뜻하지 않는다. 기존 진단 산출물은 보존했다.

## 범위

이 문서는 `runs/gripper_boundary_diagnostic_001/`의 실제 산출물만 요약한다.
진단은 동결된 한 체크포인트와 기존 입력 아티팩트에 한정했다. 실행 순서는
`freeze` 1회, seed 7 1회, seed 13 1회, `report` 1회였다. seed별 전송
예산은 최대 1회였고, 새 seed, 재시도, 학습, 하드웨어 실행은 없었다.

소스 안전 이력과 기존 산출물은 보존했다. 기존 NPZ는 명령 정렬
자료로 남아 있으며, 거부 관측값은 이 진단의 별도 JSON 이벤트에 기록됐다.

## 산출물

- 프로토콜: `runs/gripper_boundary_diagnostic_001/protocol.json`
- 보고서: `runs/gripper_boundary_diagnostic_001/report.json`
- 프로토콜 SHA256:
  `037fc7bd94502501c11af94502d46de087d88a2b64422e3b612fd029e193a4ab`
- 보고서 SHA256:
  `1eabd015ee7112692391fa9b89cc9bd4c0cfdf636455c4bc237b25f985cdde01`
- 진단 스크립트 SHA256:
  `a402c496593e3db842a2c9c6148da96b04b0e4f80f182462f9ab48aaecdb0576`

프로토콜은 입력 32개와 소스 94개의 SHA256을 기록했다. 이전 세션의
476개 해시 매니페스트는 유실됐으므로, 이 문서는 476개 전체 재검증을
주장하지 않는다. 현재 범위에서는 32개 입력, 94개 소스, 위 스크립트
해시만 확인했다.

보고서 매니페스트는 `report.json`을 제외한다. 독립 재계산 결과
`runs/gripper_boundary_diagnostic_001/`에는 파일 36개가 있고,
보고서 매니페스트 35개 항목은 모두 현재 바이트와 일치했다.

## 결과

두 조건 모두 원래 안전 거부를 재현했다. 최종 보고서의 각 trial은
`evidence_complete=true`이고 상태는 `original_rejection_reproduced`다.
각 seed의 `result.json`은 최종 감사를 요구하는 중간 상태로
`evidence_complete=false`를 기록했으며, 최종 판정은 `report.json`에 있다.

| seed | 상태 | base_send_calls | sent_count | 안전 사유 |
| --- | --- | ---: | ---: | --- |
| 7 | `original_rejection_reproduced` | 1 | 1 | `observation_joint_limit:right_gripper.pos:below` |
| 13 | `original_rejection_reproduced` | 1 | 1 | `observation_joint_limit:right_gripper.pos:below` |

두 seed 모두 `integrity_error=null`이다. 각 seed는 accepted/transmitted
servo 1개 뒤에 최종 거부 이벤트 1개를 기록했고, 그 최종 이벤트는
`accepted=false`, `sent=false`였다.

## 수치 증거

안전 하한은 관측 벡터의 `right_gripper.pos` 값에 대해 `0.0`이다.
MuJoCo 네이티브 조인트 범위는 `[-0.174533, 1.7453292]` rad,
액추에이터 `ctrlrange`는 `[-0.17453, 1.74533]` rad,
좌표계 변환의 네이티브 입력 범위는 `[-0.17453, 1.7453292]` rad이고
canonical 관측 단위는 `0..100`이다.

| metric | unit | seed 7 | seed 13 |
| --- | --- | ---: | ---: |
| observed right gripper / observed32 | canonical 0..100 | -3.430311608099146e-06 | -3.4300005609111395e-06 |
| canonical64 | canonical 0..100 | -3.430311691130403e-06 | -3.4300006483029845e-06 |
| raw_ctrl_rad | rad | -0.17453 | -0.17453 |
| qpos_rad | rad | -0.17453006585715458 | -0.174530065851183 |
| qvel_rad_s | rad/s | 2.9896313715421604e-05 | 2.9895339380390267e-05 |
| joint_margin_rad | rad | 2.9341428454121576e-06 | 2.9341488169964958e-06 |
| coordinate_margin_rad | rad | -6.585715459084263e-08 | -6.58511830065045e-08 |

`raw_ctrl_rad=-0.17453`은 액추에이터 `ctrlrange` 하한과 같다. 이 값만으로는
위반을 뜻하지 않는다. 보고서가 기록한 `raw_ctrl_rad`는 측정된 유효
액추에이터 제어값이 아니다.

`qpos_rad`는 조인트 하한보다 약 `2.934e-6` rad 위에 있었다. 그러나 같은
네이티브 qpos는 변환 입력 하한 `-0.17453`보다 약 `6.586e-8` rad 낮았다.
canonical float64도 이미 음수였으므로, 음수 부호가 float32 변환만으로
생겼다고 주장할 수 없다. 안전 거부는 이 float32 관측값이 안전 하한
`0.0`보다 작았기 때문에 발생했다.

모델 시간 전이는 `physics_timestep_s=0.005`를 4회 적용한
`model_time_s=0.02` 구간이었다. 어느 물리 substep에서 경계를 넘었는지는
관측되지 않았고, 물리적 메커니즘도 입증되지 않았다.

## 런타임

호스트 Python 빌드가 바뀌어 원래 실행 맥락과 불일치했다. 현재 호스트는
Python 3.12.3, GCC 13.3.0, 빌드 시각 `Aug 31 2026 10:18:26`이고, 원래
맥락은 Python 3.12.3, GCC 13.3.0, 빌드 시각 `Jun 19 2026 12:46:00`이다.

이 진단은 작업 캐시에 복원한 원래 빌드를 사용했다. 복원 근거는
[Ubuntu snapshot InRelease](https://snapshot.ubuntu.com/ubuntu/20260707T120000Z/dists/noble-security/InRelease)
및 [Ubuntu snapshot service 문서](https://ubuntu.com/server/docs/how-to/software/snapshot-service/)다.
Ubuntu 2018 키 `F6ECB3762474EDA9D21B7022871920D1991BC93C`로 InRelease를
검증했고, `Packages.xz` SHA256
`b8fee783b16c646709e77b0cb5273509af8b613bea907628c26ae07a2d398269`를
확인했다. 캐시 위치는 `.pytest_cache/gripper-runtime.jAPgEL/`이며,
서명 메타데이터와 python3.12 `3.12.3-1ubuntu0.15` 계열 deb 5개가 있다.
패키지 URL은 snapshot `pool/main/p/python3.12/*.deb` 경로다.

사용한 실행 환경은 다음과 같다. `freeze`, seed 7, seed 13, `report`의
고정 attempt는 이미 소비됐으므로, 이 블록은 재실행 지시가 아니라
기록이다. 아래 두 경로는 복원한 Python과 프로젝트 가상환경의 자리표시자다.

```bash
SO101_DIAG_ROOT=/absolute/path/to/restored-python
SO101_PROJECT_VENV=/absolute/path/to/project/.venv
PYTHONHOME="$SO101_DIAG_ROOT/usr" \
PYTHONNOUSERSITE=1 \
PYTHONPATH="src:$SO101_PROJECT_VENV/lib/python3.12/site-packages" \
LD_LIBRARY_PATH="$SO101_DIAG_ROOT/usr/lib/x86_64-linux-gnu" \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
"$SO101_DIAG_ROOT/usr/bin/python3.12" \
scripts/gripper_boundary_diagnostic.py <stage>
```

## 검증

이 작업 전 승인된 검증 집계는 중복 없는 952개 테스트 통과다. 구성은
호스트 런타임 892개를 10개 배치로 실행한 결과와 새 진단 테스트 60개다.
복원 런타임 표적 검증은 관련 155개 통과를 기록했으며, 이 155개는 60개
진단 테스트와 95개 관련 테스트로 구성되어 추가 unique 집계가 아니다.
실패와 skip은 0개였다. 단일 프로세스 전체 suite는 이전에 SIGTERM 143으로
중단됐고 원인은 확정되지 않았으므로, 단일 프로세스 전체 suite PASS는
주장하지 않는다.

이번 문서 작성 후 재확인한 표적 검증은 다음과 같다.

- `tests/test_gripper_diagnostic.py`: 60 passed
- Ruff `scripts/gripper_boundary_diagnostic.py`: passed
- mypy `scripts/gripper_boundary_diagnostic.py`: passed with `MYPYPATH=src`
- 문서 증거 대조: passed
- diff whitespace check: passed

## 해석 한계

이 결과는 과제 성공, 정책 개선, 일반화, zero-shot 성능, 배포 준비성,
하드웨어 안전성을 주장하지 않는다. 보고서도
`policy_improved=false`, `zero_shot_claimed=false`,
`deployment_ready=false`, `real_world_success_claimed=false`를 기록했다.
