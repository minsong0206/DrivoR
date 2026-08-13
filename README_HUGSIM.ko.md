# DrivoR HUGSIM Client

[English](README_HUGSIM.md)

---

이 저장소([valeoai/DrivoR](https://github.com/valeoai/DrivoR)의 fork)는
[HUGSIM](https://github.com/hyzhou404/HUGSIM) closed-loop 벤치마크용 DrivoR
테스트 client 구현입니다 -- 같은 벤치마크의 UniAD/VAD/LTF client와 나란히
쓰입니다.

구현 기반:

[Driving on Registers](https://valeoai.github.io/driving-on-registers/)

[arXiv Paper](https://arxiv.org/abs/2601.05083), CVPR 2026

---

## 1. 원본 `valeoai/DrivoR` 대비 변경 사항

순수 fork가 아니라, HUGSIM 연동을 위해 원본 위에 아래를 추가한 상태입니다
(원본 [valeoai/DrivoR](https://github.com/valeoai/DrivoR)에는 없음):

- **`drivor_e2e.sh`** -- HUGSIM 쪽 `sim/utils/launch_ad.py::launch()`가
  호출하는 launcher 스크립트(`zsh ${shell_path} ${cuda_id} ${output}`).
  `(cuda_id, output_dir)`를 위치 인자로 받아서 아래 client entry point를
  실행합니다.
- **`hugsim_drivor_client_raw.py`** -- 실제 entry point (확인됨: `drivor_e2e.sh`가
  이 파일을 실행하고, 이 저장소에 같이 있는 이름이 비슷한
  `hugsim_drivor_client.py`가 아님 -- 그 파일이 뭘 위한 건지는 여기서 다루지
  않음). `obs_pipe`/`plan_pipe` named pipe(§5)로 HUGSIM과 통신, observation을
  DrivoR 모델 입력 포맷으로 변환, 추론 실행, `(N, 2)` 궤적을 반환합니다.
- **HUGSIM 쪽**(별도 저장소)에는 `closed_loop_drivor.py`가 `closed_loop.py`와
  나란히 추가됨 -- `closed_loop.py` 자체는 UniAD/VAD/LTF만 연결되어 있고,
  이 client는 obs/info 형태와 루프 이후 저장/평가 단계가 달라서 재사용하지
  않았습니다.

그 외 나머지(기본 DrivoR/navsim 모델, 학습, `eval.sh`/`metric_caching.sh` 같은
원본 평가 스크립트)는 이 저장소 자체의 [README.md](README.md)를 참고하세요 --
여기서 중복 설명하지 않습니다.

---

## 2. 설치

기본 DrivoR/navsim 환경은 이 저장소 자체의
[Installation](README.md#installations) 안내를 참고하세요. 실제로 이
fork에서는 환경을 `conda`에서 `uv`로 옮겼는데(`torch==2.1.0+cu121`),
`drivor_e2e.sh`는 그걸 가정은 하지만 실제 인터프리터 경로는 env var로
읽으므로 어떤 DrivoR 환경이든 동작합니다.

**경로 설정 (env var override)**: `drivor_e2e.sh`의 `DRIVOR_PATH`/
`DRIVOR_PYTHON`, `hugsim_drivor_client_raw.py`의 `DRIVOR_CHECKPOINT_PATH`/
`DRIVOR_CONFIG_PATH`는 전부 env var로 override 가능 (기본값은 placeholder).
자신의 환경에 맞게 설정할 것:

```bash
export DRIVOR_PATH=/path/to/your/DrivoR
export DRIVOR_PYTHON=/path/to/your/DrivoR/venv/bin/python
export DRIVOR_CHECKPOINT_PATH=/path/to/your/checkpoint.pth
export DRIVOR_CONFIG_PATH=/path/to/your/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml
```

---

## 3. Client 실행

### 수동 실행

```bash
zsh ./drivor_e2e.sh ${CUDA_ID} ${output_dir}
```

### 자동 실행 (closed-loop 평가)

실제로는 `drivor_e2e.sh`를 직접 부르지 않습니다 -- HUGSIM의 closed-loop
스크립트가 시나리오마다 이걸 실행시킵니다. 아래는 **HUGSIM 저장소**에서
(이 저장소의 환경이 아니라 HUGSIM의 `hugsim_v3` docker 환경 안에서) 실행:

```bash
sim_cuda=0
ad_cuda=1

# HUGSIM 쪽 본인 시나리오 yaml 디렉토리
scenario_dir=${SCENARIO_PATH}

for cfg in ${scenario_dir}/*.yaml; do
    echo ${cfg}
    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python closed_loop_drivor.py --scenario_path ${cfg} \
                        --base_path ./configs/sim/${dataset_name}_base.yaml \
                        --camera_path ./configs/sim/${dataset_name}_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad drivor \
                        --ad_cuda ${ad_cuda}
done
```

이건 `closed_loop_drivor.py --ad drivor`입니다 (`closed_loop.py`**가 아님**,
`closed_loop.py`는 UniAD/VAD/LTF만 연결되어 있음). `configs/sim/<dataset>_base.yaml`의
`drivor_path`를 읽어오므로, 그 값을 이 저장소의 체크아웃 경로(위 §2에서
`DRIVOR_PATH`로 잡은 그 경로)로 맞춰두면, 시나리오마다 알아서
`drivor_e2e.sh`를 실행하고 §5의 `obs_pipe`/`plan_pipe` 프로토콜로 붙여줍니다.

---

## 4. Debug 이미지

`hugsim_drivor_client_raw.py`가 시뮬레이션 스텝마다 카메라별 debug 프레임을
`${output_dir}/drivor/debug_imgs/`에 3단계로 나눠 저장합니다 -- 어느 단계에서
문제가 생겼는지 추적하기 위함:

| 폴더 | 내용 |
|---|---|
| `1_hugsim_raw/` | `obs_pipe`로 HUGSIM에서 받은 원본 RGB 프레임 그대로 (리사이즈/가공 전). HUGSIM 쪽 렌더링 문제인지 먼저 배제할 때 사용. |
| `2_precam/` | navsim `Camera` 객체로 감싸지기(`create_cam`) 직전 프레임. 현재는 리사이즈가 없어서 `1_hugsim_raw/`와 동일하지만, 나중에 리사이즈가 다시 들어갈 경우를 대비해 별도 단계로 유지. |
| `3_feature/` | DrivoR feature builder가 만든 per-camera feature map(`features['image']`), 모델에 들어가기 직전 (시각화를 위해 0-255로 min-max 정규화됨). 채널 순서/dead feature 같은 feature 추출 버그를 원본 입력과 독립적으로 확인할 때 사용. |

파일명은 각 폴더 안에서 `{frame_idx:04d}_{cam_name}.jpg`.

---

## 5. 프로토콜: HUGSIM과 이 client가 통신하는 방식

모든 AD client(이 client, UniAD_SIM, VAD_SIM, NAVSIM)는 각자 독립된
저장소/프로세스입니다 -- HUGSIM이 AD 알고리즘 코드를 직접 import하지
않습니다. 대신 HUGSIM 쪽이 실행 디렉토리에 만드는 **OS named pipe(FIFO) 2개**로
통신합니다:

```text
${output_dir}/obs_pipe    -- HUGSIM이 씀, client가 읽음 (observation)
${output_dir}/plan_pipe   -- client가 씀, HUGSIM이 읽음 (planned trajectory)
```

시뮬레이션 한 스텝은 이렇게 진행됩니다 (HUGSIM 쪽은
`closed_loop_drivor.py::create_gym_env`, 이 저장소 쪽은
`hugsim_drivor_client_raw.py`의 메인 루프 참고):

```text
HUGSIM                                    AD client (이 저장소)
──────────────────────────────────────    ──────────────────────────────────
env.step() -> (obs, info)
pickle.dumps((obs, info))
  -> write to obs_pipe          ────────>  read obs_pipe, pickle.loads()
                                            build model input from obs/info
                                            run model -> trajectory
                                            pickle.dumps(plan_traj)
  read plan_pipe, pickle.loads() <────────   -> write to plan_pipe
traj2control(plan_traj, info)
  -> action -> env.step(action)
(loop back to env.step())
```

맨 마지막에는 HUGSIM이 `obs_pipe`에 `(obs, info)` 튜플 대신 문자열 그대로
`'Done'`을 한 번 씁니다 -- client는 이걸 감지해서(`hugsim_drivor_client_raw.py`의
`if raw_data == 'Done'` 분기 참고) `(obs, info)`로 unpack하려 하지 말고 깔끔하게
종료해야 합니다.

### `obs` dict

```python
obs = {
    'rgb': {
        'CAM_FRONT': <np.ndarray, HxWx3, RGB>,
        'CAM_FRONT_LEFT': ...,
        'CAM_FRONT_RIGHT': ...,
        'CAM_BACK': ...,
        'CAM_BACK_LEFT': ...,
        'CAM_BACK_RIGHT': ...,
    }
}
```

### `info` dict (이 client가 실제로 읽는 필드)

| key | 의미 |
|---|---|
| `command` | 이산 내비게이션 커맨드 인덱스(예: 직진/좌/우), one-hot `command` 벡터를 만드는 데 사용 |
| `ego_pos` | HUGSIM 월드/OpenCV 축 기준 ego 위치 -- 이 client는 `OPENCV2IMU`로 IMU 축으로 회전시켜 사용 |
| `ego_steer` | ego heading/yaw |
| `ego_velo`, `accelerate` | 스칼라 전진 속도/가속도, 위 yaw를 이용해 x/y로 분해 |
| `cam_params` | 카메라별 dict, 각각 `intrinsic`(`fovx`/`fovy`/`H`/`W`/`cx`/`cy`)과 `l2c`(lidar-to-camera 4x4) 포함 -- 모델용 카메라별 projection 생성에 사용 |
| `ego_box`, `obj_boxes`, `collision`, `rc`, `timestamp` | HUGSIM이 `data.pkl`에 로깅하는 값들, 이 client의 모델 자체는 쓰지 않음 |

### `plan_traj` (client가 반환해야 하는 값)

`(N, 2)` shape의 `numpy.ndarray` -- ego-relative `(x, y)` 좌표계의 미래
waypoint N개 (`hugsim_drivor_client_raw.py`의 `way_points[:, :2]`; 이
client의 모델은 실제로 `(N, 3)`을 예측하지만 pipe로는 앞 두 컬럼만
돌려줌). HUGSIM의 `traj2control()`이 이걸 그대로
acceleration/steer-rate control로 변환합니다 -- **HUGSIM 쪽
`traj2control`이 가정하는 형태를 먼저 확인하지 않고** 근본적으로 다른
궤적 길이/컨벤션을 돌려주지 말 것.

---

## 6. 다른 AD 알고리즘 추가하기

DrivoR 대신 다른 모델을 붙이거나(또는 나란히 하나 더 추가하고 싶다면),
UniAD_SIM/VAD_SIM/NAVSIM/DrivoR가 이미 실제로 따르고 있는 패턴을 그대로
따르면 됩니다 -- **HUGSIM이나 이 저장소에 코드를 합치는 게 아니라 독립된
새 저장소**로 만드는 것입니다:

1. **독립 저장소, 독립 환경.** AD 알고리즘은 자기 의존성/체크포인트/설정을
   HUGSIM과 완전히 분리해서 유지합니다 -- DrivoR가 HUGSIM 컨테이너와 다른
   torch 빌드로 자체 `uv` 환경을 돌리는 이유가 이것입니다.

2. **하나의 entry-point 스크립트** (이 저장소의 `hugsim_drivor_client_raw.py`,
   `drivor_e2e.sh`)가 다음을 수행:
   - `${output}/{obs,plan}_pipe`에 아직 없으면 `os.mkfifo`로 `obs_pipe`/
     `plan_pipe` 생성
   - 루프: `obs_pipe` 읽고 `pickle.loads` -> `'Done'`이면 정리하고 종료 ->
     아니면 `(obs, info)`로 unpack -> §5에 정리된 `obs`/`info` 필드로 본인
     모델 입력 구성 -> 추론 실행 -> 결과를 `(N, 2)` `numpy.ndarray`로 만들어
     `pickle.dumps` -> `plan_pipe`에 쓰기

3. **launcher 셸 스크립트** (`drivor_e2e.sh`의 역할)가 `(cuda_id, output_dir)`를
   위치 인자로 받음 -- 이게 정확히 `sim/utils/launch_ad.py::launch()`가
   호출하는 시그니처 (`zsh ${shell_path} ${cuda_id} ${output}`). 이 두
   인자를 받아서 entry-point 스크립트를 정상적으로 띄우기만 하면
   인터프리터/셸은 무엇이든 상관없음.

4. **HUGSIM config에 연결**, 둘 중 하나:
   - **(권장, 기존 client와 동일)** client가 필요로 하는 obs가
     `closed_loop.py`가 이미 만드는 것과 같다면 `closed_loop.py`에
     `uniad`/`vad`/`ltf` 옆에 새 `--ad <name>` 분기를 추가. obs/info
     형태가 다르거나 루프 이후 저장/평가 단계가 다르면(DrivoR가
     `closed_loop_drivor.py`를 따로 쓰는 이유) `closed_loop_drivor.py`를
     본떠 새 `closed_loop_<name>.py`를 추가.
   - 어느 쪽이든, `configs/sim/<dataset>_base.yaml`에 기존
     `uniad_path`/`vad_path`/`ltf_path`/`drivor_path` 필드를 따라 새
     `<name>_path` 필드를 추가해서 새 저장소를 가리키게 함.

5. **모델이 아직 준비 안 됐다면 파이프 프로토콜부터 독립적으로 테스트**할
   것 -- `obs_pipe`를 읽고 직선 `(N, 2)` 궤적을 그대로 돌려주는 스텁
   client만으로도 배관(파이프 생성, `'Done'` 처리, `traj2control`이 궤적
   형태를 받아들이는지)을 실제 추론 연결 전에 확인할 수 있음.

이 프로토콜 자체는 DrivoR 전용이 아니라 HUGSIM 자체의 closed-loop
규약이라, `closed_loop_drivor.py`/`closed_loop.py`와 이 파일의 §5만
읽어도 DrivoR 모델 코드를 몰라도 새 client를 처음부터 만들 수 있습니다.
