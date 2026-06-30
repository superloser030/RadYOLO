# tools/

RadYOLO 가 쓰는 **외부 도구(서드파티 repo)** 와 **보조 스크립트** 모음.
`src/` 의 파이프라인 모듈과 달리, 여기 것들은 독립 실행되거나 외부 의존성으로 동작한다.

> ⚠ 외부 repo(gigapose, dca1000)는 각자 자체 README/LICENSE 를 가진 별도
> 프로젝트다. 아래는 **우리 프로젝트에서의 역할** 요약이며, 사용법 상세는 각
> 폴더의 README 를 참고할 것.

## 객체 포즈 (객체 → 3D 씬 배치)

| 폴더 | 정체 | 우리 역할 |
|---|---|---|
| **gigapose/** ⭐ | 6DoF 객체 포즈 추정 (render-and-compare) | 핵심. 객체의 회전/위치 추정 → 3D 씬 배치. `inference_server.py`(단일 서버 + 템플릿 캐시), `prepare_templates.py` |

> 객체 **3D 메시 생성**은 로컬 도구 대신 `fal-ai/trellis` API 를 쓴다
> (`src/objects/trellis_gen.py`). ComfyUI 워크플로 JSON 은 `workflows_comfyui/` 참고.

## 레이더 캡처 (노트북 측 하드웨어)

| 폴더 | 정체 | 우리 역할 |
|---|---|---|
| **dca1000/** | TI DCA1000EVM CLI + FPGA 펌웨어 | raw ADC(.bin) 녹화 제어. `sender.py` 가 `DCA1000EVM_CLI_*.exe` 를 호출. `chirp_configs/`, `config/` 포함 |

## 보조 스크립트 (우리가 만든 유틸)

| 파일 | 역할 |
|---|---|
| **calib_manual.py** | 렌즈 왜곡(k1, k2) + fx 수동 보정. OpenCV 창에서 슬라이더로 조정하며, 오른쪽 depth top-down 포인트클라우드에서 벽/바닥이 직선이 되면 OK. `s` 저장 → `config/calib.json` + `data/scene/background_undist.jpg` |
| **calib_server.py** | 위 보정의 **웹 버전 백엔드**. `calib_web.html` 서빙(localhost:8765), POST `/api/calib/save` 로 `config/calib.json` 저장 + undistort 이미지 생성 |
| **calib_web.html** | 브라우저 렌즈 보정 UI (calib_server 가 띄움) |
| **create_recording_params.py** | AWR1642 `.cfg` 기반 파라미터를 **하드코딩**해 `iqData_RecordingParameters.mat` 1회 생성. 평소엔 `receiver._write_mat`(메타 수신)가 자동 생성하므로 디버그/오프라인용 |
| **webcam_probe.py** | 외장 웹캠(DSHOW idx0)의 해상도×포맷(MJPG/YUY2)별 **실제 fps 측정** → `sender.toml` 레벨 해상도 결정용. probe_*.jpg 로 화질도 확인 |

> 레이더↔카메라 **외부 캘리브레이션**은 `src/utils/radar_cam_calib.py` 로 옮겼고
> `python main_r.py --calib` 으로 실행한다(경량 수신+radar+YOLO+뷰어 + yaw/baseline
> 추정 스레드). 수렴하면 `config/calib_radar_cam.json` 자동 저장 → radar_fusion 이 자동 로드.

## 미사용으로 정리됨 (archive/legacy_tools/)

다음은 후보로 받아두었으나 실제 파이프라인에서 안 쓰여 `archive/legacy_tools/` 로 옮겼다.

| 폴더 | 정체 | 미사용 사유 |
|---|---|---|
| **InstantMesh/** | 단일 이미지 → 3D 메시 | 3D 생성 후보였으나 Trellis API 로 대체 |
| **TripoSR/** | 단일 이미지 → 3D (빠름) | 경량 3D 후보, 미사용 |
| **nvdiffrast/** | NVIDIA 미분가능 렌더러 | gigapose 백엔드로 받아뒀으나 실제 import 안 함(env 미설치, pose 정상 동작) |
| **spconv/** | sparse convolution | 동일 — 실제 미사용 |

## 파이프라인에서의 위치

```
[노트북]  dca1000/ ──(raw .bin)──> sender.py ──UDP──> receiver.py
[데스크톱] receiver → radar_live.m(레이더) / obj_crop(객체)
                         ↓ 객체 crop
                    gigapose/ (포즈) + fal-ai/trellis API (3D 메시)
```

데이터·산출물(`gigaPose_datasets/`, `pretrained/`, `dca1000/data/` 등)은 용량이
커서 보통 git 에서 제외(gitignore)된다.
