# RadYOLO

mmWave 레이더(TI AWR1642) + 웹캠 융합 기반 **실시간 거리 추정 및 3D 장면 복원** 시스템.

레이더의 정확한 거리/방위각과 카메라의 의미 정보(클래스/마스크/깊이)를 융합해
각 객체까지의 실제 거리를 추정하고, 나아가 장면을 3D로 복원한다.

> 설계·구현·문제해결 상세는 [`report/`](report/) 의 보고서 참고.

---

## 시스템 구성

| 노드 | 역할 | 실행 |
|---|---|---|
| **노트북 (송신측)** | AWR1642 + DCA1000EVM(raw ADC) + 웹캠 캡처 → UDP 송신 | `python main_s.py` |
| **데스크톱 (수신측)** | 수신 + 신호처리(MATLAB) + 추론(YOLO/DA3/SAM2/GigaPose) + 융합 + 뷰어 | `python main_r.py` |

네트워크는 Tailscale 직결, 설정은 `config/*.toml`(IP/포트/레벨/파이프라인).

---

## 다운로드해야 하는 모델

가중치 파일은 용량이 커서 저장소에 포함되지 않는다(`.gitignore`). 아래를 직접 받아
지정 경로에 두어야 한다.

### 1) 프로젝트 `models/` 폴더

| 모델 | 파일명 | 경로 | 출처 |
|---|---|---|---|
| YOLO11x-seg | `yolo11x-seg.pt` | `models/yolo11x-seg.pt` | Ultralytics ([github.com/ultralytics/assets](https://github.com/ultralytics/assets/releases)) — `ultralytics` 설치 시 자동 다운로드도 가능 |
| SAM2 (hiera-large) | `sam2.1_hiera_large.pt` | `models/sam2.1_hiera_large.pt` | Meta SAM2 ([github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2) → checkpoints) |

> SAM2 config(`configs/sam2.1/sam2.1_hiera_l.yaml`)는 `sam2` 패키지에 포함되어 있다.

### 2) ComfyUI 모델 폴더 (별도 설치, 기본 경로 `C:/dev/ComfyUI`)

| 모델 | 파일명 | 경로 | 출처 |
|---|---|---|---|
| DepthAnythingV3 (metric) | `da3metric_large.safetensors` | `ComfyUI/models/depthanything/` | Depth-Anything-V3 (HuggingFace) |
| Real-ESRGAN x4 | `RealESRGAN_x4.pth` | `ComfyUI/models/upscale_models/` | Real-ESRGAN ([github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)) |

> ComfyUI 워크플로 JSON은 [`workflows_comfyui/`](workflows_comfyui/) 에 있다.
> DA3 metric 노드는 `normalization_mode: "Raw"` + metric npy 덤프가 필요하다(보고서 Part IV 참고).

### 3) GigaPose (외부 repo, `tools/gigapose/`)

6DoF 포즈 추정. 별도 conda 환경 + 사전학습 가중치가 필요하다.
- repo: [github.com/nv-nguyen/gigapose](https://github.com/nv-nguyen/gigapose)
- 가중치: 위 repo 안내대로 `tools/gigapose/` 하위에 배치
- 별도 환경: `conda env`(예: `gigapose`) — 경로는 `src/objects/pose_estimator.py`의 `GIGAPOSE_PY` 참고
- ⚠ 현재 포즈 정합은 **미완성**(보고서 Part VII)

### 4) TRELLIS (다운로드 불필요 — API)

3D 메시 생성은 **fal.ai API**를 사용한다. 가중치 다운로드 없이 API 키만 필요하다.
- `.env` 파일에 `FAL_KEY=...` 설정 (저장소에 포함 안 됨)

---

## 외부 의존 소프트웨어

| 소프트웨어 | 용도 | 비고 |
|---|---|---|
| **ComfyUI** | DA3 깊이 / ESRGAN 업스케일 | `http://127.0.0.1:8188`, 먼저 실행해 둘 것 |
| **MATLAB** (R2026a+) | CFAR 레이더 신호처리 | Phased Array System Toolbox 필요. `main_r.py`가 `radar_live.m` 자동 실행 |
| **TI DCA1000 CLI** | raw ADC 캡처 (송신측) | `tools/dca1000/` 에 포함 |
| **Tailscale** | 노트북↔데스크톱 직결 | 양쪽 설치 |

---

## 네트워크 설정 (Tailscale)

노트북(송신)과 데스크톱(수신)은 서로 다른 망에 있어도 Tailscale로 직결(P2P)한다.
raw ADC + 웹캠을 UDP로 보내므로 **direct(P2P) 연결과 충분한 대역폭**이 중요하다.

1. **양쪽에 Tailscale 설치 후 같은 계정으로 로그인**(같은 tailnet에 묶이도록).

2. **각 머신의 Tailscale IP 확인** (`100.x.x.x` 대):
   ```
   tailscale ip -4
   ```

3. **`config/network.toml` 에 데스크톱(수신측) IP와 포트 기입**:
   ```toml
   [network]
   desktop_ip  = "100.x.x.x"   # 데스크톱(수신) Tailscale IP
   radar_port  = 5006
   webcam_port = 5007
   meta_port   = 5008
   matlab_port = 5009          # 데스크톱 내부 (receiver -> MATLAB)
   ```

4. **direct 연결 확인** (relay 경유 시 대역폭이 급감):
   ```
   tailscale ping <상대-호스트명>
   ```
   `via DERP`(relay)가 아니라 **`direct`** 가 떠야 한다. relay로만 잡히면
   양쪽 방화벽/NAT를 점검하고, Tailscale의 직결이 잡히도록 잠시 기다리거나
   클라이언트 포트 고정(`randomizeClientPort` 비활성) 등을 시도한다.
   - 직결이 잡히면 실측 대역폭은 수백 Mbps 수준(레이더 raw + 웹캠 전송에 충분).

5. **방화벽**: 데스크톱에서 위 UDP 포트(radar/webcam/meta) 인바운드를 허용한다.

> 대역폭이 부족하면 `sender.toml` 의 `sender_mode = 1`(자동 레벨)로 두면, 측정된
> 대역폭에 맞춰 해상도/chirp 레벨을 자동으로 낮춘다.

---

## 설치 및 실행

### 1. 환경

```
conda create -n radyolo python=3.11
conda activate radyolo
pip install -r requirements.txt   # ultralytics, opencv, numpy, scipy, fal-client, sam2 등
```

### 2. 설정 (`config/`)

- `network.toml` : 노트북/데스크톱 IP, 포트
- `sender.toml`  : 레벨(해상도/fps/chirp), 카메라, DCA, 레이더 파라미터
- `receiver.toml`: 저장/파이프라인/YOLO/DynBG 설정

### 3. 실행 순서

```
# (데스크톱) ComfyUI 먼저 실행
python C:/dev/ComfyUI/main.py

# (노트북) 송신 — DCA1000 + 웹캠
python main_s.py

# (데스크톱) 수신 + 파이프라인 + 뷰어
python main_r.py
```

뷰어: `http://<데스크톱_IP>:8000/src/viewer/viewer.html` (자동 오픈)

### 주요 플래그 (`main_r.py`)

| 플래그 | 의미 |
|---|---|
| (없음) | 전체 파이프라인 (배경→깊이→마스크→라이브) |
| `--calib` | 레이더↔카메라 외부 캘리브(yaw) — 사람이 중앙에서 좌우로 이동 |
| `--no-radar` | 레이더 없이 웹캠만 |
| `--skip-bg` / `--skip-depth` / `--skip-3d` | 해당 단계 생략 |
| `--viewer-only` | 뷰어 서버만 |

뷰어 콘솔: `2d` / `3d` 전환, `dist live|debug|full`(거리 표시 모드).

---

## 디렉토리 구조

```
RadYOLO/
  main_r.py / main_s.py     # 수신측 / 송신측 진입점
  src/
    transmission/           # sender, receiver
    objects/                # radar_fusion, obj_crop, trellis_gen, pose_estimator
    background/             # depth, dynamic_bg_fill, upscale, bg_select
    utils/                  # config, gpu_scheduler, radar_cam_calib, archive
    viewer/                 # viewer.html (2D/3D 뷰어)
  matlab/                   # radar_live.m + modules (CFAR)
  workflows_comfyui/        # DA3/ESRGAN ComfyUI 워크플로
  config/                   # network/sender/receiver .toml
  tools/                    # dca1000(캡처 CLI), gigapose(외부 repo)
  report/                   # 보고서 (LaTeX)
  models/ data/ db/ archive/ logs/   # 가중치·생성물 (gitignore, 구조만 유지)
```

> `models/` `data/` `db/` `archive/` `logs/` 는 가중치·생성물이라 내용은 저장소에
> 포함되지 않고 **디렉토리 구조만** 유지된다(`.gitkeep`).
