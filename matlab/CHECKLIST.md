# radar_live.m 검증 체크리스트

새 예제 기반 파이프라인을 **처음 켤 때** 순서대로 확인. 혼자 못 돌려봤으니
(MATLAB·하드웨어·라이브 데이터 필요) 아래를 직접 확인해야 함.

기존 `cfar_detect_live.m` 등은 `legacy_matlab/` 에 백업됨 — 새 게 깨지면 비교/복귀 가능.

---

## 0. 사전 준비

- [ ] **Sensor Fusion and Tracking Toolbox** 설치 확인
  ```matlab
  ver   % 목록에 'Sensor Fusion and Tracking Toolbox' 있어야 partitionDetections/trackerJPDA 동작
  ```
  없으면 Add-On Explorer 에서 설치.
- [ ] 레이더 캡처 중 (`data/radar/*.bin` 이 쌓이고 있어야 함)
- [ ] `data/radar/iqData_RecordingParameters.mat` 존재 (없으면 send_meta 로 생성)
- [ ] `data/radar/iqData_timestamps.csv` 존재

## 1. 첫 실행 — 에러 위치 파악

```matlab
cd D:\projects\RadYOLO\matlab
radar_live
```
- [ ] 객체 생성 단계(ULA/rdresp/cfar/doa/tracker) 통과 — 여기서 에러면 파라미터 문제
- [ ] 첫 폴링에서 `[RADAR] frame N | X det | Y track | Z s` 출력 시작

## 2. ⚠ 위험 1 — 데이터큐브 차원 (readRadarCube)

가장 먼저 깨질 수 있는 곳. `read(fr,1)` 큐브가 `(samples x 4 x chirps)` 가정.
- [ ] 폴링 루프 멈추고 수동 확인:
  ```matlab
  fr = dca1000FileReader(recordLocation="D:\projects\RadYOLO\data\radar");
  fr.CurrentPosition = 1;  iq = read(fr,1);  size(iq{1})
  % → [256 x 4 x chirps] 형태여야 함 (samples=256, rx=4)
  ```
- [ ] 차원 순서가 다르면 `readRadarCube.m` 의 `cat(2,...)`/`squeeze` 축 수정

## 3. ⚠ 위험 2 — angle 좌우 부호 (estimateAzimuth) ⭐

**가장 중요.** 가상배열 채널 순서가 한 번에 맞으리란 보장 없음.
- [ ] **정면(0°)** 에 물체 1개 → 출력 az 가 0 근처인지
- [ ] 물체를 **오른쪽**으로 → az 부호가 일관되게 한쪽으로 (왼쪽이면 반대)
- [ ] 좌우가 뒤집히면 둘 중 하나로 교정:
  - `radar_live.m` 의 `ElementSpacing` → `-lambda/2` 로 부호 반전, 또는
  - `radar_fusion.py` 의 `NEGATE_AZIMUTH = True`
- [ ] 2D 뷰어에서 chair/laptop 방향이 화면과 맞는지 최종 확인

## 4. ⚠ 위험 3 — CFAR 검출 수 (cfarDetect)

- [ ] `X det` 가 0 이면 → 검출 없음: `ProbabilityFalseAlarm` ↑ (1e-4 → 1e-3),
      또는 거리게이트 `p.rGateMin/Max` 확대
- [ ] `X det` 가 수백이면 → 노이즈 과다: `ProbabilityFalseAlarm` ↓ (1e-4 → 1e-6)
- [ ] 적정: 물체당 수 개~수십 개 점

## 5. 추적 안정성 (trackObjects → targets.json)

- [ ] `Y track` 수가 실제 물체 수와 비슷한지 (의자 1 + 노트북 1 ≈ 2)
- [ ] `data/radar/targets.json` 이 갱신되는지:
  ```matlab
  fileread('D:\projects\RadYOLO\data\radar\targets.json')
  ```
- [ ] **거리가 안정적인지** (예전엔 1→3→5m 튐 / 이제 track 이라 부드러워야 함)
- [ ] 실제 1m 물체가 1m 근처로 나오는지 (벽 반사에 안 끌려가는지)

## 6. Python 융합 연동

- [ ] `main_r.py` 가 자동 실행하는 MATLAB 호출을 `radar_live.m` 로 교체했는지
      (기존엔 `cfar_detect_live.m` 를 띄웠음 — `_start_matlab_cfar` 수정 필요)
- [ ] `radar_fusion.py` 키 호환: targets.json 의 `range_m/azimuth_deg/velocity_mps`
      — 키 동일하므로 그대로 동작. NEGATE_AZIMUTH 만 3번 결과로 조정
- [ ] 2D 뷰어에 거리/방위 표시되는지

---

## 알려진 미해결 / 다음 작업

- `main_r.py::_start_matlab_cfar` 가 `cfar_detect_live.m` 를 가리킴 → `radar_live.m` 로 변경 필요
- range-rate(doppler) 부호 규약: 다가옴(+)/멀어짐(-) — tracker 와 일치 확인
- velocity 정확도: TDM 최대속도 ~1 m/s 한계 (빠른 움직임 aliasing)
- tracker 파라미터(ClutterDensity 등)는 ParkingLot 차량 기준 → 실내 물체로 튜닝 여지
