import json
import numpy as np
import cv2
from pathlib import Path

from src.utils.config import load_camera

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPTH_PATH   = PROJECT_ROOT / "data" / "scene" / "depth.png"
BG_TS_PATH   = PROJECT_ROOT / "data" / "scene" / "background_ts.txt"
TARGETS_PATH = PROJECT_ROOT / "data" / "radar" / "targets.json"

SYNC_WINDOW_MS = 1000


def apply_calib(depth_norm, calib):
    """raw DA3 depth_norm(0~1, 1=밝음=가까움) → 보정 corrected_norm(0~1, 1=가까움).

    depth_calib.json 의 계수(a,b,range)를 재적용한다. 배경 보정(calibrate_depth)과
    '같은' 계수로 새 프레임 DA3 depth 를 변환 → 새 물체 거리도 일관되게(상태머신의
    새 물체 DA3 재추론에서 사용). 범위 밖은 clip.
    """
    a, b = calib["a"], calib["b"]
    rmin, rmax = calib["range_min_m"], calib["range_max_m"]
    corrected = np.exp(a * depth_norm + b)
    log_c   = np.log(np.clip(corrected, rmin, rmax))
    log_min, log_max = np.log(rmin), np.log(rmax)
    return 1.0 - (log_c - log_min) / (log_max - log_min + 1e-8)


def calibrate_depth():
    if not BG_TS_PATH.exists():
        print("[Calib] background_ts.txt 없음, 건너뜀")
        return
    if not TARGETS_PATH.exists():
        print("[Calib] targets.json 없음. MATLAB에서 radar_live.m 실행 후 재시도하세요.")
        return
    if not DEPTH_PATH.exists():
        print("[Calib] depth.png 없음, 건너뜀")
        return

    cam = load_camera()
    fx, cx = cam["fx"], cam["cx"]

    depth_img  = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        print("[Calib] depth.png 읽기 실패, 건너뜀")
        return
    if depth_img.ndim == 3:          # 혹시 다채널로 읽히면 첫 채널만
        depth_img = depth_img[:, :, 0]
    h, w       = depth_img.shape
    depth_norm = depth_img.astype(np.float32) / 255.0

    # 레이더 range 라이브 누적 — targets.json 은 최신 프레임만 덮어쓰므로 과거
    # background_ts 로는 못 찾는다(=0개 버그). calib 시점에 몇 초 폴링해 분포를 모은다.
    import time as _t
    R_all, seen = [], set()
    t0 = _t.time()
    print("[Calib] 레이더 range 수집 중 (4초)...")
    while _t.time() - t0 < 4.0:
        try:
            frames = json.loads(TARGETS_PATH.read_text())
        except (OSError, ValueError):
            frames = []
        for frame in frames:
            if not frame:
                continue
            fidx = frame.get("frame_idx")
            if fidx in seen:
                continue
            seen.add(fidx)
            for t in frame.get("targets", []):
                r = t.get("range_m", 0)
                if r > 0.3:
                    R_all.append(r)
        _t.sleep(0.1)

    if len(R_all) < 10:
        print(f"[Calib] 레이더 타겟 부족 ({len(R_all)}개), 건너뜀")
        return
    print(f"[Calib] 레이더 range {len(R_all)}개 수집 ({len(seen)} 프레임)")

    D_flat = depth_norm.flatten()

    # 분위수 매칭 회귀: DA3 분위수 ↔ 레이더 역분위수 (여러 점 → 끝점 2개보다 robust,
    # 먼 쪽 끝점 1개에 안 끌려 가까운 영역 정확 + 먼 쪽은 회귀선으로 외삽).
    # DA3 높음(밝음)=가까움 ↔ 레이더 작음=가까움 이라 R 은 역분위로 매칭.
    qs   = np.arange(5, 100, 5)                  # 5,10,...,95 분위
    D_q  = np.percentile(D_flat, qs)             # 오름차순 (먼→가까움)
    R_q  = np.percentile(R_all, 100 - qs)        # 내림차순 (먼→가까움)
    a, b = np.polyfit(D_q, np.log(R_q), 1)       # log(R) = a*D + b 회귀
    a, b = float(a), float(b)

    print(f"[Calib] log회귀  a={a:.3f}, b={b:.3f} ({len(qs)}개 분위 매칭)")
    print(f"[Calib] D=1(최근): {np.exp(a+b):.2f}m  D=0(최원/외삽): {np.exp(b):.2f}m")

    # 계수+범위 산출(1회) → depth_calib.json. 변환은 apply_calib 로 일원화
    # (새 물체 DA3 재추론 시 같은 계수 재적용 위함).
    corrected    = np.exp(a * depth_norm + b)
    c_min, c_max = float(corrected.min()), float(corrected.max())
    calib = {"model": "log_linear", "a": float(a), "b": float(b),
             "range_min_m": c_min, "range_max_m": c_max}
    (DEPTH_PATH.parent / "depth_calib.json").write_text(json.dumps(calib, indent=2))

    # log 스케일 정규화(근거리 대비 유지)는 apply_calib 안에 있음
    corrected_norm = apply_calib(depth_norm, calib)
    cv2.imwrite(str(DEPTH_PATH), (corrected_norm * 255).astype(np.uint8))
    print(f"[Calib] depth.png 보정 완료  (범위 {c_min:.2f}~{c_max:.2f}m)")


if __name__ == "__main__":
    calibrate_depth()
