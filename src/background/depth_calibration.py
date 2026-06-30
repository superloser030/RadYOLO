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
    if depth_img.ndim == 3:
        depth_img = depth_img[:, :, 0]
    h, w       = depth_img.shape
    depth_norm = depth_img.astype(np.float32) / 255.0

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

    qs   = np.arange(5, 100, 5)
    D_q  = np.percentile(D_flat, qs)
    R_q  = np.percentile(R_all, 100 - qs)
    a, b = np.polyfit(D_q, np.log(R_q), 1)
    a, b = float(a), float(b)

    print(f"[Calib] log회귀  a={a:.3f}, b={b:.3f} ({len(qs)}개 분위 매칭)")
    print(f"[Calib] D=1(최근): {np.exp(a+b):.2f}m  D=0(최원/외삽): {np.exp(b):.2f}m")

    corrected    = np.exp(a * depth_norm + b)
    c_min, c_max = float(corrected.min()), float(corrected.max())
    calib = {"model": "log_linear", "a": float(a), "b": float(b),
             "range_min_m": c_min, "range_max_m": c_max}
    (DEPTH_PATH.parent / "depth_calib.json").write_text(json.dumps(calib, indent=2))

    corrected_norm = apply_calib(depth_norm, calib)
    cv2.imwrite(str(DEPTH_PATH), (corrected_norm * 255).astype(np.uint8))
    print(f"[Calib] depth.png 보정 완료  (범위 {c_min:.2f}~{c_max:.2f}m)")


if __name__ == "__main__":
    calibrate_depth()
