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


def calibrate_depth():
    if not BG_TS_PATH.exists():
        print("[Calib] background_ts.txt 없음, 건너뜀")
        return
    if not TARGETS_PATH.exists():
        print("[Calib] targets.json 없음. MATLAB에서 cfar_detect.m 실행 후 재시도하세요.")
        return
    if not DEPTH_PATH.exists():
        print("[Calib] depth.png 없음, 건너뜀")
        return

    cam = load_camera()
    fx, cx = cam["fx"], cam["cx"]

    bg_ts = int(BG_TS_PATH.read_text().strip())

    depth_img  = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
    h, w       = depth_img.shape
    depth_norm = depth_img.astype(np.float32) / 255.0

    all_frames = json.loads(TARGETS_PATH.read_text())

    # 레이더 range 수집
    R_all = []
    for frame in all_frames:
        if frame is None:
            continue
        if abs(frame.get("ts_ms", 0) - bg_ts) > SYNC_WINDOW_MS:
            continue
        for t in frame.get("targets", []):
            r = t.get("range_m", 0)
            if r > 0.3:
                R_all.append(r)

    if len(R_all) < 10:
        print(f"[Calib] 레이더 타겟 부족 ({len(R_all)}개), 건너뜀")
        return

    D_flat = depth_norm.flatten()

    # 앵커 1: 가장 밝은 픽셀(1% 분위) ↔ 가장 가까운 레이더(1% 분위)
    # 앵커 2: 가장 어두운 픽셀(99% 분위) ↔ 가장 먼 레이더(99% 분위)
    D_near = float(np.percentile(D_flat, 99))   # 밝은 쪽 (near)
    D_far  = float(np.percentile(D_flat,  1))   # 어두운 쪽 (far)
    R_near = float(np.percentile(R_all,   1))   # 가까운 레이더
    R_far  = float(np.percentile(R_all,  99))   # 먼 레이더

    # log(R) = a*D + b 를 2x2 연립방정식으로 직접 풀기
    A_mat = np.array([[D_near, 1.0], [D_far, 1.0]])
    logR  = np.array([np.log(R_near), np.log(R_far)])
    a, b  = np.linalg.solve(A_mat, logR)

    print(f"[Calib] log모델  a={a:.3f}, b={b:.3f}")
    print(f"[Calib] D=1(최근): {np.exp(a+b):.2f}m  D=0(최원): {np.exp(b):.2f}m")

    # corrected = exp(a*D + b): near=small, far=large
    corrected    = np.exp(a * depth_norm + b)
    c_min, c_max = float(corrected.min()), float(corrected.max())

    # log 스케일 정규화: 근거리 대비 유지 (1→2m = 5→10m 시각적으로 동일)
    log_c     = np.log(corrected)
    log_min   = np.log(c_min)
    log_max   = np.log(c_max)
    corrected_norm = 1.0 - (log_c - log_min) / (log_max - log_min + 1e-8)

    cv2.imwrite(str(DEPTH_PATH), (corrected_norm * 255).astype(np.uint8))

    calib = {"model": "log_linear", "a": float(a), "b": float(b),
             "range_min_m": c_min, "range_max_m": c_max}
    (DEPTH_PATH.parent / "depth_calib.json").write_text(json.dumps(calib, indent=2))
    print(f"[Calib] depth.png 보정 완료  (범위 {c_min:.2f}~{c_max:.2f}m)")


if __name__ == "__main__":
    calibrate_depth()
