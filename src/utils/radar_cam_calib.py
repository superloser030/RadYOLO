import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERLAY_PATH = PROJECT_ROOT / "data" / "scene" / "live_overlay.json"
OUT_PATH     = PROJECT_ROOT / "config" / "calib_radar_cam.json"

AZ_CENTER    = 35.0
NEAR_GATE    = 2.0
FAR_GATE     = 6.0
MIN_SAMPLES  = 20
MAX_SAMPLES  = 800
CONVERGE_DEG = 0.15
POLL_SEC     = 0.3
MIN_N        = 3
DUP_DR       = 0.3
DUP_DAZ      = 3.0


def _residuals(params, az_cam, az_rad, rng):
    yaw, tx, tz = params
    yawr = math.radians(yaw)
    azr  = np.radians(az_rad)
    xr = rng * np.sin(azr)
    zr = rng * np.cos(azr)
    xc = np.cos(yawr) * xr + np.sin(yawr) * zr + tx
    zc = -np.sin(yawr) * xr + np.cos(yawr) * zr + tz
    az_pred = np.degrees(np.arctan2(xc, zc))
    return az_cam - az_pred


def _read_overlay():
    try:
        return json.loads(OVERLAY_PATH.read_text())
    except (OSError, ValueError):
        return None


def calibrate_loop(cam, shutdown_event=None):
    fx, cx = cam["fx"], cam["cx"]
    print(f"[Calib] 외부 캘리브 시작 — 물체를 가까이(1~2m) 중앙에서 좌우로 움직이세요.")

    samples    = deque(maxlen=MAX_SAMPLES)
    range_gate = NEAR_GATE
    stage      = "near"
    prev_yaw   = None
    yaw = tx = tz = 0.0

    while shutdown_event is None or not shutdown_event.is_set():
        overlay = _read_overlay()
        if overlay:
            objs = []
            for o in overlay:
                r, az, bb, npts = o.get("range_m"), o.get("az"), o.get("bbox"), o.get("n", 0)
                if r is None or az is None or not bb or not npts:
                    continue
                objs.append((float(r), float(az), int(npts), bb))
            for i, (r, az, npts, bb) in enumerate(objs):
                if npts < MIN_N:
                    continue
                if any(j != i and abs(r - r2) < DUP_DR and abs(az - az2) < DUP_DAZ
                       for j, (r2, az2, _, _) in enumerate(objs)):
                    continue
                bcx = (bb[0] + bb[2]) / 2.0
                az_cam = math.degrees(math.atan2(bcx - cx, fx))
                if abs(az_cam) > AZ_CENTER:
                    continue
                if r > range_gate:
                    continue
                samples.append((az_cam, az, r))

        n = len(samples)
        if n >= MIN_SAMPLES:
            arr = np.array(samples)
            ac, ar, rg = arr[:, 0], arr[:, 1], arr[:, 2]

            yaw = float(np.median(ac - ar)); tx = tz = 0.0
            res = _residuals([yaw, 0, 0], ac, ar, rg)
            rms = float(np.sqrt(np.mean(res ** 2)))

            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(json.dumps({
                "yaw_offset_deg": round(yaw, 3),
                "tx": round(tx, 4), "tz": round(tz, 4),
                "n_samples": n, "rms_residual_deg": round(rms, 3),
                "stage": stage,
            }, indent=2))

            d = abs(yaw - prev_yaw) if prev_yaw is not None else 99
            print(f"[Calib] {stage:4} n={n:3} gate≤{range_gate:.0f}m | "
                  f"yaw={yaw:+.2f}° | rms={rms:.2f}° Δ={d:.2f}°"
                  + ("  ✓수렴" if d < CONVERGE_DEG else ""))

            if stage == "near" and d < CONVERGE_DEG and n >= MIN_SAMPLES * 2:
                stage = "fine"; range_gate = FAR_GATE
                samples.clear()
                print(f"[Calib] ── near 수렴(yaw={yaw:+.2f}°) → fine: 먼 거리(≤{FAR_GATE:.0f}m) 추가 ──")
            prev_yaw = yaw

        time.sleep(POLL_SEC)

    print(f"[Calib] 종료. 최종 결과 → {OUT_PATH.relative_to(PROJECT_ROOT)}")
    if OUT_PATH.exists():
        print("  " + OUT_PATH.read_text().replace("\n", "\n  "))
