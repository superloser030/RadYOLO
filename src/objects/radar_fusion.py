import json
import math
import statistics
import time as _time
from pathlib import Path

# ⚠ 좌우(az 부호)가 뒤집혀 매칭되면 True 로 토글
NEGATE_AZIMUTH = False

_CALIB_PATH = Path(__file__).resolve().parents[2] / "config" / "calib_radar_cam.json"


def _load_calib() -> dict:
    try:
        c = json.loads(_CALIB_PATH.read_text())
        return {"yaw_offset_deg": float(c.get("yaw_offset_deg", 0.0)),
                "tx": float(c.get("tx", 0.0)), "tz": float(c.get("tz", 0.0))}
    except (OSError, ValueError):
        return {"yaw_offset_deg": 0.0, "tx": 0.0, "tz": 0.0}


_calib = _load_calib()


def radar_to_x(range_m: float, azimuth_deg: float, cam: dict):
    az = -azimuth_deg if NEGATE_AZIMUTH else azimuth_deg
    az_r = math.radians(az)
    xr = range_m * math.sin(az_r)
    zr = range_m * math.cos(az_r)
    yaw = math.radians(_calib["yaw_offset_deg"])
    xc = math.cos(yaw) * xr + math.sin(yaw) * zr + _calib["tx"]
    zc = -math.sin(yaw) * xr + math.cos(yaw) * zr + _calib["tz"]
    if zc <= 1e-6:
        return None
    return cam["cx"] + cam["fx"] * xc / zc


_last_nonempty: list = []
_last_nonempty_at: float = 0.0
CFAR_FALLBACK_TTL = float("inf")


def load_latest_targets(path, ts_ms=None, tol_ms=3000):
    global _last_nonempty, _last_nonempty_at
    p = Path(path)
    if not p.exists():
        return _last_nonempty
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError):
        return _last_nonempty
    if not data:
        return _last_nonempty

    if ts_ms is None:
        frame = data[-1]
    else:
        frame = min(data, key=lambda f: abs(f.get("ts_ms", 0) - ts_ms))
        best_ts = frame.get("ts_ms", 0)
        if best_ts > 0 and abs(best_ts - ts_ms) > tol_ms:
            return _last_nonempty

    targets = frame.get("targets", [])
    if targets:
        _last_nonempty = targets
        _last_nonempty_at = _time.time()
    return _last_nonempty


def depth_to_range(d_norm: float, calib: dict):
    rmin = calib.get("range_min_m")
    rmax = calib.get("range_max_m")
    if not rmin or not rmax or rmin <= 0 or rmax <= 0:
        return None
    d = max(0.0, min(1.0, d_norm))
    return rmin * (rmax / rmin) ** (1.0 - d)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _azimuth_profile(bbox, mask, cam):
    if mask is not None:
        return ("mask", mask.max(axis=0) > 0.5, mask.shape[1], cam.get("width", 1920))
    return ("bbox", bbox[0], bbox[2])


def _in_profile(sx, prof):
    if sx is None:
        return False
    if prof[0] == "mask":
        _, col_has, mw, cam_w = prof
        col = int(sx * mw / cam_w)
        return 0 <= col < mw and bool(col_has[col])
    return prof[1] <= sx <= prof[2]


def _aggregate(pts):
    if not pts:
        return None
    med    = statistics.median(t["range_m"] for t in pts)
    med_az = statistics.median(t["azimuth_deg"] for t in pts)
    med_v  = statistics.median(t.get("velocity_mps", 0.0) for t in pts)
    max_snr = max((t.get("snr", 0.0) for t in pts), default=0.0)
    max_pow = max((t.get("power", 0.0) for t in pts), default=0.0)
    return {
        "range_m":      round(float(med), 3),
        "azimuth_deg":  round(float(med_az), 2),
        "velocity_mps": round(float(med_v), 3),
        "n_points":     len(pts),
        "snr":          round(float(max_snr), 2),
        "power":        round(float(max_pow), 1),
    }


def match_one(targets, bbox, cam, mask=None):
    prof = _azimuth_profile(bbox, mask, cam)
    inside = []
    for t in targets:
        sx = radar_to_x(t["range_m"], t["azimuth_deg"], cam)
        if _in_profile(sx, prof):
            inside.append(t)
    return _aggregate(inside)


def match_all(targets, objs, cam, gate_frac=0.5, gate_min=1.2):
    profs = [_azimuth_profile(o["bbox"], o.get("mask"), cam) for o in objs]
    obj_pts = {i: [] for i in range(len(objs))}

    for t in targets:
        sx = radar_to_x(t["range_m"], t["azimuth_deg"], cam)
        if sx is None:
            continue
        cands = []
        for i, prof in enumerate(profs):
            if not _in_profile(sx, prof):
                continue
            exp = objs[i].get("expected")
            if exp is None:
                cands.append((i, 0.0))
            else:
                gate = max(gate_min, exp * gate_frac)
                dist = abs(t["range_m"] - exp)
                if dist <= gate:
                    cands.append((i, dist))
        if cands:
            obj_pts[min(cands, key=lambda c: c[1])[0]].append(t)

    return {objs[i]["tid"]: _aggregate(obj_pts[i]) for i in range(len(objs))}


def match_targets_to_objects(targets, objects, cam):
    out = {}
    for obj in objects:
        m = match_one(targets, obj["bbox"], cam)
        if m:
            out[obj["name"]] = m
    return out
