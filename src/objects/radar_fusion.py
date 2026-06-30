"""레이더 targets ↔ YOLO 객체 거리 매칭.

targets 는 radar_live.m(예제 기반 파이프라인)의 trackerJPDA 출력 = 안정된
track centroid 들이다. 즉 raw 검출점이 아니라 DBSCAN+칼만으로 정제된 소수의
물체 단위 추정이라, 매칭이 단순하고 거리도 시간적으로 안정돼 있다.

레이더와 카메라가 거의 같은 위치/방향이라 가정하고, target 의 azimuth 를
화면 x 픽셀로 투영해 YOLO bbox 가로 범위와 매칭한다.
  화면 x = cx + fx * tan(azimuth)   (레이더는 수평 2D → x 만 결정, y 는 bbox 사용)

fx, cx 는 1920x1080 기준 intrinsic (camera.json). live_update_loop 의 bbox 도
cam_w(=1920) 로 스케일되어 저장되므로 좌표계가 일치한다.

⚠ azimuth 부호: +가 화면 오른쪽이라 가정. 실측 시 좌우가 뒤집히면
  radar_live.m 의 az 정의(estimateAzimuth/가상배열)에 맞춰 NEGATE_AZIMUTH 토글.
  (CHECKLIST.md 의 "위험 2 — angle 좌우 부호" 검증 결과로 결정)
"""
import json
import math
import statistics
import time as _time
from pathlib import Path

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
    """레이더 (거리, 방위각) → 화면 x 픽셀. 외부 캘리브(yaw+baseline) 적용.

    레이더 좌표 3D 점을 카메라 좌표로 변환(yaw 회전 + 평행이동) 후 투영한다.
    yaw=0, tx=tz=0 이면 기존 단순식(cx + fx·tan(az))과 동일.
    zc<=0(카메라 뒤)면 None.
    """
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
    """targets.json 에서 ts 에 가장 가까운 프레임의 targets 반환.

    어떤 경로에서든 현재 프레임이 비어 있으면 _last_nonempty 폴백.
    타임스탬프 동기 실패 / 파일 오류 / CFAR 0 모두 폴백 대상.
    targets.json: [{frame_idx, ts_ms, targets:[{range_m, velocity_mps, azimuth_deg}]}]
    """
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
    """보정된 DA3 depth(0~1, 1=가까움) → 실제 거리(m). calib: depth_calib.json.

    depth_calibration.py 의 log 정규화 역산:
      range = range_min * (range_max/range_min) ** (1 - d_norm)
    """
    rmin = calib.get("range_min_m")
    rmax = calib.get("range_max_m")
    if not rmin or not rmax or rmin <= 0 or rmax <= 0:
        return None
    d = max(0.0, min(1.0, d_norm))
    return rmin * (rmax / rmin) ** (1.0 - d)


def iou(a, b):
    """두 bbox [x1,y1,x2,y2] 의 IoU (객체 추적: 직전 프레임 박스 매칭용)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _azimuth_profile(bbox, mask, cam):
    """객체의 방위각(화면 x) 판정 프로파일. mask 우선(가로투영), 없으면 bbox x범위."""
    if mask is not None:
        return ("mask", mask.max(axis=0) > 0.5, mask.shape[1], cam.get("width", 1920))
    return ("bbox", bbox[0], bbox[2])


def _in_profile(sx, prof):
    """화면 x(sx)가 객체 방위각 프로파일 안인가."""
    if sx is None:
        return False
    if prof[0] == "mask":
        _, col_has, mw, cam_w = prof
        col = int(sx * mw / cam_w)
        return 0 <= col < mw and bool(col_has[col])
    return prof[1] <= sx <= prof[2]


def _aggregate(pts):
    """레이더 점 리스트 → {range_m, az, v, n, snr, power} 중앙값 집계. 비면 None.

    az 는 빔폭 noise(±15°)가 커 점 1개를 쓰면 정지 물체도 프레임마다 출렁이므로
    median 으로 합쳐 noise 를 완화한다. 거리 게이트 없음 — 방위각으로 모인 점 그대로.
    """
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
    """bbox/마스크 방위각 안의 레이더 점 median 거리 (거리 게이트 없음).

    DA3 거리 게이트를 쓰지 않는다 — DA3 절대값은 스케일이 부정확해, 게이트로 쓰면
    멀쩡한 레이더 점(예: 벽 근처 정적물체)을 엉뚱하게 끌어내렸다. 방위각으로 모인
    점의 중앙값을 그대로 거리로 쓴다. 겹친 물체(같은 az) 분리는 match_all 이 담당.
    반환 {range_m, azimuth_deg, velocity_mps, n_points, snr, power} 또는 None.
    """
    prof = _azimuth_profile(bbox, mask, cam)
    inside = []
    for t in targets:
        sx = radar_to_x(t["range_m"], t["azimuth_deg"], cam)
        if _in_profile(sx, prof):
            inside.append(t)
    return _aggregate(inside)


def match_all(targets, objs, cam, gate_frac=0.5, gate_min=1.2):
    """여러 객체 일괄 매칭 — expected(DA3×비율) 기준 거리 게이트로 클러스터 구분.

    거리 게이트의 기준(ref)을 DA3 절대값(부정확) 대신 expected=DA3×비율(metric)로
    쓴다. 그래서 게이트가 클러스터를 정확히 구분한다:
      - 단일 물체: expected±gate 안의 점만 → 벽/배경 누수 제거, 실측 그대로
      - 겹친 물체: 한 점이 여러 객체 방위각에 들어도 각자 expected 게이트로 갈림.
        약반사로 자기 거리에 점 없는 물체는 먼 점이 게이트 밖이라 안 받음 → miss
      - 어느 게이트에도 안 들면 그 점은 배경/노이즈로 버림

    gate = max(gate_min, expected*gate_frac). 한 점이 여러 게이트 통과 시 expected
    최근접 객체 1개에만. objs: [{"tid","bbox","mask"?,"da3"?,"expected"?}].
    반환 {tid: matchdict | None}.
    """
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
    """여러 객체 일괄 매칭. objects: [{name, bbox:[x1,y1,x2,y2]}] → {name: matchdict}."""
    out = {}
    for obj in objects:
        m = match_one(targets, obj["bbox"], cam)
        if m:
            out[obj["name"]] = m
    return out
