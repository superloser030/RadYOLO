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
from pathlib import Path

NEGATE_AZIMUTH = False   # 좌우 반대로 매칭되면 True 로

# ── 외부 캘리브레이션 (src/utils/radar_cam_calib.py, `main_r.py --calib` 산출) ──
# 카메라↔레이더 상대 자세. yaw(좌우 방향 오프셋)+baseline(tx,tz 위치차)를
# 보정한다. 파일 없으면 0(=기존 "완전 동일 위치/방향" 가정).
#   yaw_offset_deg: 레이더 az 에 더할 좌우 방향 보정(도)
#   tx, tz        : 레이더 기준 카메라 위치 오프셋(m) — baseline
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
    xr = range_m * math.sin(az_r)        # 레이더 좌표 (수평면, y=0)
    zr = range_m * math.cos(az_r)
    yaw = math.radians(_calib["yaw_offset_deg"])
    xc = math.cos(yaw) * xr + math.sin(yaw) * zr + _calib["tx"]   # 카메라 좌표
    zc = -math.sin(yaw) * xr + math.cos(yaw) * zr + _calib["tz"]
    if zc <= 1e-6:
        return None
    return cam["cx"] + cam["fx"] * xc / zc


_last_targets = []   # 읽기 실패(쓰는 중/잠김) 시 직전 값 유지


def load_latest_targets(path, ts_ms=None, tol_ms=3000):
    """targets.json 에서 ts 에 가장 가까운 프레임의 targets 반환.

    ts_ms=None → 최신 프레임. ts 차이가 tol_ms 초과면 [] (동기 실패).
    파일이 없거나 쓰는 중(파싱 실패)이면 직전 값 유지 (깜빡임 방지).
    targets.json: [{frame_idx, ts_ms, targets:[{range_m, velocity_mps, azimuth_deg}]}]
    """
    global _last_targets
    p = Path(path)
    if not p.exists():
        return _last_targets
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError):
        return _last_targets   # movefile 직전/쓰는 중 → 직전 유지
    if not data:
        return _last_targets
    if ts_ms is None:
        _last_targets = data[-1].get("targets", [])
        return _last_targets
    best = min(data, key=lambda f: abs(f.get("ts_ms", 0) - ts_ms))
    if abs(best.get("ts_ms", 0) - ts_ms) > tol_ms:
        return []   # 동기 실패(오래된 데이터)는 빈 값 — 직전 유지 안 함
    _last_targets = best.get("targets", [])
    return _last_targets


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


def match_one(targets, bbox, cam, bbox_dist=None, last_range=None,
              gate_da3=1.2, gate_track=0.5, mask=None):
    """bbox 안 레이더 검출점으로 거리 추정 (DA3 초기화 + 연속성 추적 게이트).

    mask(fh×fw, >0.5=물체)를 주면 bbox 사각형 대신 마스크 가로투영(그 x열에
    물체 픽셀이 존재)으로 점을 거른다 — 겹친 물체끼리 같은 레이더 점을 공유하는
    누수를 줄인다. 레이더는 y 가 없어 2D 마스크를 가로로 투영해서만 쓸 수 있다.
    (거리 추정 _mask_dist 가 마스크 기준인 것과 매칭 기준을 통일)

    1) bbox 가로 범위(또는 마스크 x열) 안에 투영되는 검출점만 추림 (좌우 좁히기)
    2) 거리(앞뒤) 게이트 — 벽/뒷배경 점 제거:
       - last_range(직전 프레임 확정 거리)가 있으면 그 ±gate_track (좁게, 추적)
         → 움직이는 물체도 직전 거리를 따라가고, 1프레임 변화는 작으니 좁게 잡아도 됨
       - 없으면(첫 프레임) bbox_dist(DA3 대략 거리) ±gate_da3 (넓게, 초기화)
       - 둘 다 없으면 게이트 생략(bbox 안 전체)
    3) 남은 점들의 거리 중앙값 → 노이즈에 강한 안정 거리

    반환 {range_m, azimuth_deg, velocity_mps, n_points} 또는 None.
    """
    x1, _, x2, _ = bbox
    col_has = mw = None
    if mask is not None:
        col_has = mask.max(axis=0) > 0.5     # [fw] 각 x열에 물체 픽셀 존재?
        mw = mask.shape[1]
        cam_w = cam.get("width", 1920)
    inside = []
    for t in targets:
        sx = radar_to_x(t["range_m"], t["azimuth_deg"], cam)
        if sx is None:
            continue
        if col_has is not None:
            col = int(sx * mw / cam_w)        # cam_w(1920) 좌표 → 마스크 열
            if not (0 <= col < mw) or not col_has[col]:
                continue
        elif not (x1 <= sx <= x2):
            continue
        inside.append(t)
    if not inside:
        return None

    # 거리 게이트: 추적(직전 거리) 우선 → 없으면 DA3 초기화
    if last_range is not None:
        ref, gate = last_range, gate_track
    elif bbox_dist is not None:
        ref, gate = bbox_dist, gate_da3
    else:
        ref, gate = None, None
    if ref is not None:
        gated = [t for t in inside if abs(t["range_m"] - ref) <= gate]
        if gated:
            inside = gated
        elif bbox_dist is not None:
            # 그 거리에 레이더 점 없음(멀어 약한 물체 등) → DA3 거리 신뢰(앞물체 오염 방지)
            az_rep = sum(t["azimuth_deg"] for t in inside) / len(inside)
            return {
                "range_m":      round(float(bbox_dist), 3),
                "azimuth_deg":  round(float(az_rep), 2),
                "velocity_mps": 0.0,
                "n_points":     0,
            }

    # 거리·방위각·속도 각각 중앙값 (점 1개 대표보다 robust).
    # 특히 az 는 빔폭 noise(±15°)가 커서, 점 1개를 쓰면 정지 물체도 프레임마다
    # ±18° 출렁였다. inside 점들의 median 으로 합쳐 noise 를 √n 완화한다.
    med    = statistics.median(t["range_m"] for t in inside)
    med_az = statistics.median(t["azimuth_deg"] for t in inside)
    med_v  = statistics.median(t.get("velocity_mps", 0.0) for t in inside)
    return {
        "range_m":      round(float(med), 3),
        "azimuth_deg":  round(float(med_az), 2),
        "velocity_mps": round(float(med_v), 3),
        "n_points":     len(inside),
    }


def match_targets_to_objects(targets, objects, cam):
    """여러 객체 일괄 매칭. objects: [{name, bbox:[x1,y1,x2,y2]}] → {name: matchdict}."""
    out = {}
    for obj in objects:
        m = match_one(targets, obj["bbox"], cam)
        if m:
            out[obj["name"]] = m
    return out
