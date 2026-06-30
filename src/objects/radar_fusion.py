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


_last_nonempty: list = []       # 마지막으로 검출 > 0 이었던 targets
_last_nonempty_at: float = 0.0  # 그 wall-clock 시각
CFAR_FALLBACK_TTL = float("inf")  # 정적물체 도플러 억압 대응: 마지막 검출 무기한 유지


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
        # 타임스탬프 동기 실패도 폴백 사용 (동기 실패 ≠ 레이더 없음)
        if best_ts > 0 and abs(best_ts - ts_ms) > tol_ms:
            return _last_nonempty

    targets = frame.get("targets", [])
    if targets:
        _last_nonempty = targets
        _last_nonempty_at = _time.time()
    # 현재 프레임 비어 있으면 항상 폴백 (CFAR 0 / 정적물체 도플러 억압 모두 포함)
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


def _split_by_gaps(pts, n):
    """거리 오름차순 점 리스트를 거리 최대 갭 n-1 곳에서 n개 클러스터로 분할.

    겹친 n개 물체에 점을 나눠 줄 때 사용 — 자연스러운 거리 간격(앞물체/뒷물체
    사이 빈 공간)에서 끊는다. 점 수 < n 이면 가능한 만큼만(뒤 클러스터는 빔)."""
    if n <= 1 or len(pts) <= 1:
        return [pts]
    gaps = sorted(((pts[i+1]["range_m"] - pts[i]["range_m"], i)
                   for i in range(len(pts) - 1)), reverse=True)
    cuts = sorted(i for _, i in gaps[:n-1])
    clusters, start = [], 0
    for c in cuts:
        clusters.append(pts[start:c+1])
        start = c + 1
    clusters.append(pts[start:])
    return clusters


def _snr_cluster(pts, gap_m=0.7):
    """객체 점들을 거리 갭(>gap_m)으로 클러스터링 → SNR 합 최대 클러스터만 반환.

    거리 게이트가 없으면 한 물체 방위각에 앞(물체)+뒤(배경) 점이 섞여 median 이
    출렁인다. 실제 물체는 강반사(SNR 큼)라, SNR 합이 가장 큰 거리 클러스터를
    물체로 보고 나머지(약한 배경 누수)를 버린다. DA3 절대값은 안 쓴다."""
    if len(pts) <= 1:
        return pts
    ps = sorted(pts, key=lambda t: t["range_m"])
    clusters = [[ps[0]]]
    for t in ps[1:]:
        if t["range_m"] - clusters[-1][-1]["range_m"] > gap_m:
            clusters.append([t])
        else:
            clusters[-1].append(t)
    return max(clusters, key=lambda c: sum(t.get("snr", 0.0) for t in c))


def match_all(targets, objs, cam):
    """여러 객체 일괄 매칭 — 거리 게이트 없음 + 겹친 물체 DA3 깊이순 분리.

    objs: [{"tid", "bbox", "mask"(optional), "da3"(optional)}].
    반환 {tid: matchdict | None}.

    1) 각 레이더 점을 방위각으로 매칭되는 객체(들)에 배정
    2) 1개 객체만 매칭 → 그 객체 점
    3) 여러 객체 겹침(같은 az) → 그 점들을 거리 클러스터로 나눠, 객체를 DA3
       오름차순(앞→뒤)으로 정렬해 가까운 클러스터→앞 객체로 배정. DA3 절대값은
       안 쓰고 '앞/뒤 순서'만 쓴다 — DA3 스케일이 틀려도 분리는 정확하다.
    4) 객체별 배정 점 median → 거리
    """
    profs = [_azimuth_profile(o["bbox"], o.get("mask"), cam) for o in objs]
    obj_pts = {i: [] for i in range(len(objs))}
    overlap = {}   # frozenset(obj idx) -> [point]

    for t in targets:
        sx = radar_to_x(t["range_m"], t["azimuth_deg"], cam)
        if sx is None:
            continue
        hit = [i for i, prof in enumerate(profs) if _in_profile(sx, prof)]
        if len(hit) == 1:
            obj_pts[hit[0]].append(t)
        elif len(hit) > 1:
            overlap.setdefault(frozenset(hit), []).append(t)

    # 겹침 그룹: 거리 클러스터 ↔ DA3 깊이 순서 배정
    for grp, pts in overlap.items():
        grp_objs = sorted(grp, key=lambda i: (objs[i].get("da3")
                                              if objs[i].get("da3") is not None else 1e9))
        pts_sorted = sorted(pts, key=lambda t: t["range_m"])
        clusters = _split_by_gaps(pts_sorted, len(grp_objs))
        for ci, cl in enumerate(clusters):
            obj_pts[grp_objs[min(ci, len(grp_objs) - 1)]].extend(cl)

    # 객체별: 배경 누수 제거 — SNR 합 최대 거리 클러스터만 남겨 median
    return {objs[i]["tid"]: _aggregate(_snr_cluster(obj_pts[i]))
            for i in range(len(objs))}


def match_targets_to_objects(targets, objects, cam):
    """여러 객체 일괄 매칭. objects: [{name, bbox:[x1,y1,x2,y2]}] → {name: matchdict}."""
    out = {}
    for obj in objects:
        m = match_one(targets, obj["bbox"], cam)
        if m:
            out[obj["name"]] = m
    return out
