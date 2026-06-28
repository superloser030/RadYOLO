"""레이더 targets ↔ YOLO 객체 거리 매칭.

레이더와 카메라가 거의 같은 위치/방향이라 가정하고, target 의 azimuth 를
화면 x 픽셀로 투영해 YOLO bbox 가로 범위와 매칭한다.
  화면 x = cx + fx * tan(azimuth)   (레이더는 수평 2D → x 만 결정, y 는 bbox 사용)

fx, cx 는 1920x1080 기준 intrinsic (camera.json). live_update_loop 의 bbox 도
cam_w(=1920) 로 스케일되어 저장되므로 좌표계가 일치한다.

⚠ azimuth 부호: +가 화면 오른쪽이라 가정. 실측 시 좌우가 뒤집히면
  cfar_detect.m 의 azimuth 정의에 맞춰 부호를 반전(NEGATE_AZIMUTH).
"""
import json
import math
from pathlib import Path

NEGATE_AZIMUTH = False   # 좌우 반대로 매칭되면 True 로


def azimuth_to_x(azimuth_deg: float, cam: dict) -> float:
    if NEGATE_AZIMUTH:
        azimuth_deg = -azimuth_deg
    return cam["cx"] + cam["fx"] * math.tan(math.radians(azimuth_deg))


_last_targets = []   # 읽기 실패(쓰는 중/잠김) 시 직전 값 유지


def load_latest_targets(path, ts_ms=None, tol_ms=500):
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


def match_one(targets, bbox, cam):
    """단일 bbox 에 레이더 target 매칭 → {range_m, azimuth_deg, velocity_mps} 또는 None.

    bbox 가로 범위 안에 투영되는 target 중, bbox 중심에 x 가 가장 가까운 것 선택.
    """
    x1, _, x2, _ = bbox
    cx_box = (x1 + x2) / 2
    best, best_dx = None, None
    for t in targets:
        sx = azimuth_to_x(t["azimuth_deg"], cam)
        if x1 <= sx <= x2:
            dx = abs(sx - cx_box)
            if best_dx is None or dx < best_dx:
                best_dx, best = dx, t
    if best is None:
        return None
    return {
        "range_m":      round(float(best["range_m"]), 3),
        "azimuth_deg":  round(float(best["azimuth_deg"]), 2),
        "velocity_mps": round(float(best.get("velocity_mps", 0.0)), 3),
    }


def match_targets_to_objects(targets, objects, cam):
    """여러 객체 일괄 매칭. objects: [{name, bbox:[x1,y1,x2,y2]}] → {name: matchdict}."""
    out = {}
    for obj in objects:
        m = match_one(targets, obj["bbox"], cam)
        if m:
            out[obj["name"]] = m
    return out
