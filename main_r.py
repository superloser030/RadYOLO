import os
import sys
import json
import time
import shutil
import threading
import subprocess
import webbrowser
import http.server
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
OBJECTS_DIR  = PROJECT_ROOT / "data" / "objects"
DB_DIR       = PROJECT_ROOT / "db"

from src.utils.archive import (
    archive_data,
    record_session_start,
    record_session_end,
    start_heartbeat_thread,
)
from src.utils.config import load_camera, export_camera_json
from src.utils.gpu_scheduler import GPUManager
from src.transmission import receiver
from src.transmission.receiver  import radar_receive, webcam_receive, meta_receive
from src.background.bg_select   import select_background
from src.background.upscale     import upscale_image
from src.background.depth            import generate_depth
from src.background.depth_calibration import calibrate_depth
from src.background.yolo_mask   import generate_mask    
from src.objects.obj_crop       import crop_objects
from src.objects.trellis_gen    import generate_3d


def estimate_object_poses():
    from src.objects.pose_estimator import prepare_templates, estimate_pose

    cam_cfg    = load_camera()
    camera_k   = [cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]]
    bg_image   = str(PROJECT_ROOT / "data" / "scene" / "background.jpg")
    objects_dir = PROJECT_ROOT / "data" / "objects"
    manifest   = []

    if not objects_dir.exists():
        return

    for obj_dir in sorted(objects_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        meta_path = obj_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        entry = {
            "dir":       obj_dir.name,
            "class":     meta.get("class", ""),
            "has_model": False,
            "has_pose":  False,
        }

        mesh_path = obj_dir / "model_trellis.glb"
        if not mesh_path.exists():
            manifest.append(entry)
            continue
        entry["has_model"] = True

        template_dir = str(obj_dir / "templates")
        print(f"\n=== Pose: {obj_dir.name} ===")
        prepare_templates(str(mesh_path), template_dir)

        bbox = meta.get("bbox")
        if bbox is None:
            manifest.append(entry)
            continue

        pose = estimate_pose(
            image_path=bg_image,
            bbox=tuple(bbox),
            template_dir=template_dir,
            camera_k=camera_k,
        )
        if pose:
            (obj_dir / "pose.json").write_text(json.dumps(pose, indent=2))
            entry["has_pose"] = True
            print(f"[Pose] {obj_dir.name}: score={pose.get('score', 0):.3f}  t={[round(v,3) for v in pose['t']]}")
        else:
            print(f"[Pose] {obj_dir.name}: 추정 실패")
        manifest.append(entry)

    (PROJECT_ROOT / "data" / "objects" / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"[Pose] manifest.json 저장 ({len(manifest)}개 객체)")


def _appearance(frame, bbox, m_frame):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
    if crop.size == 0:
        return None
    mask8 = None
    if m_frame is not None:
        mc = m_frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        mask8 = (mc > 0.5).astype(np.uint8) * 255
        if mask8.shape[:2] != crop.shape[:2]:
            mask8 = cv2.resize(mask8, (crop.shape[1], crop.shape[0]))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask8, [16, 4], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def _try_reid(graveyard, cls, bbox, appearance, now,
              graveyard_ttl=120.0, hist_th=0.60):
    bw  = max(1.0, bbox[2] - bbox[0])
    cx  = (bbox[0] + bbox[2]) / 2
    cy  = (bbox[1] + bbox[3]) / 2
    best_tid, best_score, best_entry = None, -1.0, None
    for tid, g in graveyard.items():
        if g["cls"] != cls or now - g["evicted_at"] > graveyard_ttl:
            continue
        gcx = (g["bbox"][0] + g["bbox"][2]) / 2
        gcy = (g["bbox"][1] + g["bbox"][3]) / 2
        dist_norm = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5 / bw
        if dist_norm > 1.5:
            continue
        pos_score  = max(0.0, 1.0 - dist_norm / 1.5)
        hist_score = 0.5
        if appearance is not None and g.get("appearance") is not None:
            hist_score = max(0.0, float(cv2.compareHist(
                appearance, g["appearance"], cv2.HISTCMP_CORREL)))
        if hist_score < hist_th:
            continue
        score = pos_score * 0.4 + hist_score * 0.6
        if score > best_score:
            best_score, best_tid, best_entry = score, tid, g
    return (best_tid, best_entry) if best_tid is not None else None


def _update_manifest():
    if not OBJECTS_DIR.exists():
        return
    manifest = []
    for obj_dir in sorted(OBJECTS_DIR.iterdir()):
        if not obj_dir.is_dir():
            continue
        meta_path = obj_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:
            meta = {}
        manifest.append({
            "dir": obj_dir.name,
            "class": meta.get("class", ""),
            "has_model": (obj_dir / "model_trellis.glb").exists(),
            "has_pose":  (obj_dir / "pose.json").exists(),
        })
    (OBJECTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _save_to_db(od: Path, cls_name: str, appearance):
    import uuid
    if appearance is None or not (od / "model_trellis.glb").exists():
        return
    DB_DIR.mkdir(exist_ok=True)
    db_entry = DB_DIR / f"{cls_name}_{uuid.uuid4().hex[:8]}"
    db_entry.mkdir(exist_ok=True)
    shutil.copy2(str(od / "model_trellis.glb"), str(db_entry / "model_trellis.glb"))
    tmpl_src = od / "templates"
    if tmpl_src.exists():
        tmpl_dst = db_entry / "templates"
        if tmpl_dst.exists():
            shutil.rmtree(str(tmpl_dst))
        shutil.copytree(str(tmpl_src), str(tmpl_dst))
    np.save(str(db_entry / "appearance.npy"), appearance)
    (db_entry / "meta.json").write_text(json.dumps(
        {"class": cls_name, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"[DB] 저장: {db_entry.name}")


def _try_db_match(cls: str, appearance, threshold=0.65):
    if appearance is None or not DB_DIR.exists():
        return None
    best_path, best_score = None, -1.0
    for entry in DB_DIR.iterdir():
        if not entry.is_dir():
            continue
        meta_p = entry / "meta.json"
        app_p  = entry / "appearance.npy"
        if not meta_p.exists() or not app_p.exists():
            continue
        try:
            if json.loads(meta_p.read_text()).get("class") != cls:
                continue
            db_app = np.load(str(app_p))
            score  = float(cv2.compareHist(appearance, db_app, cv2.HISTCMP_CORREL))
        except Exception:
            continue
        if score > best_score:
            best_score, best_path = score, entry
    if best_score >= threshold:
        print(f"[DB] 매칭: {cls} ← {best_path.name} (score={best_score:.3f})")
        return best_path
    return None


def _restore_from_db(db_entry: Path, cls_name: str, tid: int) -> Path:
    od = OBJECTS_DIR / f"{cls_name}_{tid}"
    od.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(db_entry / "model_trellis.glb"), str(od / "model_trellis.glb"))
    tmpl_src = db_entry / "templates"
    if tmpl_src.exists():
        tmpl_dst = od / "templates"
        if tmpl_dst.exists():
            shutil.rmtree(str(tmpl_dst))
        shutil.copytree(str(tmpl_src), str(tmpl_dst))
    return od


def _run_pose_bg(obj_dir, frame_path, bbox, cam_cfg):
    from src.objects.pose_estimator import estimate_pose
    template_dir = str(obj_dir / "templates")
    if not (obj_dir / "templates" / "meta.json").exists():
        return
    camera_k = [cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]]
    pose = estimate_pose(
        image_path=frame_path,
        bbox=tuple(bbox),
        template_dir=template_dir,
        camera_k=camera_k,
    )
    if pose:
        (obj_dir / "pose.json").write_text(json.dumps(pose, indent=2))
        _update_manifest()
        print(f"[Live-R] {obj_dir.name}: score={pose.get('score',0):.3f}  t={[round(v,3) for v in pose['t']]}")


def live_update_loop(cam_cfg, enable_pose=True):
    import time, cv2
    import numpy as np
    from ultralytics import YOLO
    from src.objects.radar_fusion import load_latest_targets, match_all, depth_to_range, iou
    from src.objects.obj_crop import load_sam2, crop_one, crop_one_extra, SKIP_CLASSES
    from src.objects.trellis_gen import generate_3d
    from src.background.depth import generate_depth
    from src.background.upscale import upscale_image
    from src.background.depth_calibration import apply_calib
    from src.utils.config import load_receiver

    YOLO_MODEL   = str(PROJECT_ROOT / "models" / "yolo11x-seg.pt")
    TARGETS_PATH = PROJECT_ROOT / "data" / "radar" / "targets.json"
    SCENE        = PROJECT_ROOT / "data" / "scene"
    DEPTH_PATH   = SCENE / "depth.png"
    CALIB_PATH   = SCENE / "depth_calib.json"
    METRIC_PATH  = SCENE / "depth_metric.npy"
    OBJECTS_DIR  = PROJECT_ROOT / "data" / "objects"
    DA3_INPUT    = SCENE / "da3_frame.jpg"

    ycfg        = load_receiver().get("yolo", {})
    obj_conf    = float(ycfg.get("obj_conf",   0.72))
    model_conf  = float(ycfg.get("model_conf", 0.85))
    live_conf   = float(ycfg.get("live_conf",  0.5))
    detect_conf = min(live_conf, obj_conf)

    _depth = [None]; _depth_calib = [None]; _depth_metric = [None]
    def _reload_depth():
        try:
            if METRIC_PATH.exists():
                _depth_metric[0] = np.load(str(METRIC_PATH))
            if DEPTH_PATH.exists() and CALIB_PATH.exists():
                d = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
                if d is not None and d.ndim != 2:
                    d = d[:, :, 0]
                _depth[0] = d
                _depth_calib[0] = json.loads(CALIB_PATH.read_text())
        except Exception:
            pass
    _reload_depth()
    if _depth_metric[0] is not None:
        print("[Fusion] DA3 metric(미터) 직접 사용 — 보정 우회")
    elif _depth[0] is not None:
        print("[Fusion] DA3 depth 게이트 활성")

    yolo  = YOLO(YOLO_MODEL)
    cam_w = cam_cfg.get("width",  1920)
    cam_h = cam_cfg.get("height", 1080)
    _sam2 = [None]
    def _get_sam2():
        if _sam2[0] is None:
            print("[State] SAM2 lazy 로드...")
            _sam2[0] = load_sam2()
        return _sam2[0]
    _sam2_lock = threading.Lock()
    _da3_lock  = threading.Lock()
    _da3_last  = [0.0]

    ERODE_FRAC = 0.15

    def _erode_vals(arr, m_frame):
        h, w = arr.shape[:2]
        mm = cv2.resize(m_frame, (w, h)) if (h, w) != m_frame.shape else m_frame
        mmb = (mm > 0.5).astype(np.uint8)
        ys, xs = np.where(mmb > 0)
        if len(xs) == 0:
            return arr[mmb > 0]
        side = min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        er_px = max(1, int(side * ERODE_FRAC))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er_px + 1, 2 * er_px + 1))
        er = cv2.erode(mmb, k)
        sel = er if er.any() else mmb
        return arr[sel > 0]

    def _mask_dist(m_frame, cx, cy):
        met = _depth_metric[0]
        if met is not None:
            dh, dw = met.shape[:2]
            if m_frame is not None:
                vals = _erode_vals(met, m_frame)
                if len(vals) > 0:
                    return float(np.median(vals))
            bx = max(0, min(dw - 1, int(cx * dw / cam_w)))
            by = max(0, min(dh - 1, int(cy * dh / cam_h)))
            return float(met[by, bx])
        dep, cal = _depth[0], _depth_calib[0]
        if dep is None or cal is None:
            return None
        dh, dw = dep.shape[:2]
        if m_frame is not None:
            vals = _erode_vals(dep, m_frame)
            if len(vals) > 0:
                return depth_to_range(float(np.median(vals)) / 255.0, cal)
        bx = max(0, min(dw - 1, int(cx * dw / cam_w)))
        by = max(0, min(dh - 1, int(cy * dh / cam_h)))
        return depth_to_range(dep[by, bx] / 255.0, cal)

    def _da3_rerun(frame_raw):
        # live 모드: 현재 프레임 → depth_metric.npy 만 갱신(_mask_dist 객체거리).
        # 뷰어 배경(depth_bg.png/calib)은 안 건드림 — 배경/현재프레임 depth 충돌 방지.
        try:
            raw = SCENE / "da3_raw.jpg"
            cv2.imwrite(str(raw), frame_raw)
            generate_depth(input_path=str(raw), mode="live")
            _reload_depth()
            print("[State] DA3 재추론(live) → metric 갱신")
        except Exception as e:
            print(f"[State] DA3 재추론 실패: {e}")

    def _da3_rerun_shared(frame_big):
        if not _da3_lock.acquire(blocking=False):
            return
        try:
            if time.time() - _da3_last[0] < 7.0:
                return
            _da3_rerun(frame_big)
            _da3_last[0] = time.time()
        finally:
            _da3_lock.release()

    registry  = {}
    graveyard = {}
    _busy     = {}
    _next_tid = [0]
    MISS_MAX, IOU_TH = 10, 0.15
    GRAVEYARD_TTL  = 120.0
    MIN_VIEWS      = 3
    COLLECT_TIMEOUT = 30.0
    MOVE_TH        = 0.30
    _last_fusion = 0.0
    _da3_scale   = [3.0]
    _prev_da3_ts = [0.0]
    SNR_MIN      = 4.0

    while not receiver.shutdown_event.is_set():
        frame = receiver.get_latest_frame()
        if frame is None:
            time.sleep(0.2)
            continue

        fh, fw = frame.shape[:2]
        sx = cam_w / fw
        sy = cam_h / fh
        frame_ts = receiver.get_latest_frame_ts() or None
        targets  = load_latest_targets(TARGETS_PATH, frame_ts)

        results = yolo(frame, verbose=False, conf=detect_conf)
        masks = results[0].masks

        dets = []
        for i in range(len(results[0].boxes)):
            box  = results[0].boxes[i]
            name = yolo.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            mbbox = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
            m_frame = None
            if masks is not None and i < len(masks.data):
                m = masks.data[i].cpu().numpy()
                if m.shape != (fh, fw):
                    m = cv2.resize(m, (fw, fh))
                ys, xs = np.where(m > 0.5)
                if len(xs) > 0:
                    mbbox = [float(xs.min()) * sx, float(ys.min()) * sy,
                             float(xs.max()) * sx, float(ys.max()) * sy]
                    m_frame = m
            dets.append({"name": name, "conf": conf, "bbox": mbbox, "m_frame": m_frame,
                         "mdist": _mask_dist(m_frame, cx, cy),
                         "appearance": _appearance(frame, mbbox, m_frame),
                         "cx": (mbbox[0]+mbbox[2])/2, "cy": (mbbox[1]+mbbox[3])/2, "tid": None})

        matched = set()
        for d in dets:
            bw = max(1.0, d["bbox"][2] - d["bbox"][0])
            best_tid, best_sc = None, IOU_TH
            for tid, r in registry.items():
                if tid in matched or r["cls"] != d["name"]:
                    continue
                sc = iou(d["bbox"], r["bbox"])
                pcx, pcy = (r["bbox"][0]+r["bbox"][2])/2, (r["bbox"][1]+r["bbox"][3])/2
                if ((d["cx"]-pcx)**2 + (d["cy"]-pcy)**2) ** 0.5 < bw * 0.5:
                    sc = max(sc, 0.2)
                if sc > best_sc:
                    best_sc, best_tid = sc, tid
            if best_tid is not None:
                d["tid"] = best_tid; matched.add(best_tid)
                r = registry[best_tid]
                r["bbox"] = d["bbox"]; r["conf"] = max(r["conf"], d["conf"]); r["miss"] = 0
                r["appearance"] = d["appearance"]
            elif d["conf"] >= obj_conf:
                reid = _try_reid(graveyard, d["name"], d["bbox"], d["appearance"],
                                 time.time(), GRAVEYARD_TTL)
                if reid:
                    rtid, rentry = reid
                    del graveyard[rtid]
                    st = rentry.get("state", "NEW")
                    if st == "MODELING":
                        st = "NEW"
                    d["tid"] = rtid; matched.add(rtid)
                    registry[rtid] = {**rentry, "bbox": d["bbox"], "conf": d["conf"],
                                      "miss": 0, "state": st, "appearance": d["appearance"]}
                    registry[rtid].pop("evicted_at", None)
                    print(f"[ReID] {d['name']}_{rtid} graveyard→재등록 "
                          f"(state={st}, glb={'있음' if rentry.get('glb') else '없음'})")
                else:
                    tid = _next_tid[0]; _next_tid[0] += 1
                    d["tid"] = tid; matched.add(tid)
                    db_match = _try_db_match(d["name"], d["appearance"])
                    if db_match:
                        od = _restore_from_db(db_match, d["name"], tid)
                        registry[tid] = {"cls": d["name"], "bbox": d["bbox"], "conf": d["conf"],
                                         "miss": 0, "state": "READY",
                                         "glb": str(od / "model_trellis.glb"),
                                         "appearance": d["appearance"]}
                        _update_manifest()
                        print(f"[DB-ReID] {d['name']}_{tid} DB 복원 → READY 즉시 진입")
                    else:
                        registry[tid] = {"cls": d["name"], "bbox": d["bbox"], "conf": d["conf"],
                                         "miss": 0, "state": "NEW", "glb": None,
                                         "appearance": d["appearance"]}

        for tid in list(registry):
            if tid not in matched:
                registry[tid]["miss"] += 1
                if registry[tid]["miss"] > MISS_MAX:
                    entry = registry.pop(tid)
                    entry["evicted_at"] = time.time()
                    graveyard[tid] = entry
                    _busy.pop(tid, None); _busy.pop(f"pose{tid}", None)
                    print(f"[State] {entry['cls']}_{tid} → graveyard")

        det_by_tid   = {d["tid"]: d for d in dets if d["tid"] is not None}

        if time.time() - _da3_last[0] >= 7.0:
            _fb = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
            threading.Thread(target=_da3_rerun_shared, args=(_fb,), daemon=True).start()
        if _da3_last[0] != _prev_da3_ts[0]:
            _prev_da3_ts[0] = _da3_last[0]
            for _tid in registry:
                registry[_tid]["da3_0"] = None

        for tid, d in det_by_tid.items():
            if tid in registry and registry[tid].get("da3_0") is None and d.get("mdist"):
                registry[tid]["da3_0"] = round(float(d["mdist"]), 3)
        match_objs   = []
        for tid, d in det_by_tid.items():
            if tid not in registry:
                continue
            da3 = registry[tid].get("da3_0")
            exp = da3 * (registry[tid].get("ratio") or _da3_scale[0]) if da3 else None
            match_objs.append({"tid": tid, "bbox": registry[tid]["bbox"],
                               "mask": d["m_frame"], "da3": da3, "expected": exp})
        radar_by_tid = match_all(targets, match_objs, cam_cfg)

        for t in radar_by_tid:
            rd = radar_by_tid[t]
            da3_0 = registry[t].get("da3_0") if t in registry else None
            if rd and rd["n_points"] > 0 and rd.get("snr", 0) >= SNR_MIN and da3_0 and da3_0 > 0.05:
                ratio = rd["range_m"] / da3_0
                old = registry[t].get("ratio")
                registry[t]["ratio"] = 0.7 * old + 0.3 * ratio if old else ratio
        all_ratios = [registry[t]["ratio"] for t in registry if registry[t].get("ratio")]
        if all_ratios:
            _da3_scale[0] = float(np.median(all_ratios))

        fusion_lines = []; overlay = []
        for tid, r in list(registry.items()):
            d = det_by_tid.get(tid)
            if d is None:
                continue
            inst = f"{r['cls']}_{tid}"
            radar = radar_by_tid.get(tid)
            da3 = r.get("da3_0")
            o = {"name": inst, "cls": r["cls"], "conf": round(r["conf"], 2),
                 "bbox": [round(v) for v in r["bbox"]], "state": r["state"]}
            if da3 is not None:
                o["da3_m"] = round(float(da3), 2)
            da3_str = f"DA3 {da3:5.2f}m" if da3 is not None else "DA3  --  "
            if radar and radar.get("snr", 0) >= SNR_MIN:
                # v 적응 거리 EMA: 정지(v≈0)면 강한 smooth(R 널뛰기 흡수), 동적이면 추종.
                # v 는 range 와 같은 검출점 도플러 — 정지물체는 v≈0 가 robust 해 R 안정화에 신뢰.
                R_meas = radar["range_m"]
                alpha  = 0.5 if abs(radar["velocity_mps"]) >= 0.1 else 0.15
                prev   = r.get("r_smooth")
                r_sm   = prev + alpha * (R_meas - prev) if prev is not None else R_meas
                r["r_smooth"] = r_sm
                o["range_m"] = round(r_sm, 2); o["az"] = radar["azimuth_deg"]
                o["v"] = radar["velocity_mps"];  o["n"]  = radar.get("n_points")
                o["snr"] = radar.get("snr")
                fusion_lines.append(f"[Fusion] {inst:14} R {r_sm:5.2f}m(meas {R_meas:5.2f}) | {da3_str} | "
                                    f"az {radar['azimuth_deg']:+6.1f} | n {radar.get('n_points')} | "
                                    f"snr {radar.get('snr', 0):5.1f} | v {radar['velocity_mps']:+5.2f} | {r['state']}")
            elif da3 is not None and (r.get("ratio") or _da3_scale[0] > 0):
                ratio = r.get("ratio") or _da3_scale[0]
                est = da3 * ratio
                o["range_m"] = round(est, 2); o["n"] = 0; o["est"] = True
                snr_note = f" (radar snr {radar['snr']:.1f}<{SNR_MIN:.0f})" if radar else ""
                fusion_lines.append(f"[Fusion] {inst:14} R~{est:5.2f}m | {da3_str} | "
                                    f"est(DA3×{ratio:.2f}){snr_note} | {r['state']}")
            else:
                fusion_lines.append(f"[Fusion] {inst:14} miss      | {da3_str} | {r['state']}")
            overlay.append(o)

            if not enable_pose:
                continue
            if (r["state"] == "NEW" and r["conf"] >= model_conf
                    and r["cls"] not in SKIP_CLASSES):
                r["state"] = "COLLECTING"
                r["views"] = []; r["collect_start"] = time.time()
                r["last_view_cx"] = r["last_view_cy"] = None
                od = OBJECTS_DIR / f"{r['cls']}_{tid}"
                (od / "views").mkdir(parents=True, exist_ok=True)
                print(f"[State] {r['cls']}_{tid} NEW → COLLECTING")

            elif r["state"] == "COLLECTING" and d is not None and not _busy.get(tid):
                bw = max(1.0, r["bbox"][2] - r["bbox"][0])
                lcx, lcy = r.get("last_view_cx"), r.get("last_view_cy")
                moved = (lcx is None or
                         ((d["cx"] - lcx)**2 + (d["cy"] - lcy)**2)**0.5 > bw * MOVE_TH)
                if moved and d["conf"] >= model_conf and d["m_frame"] is not None:
                    od = OBJECTS_DIR / f"{r['cls']}_{tid}"
                    vp = od / "views" / f"view_{len(r['views']):03d}.jpg"
                    cv2.imwrite(str(vp), cv2.resize(frame, (cam_w, cam_h)))
                    r["views"].append({"cx": d["cx"], "cy": d["cy"],
                                       "bbox": list(r["bbox"]), "path": str(vp)})
                    r["last_view_cx"] = d["cx"]; r["last_view_cy"] = d["cy"]
                    print(f"[Collect] {r['cls']}_{tid} view #{len(r['views'])}")
                elapsed = time.time() - r.get("collect_start", time.time())
                if elapsed > COLLECT_TIMEOUT and not r["views"] and d["m_frame"] is not None:
                    od = OBJECTS_DIR / f"{r['cls']}_{tid}"
                    vp = od / "views" / "view_000.jpg"
                    cv2.imwrite(str(vp), cv2.resize(frame, (cam_w, cam_h)))
                    r["views"].append({"cx": d["cx"], "cy": d["cy"],
                                       "bbox": list(r["bbox"]), "path": str(vp)})
                if r["views"] and (len(r["views"]) >= MIN_VIEWS or elapsed > COLLECT_TIMEOUT):
                    _busy[tid] = True; r["state"] = "MODELING"
                    raw = frame.copy(); views = list(r["views"]); cls_name = r["cls"]
                    def _model_bg(tid=tid, raw=raw, views=views, cls_name=cls_name):
                        try:
                            _da3_rerun_shared(raw)
                            od = OBJECTS_DIR / f"{cls_name}_{tid}"
                            extra_cutouts = []
                            with _sam2_lock:
                                sam = _get_sam2()
                                f0 = cv2.imread(views[0]["path"])
                                sam.set_image(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
                                crop_one(f0, views[0]["bbox"], cls_name, tid, sam)
                                for i, v in enumerate(views[1:], 1):
                                    vf = cv2.imread(v["path"])
                                    if vf is None:
                                        continue
                                    sam.set_image(cv2.cvtColor(vf, cv2.COLOR_BGR2RGB))
                                    ct = crop_one_extra(vf, v["bbox"], sam)
                                    if ct is not None:
                                        cp = od / "views" / f"view_{i:03d}_cutout.jpg"
                                        cv2.imwrite(str(cp), ct, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                        extra_cutouts.append(cp)
                            generate_3d(od, extra_cutouts=extra_cutouts or None)
                            if tid in registry:
                                registry[tid]["state"] = "READY"
                                registry[tid]["glb"]   = str(od / "model_trellis.glb")
                                _update_manifest()
                                _save_to_db(od, cls_name, registry[tid].get("appearance"))
                            print(f"[State] {cls_name}_{tid} → READY "
                                  f"({'멀티뷰 ' + str(1+len(extra_cutouts)) + '장' if extra_cutouts else '단일뷰'})")
                        except Exception as e:
                            print(f"[State] {cls_name}_{tid} MODELING 실패: {e}")
                            if tid in registry:
                                registry[tid]["state"] = "NEW"
                        finally:
                            _busy[tid] = False
                    threading.Thread(target=_model_bg, daemon=True).start()

            elif (r["state"] == "READY"
                  and not _busy.get(f"pose{tid}")
                  and time.time() >= r.get("pose_retry_at", 0)):
                od = OBJECTS_DIR / f"{r['cls']}_{tid}"
                if (od / "model_trellis.glb").exists():
                    _busy[f"pose{tid}"] = True
                    fp = str(od / "live_frame.jpg")
                    cv2.imwrite(fp, cv2.resize(frame, (cam_w, cam_h)))
                    bb = list(r["bbox"])
                    def _pose_bg(od=od, fp=fp, bb=bb, tid=tid):
                        try:
                            from src.objects.pose_estimator import prepare_templates
                            prepare_templates(str(od / "model_trellis.glb"), str(od / "templates"))
                            _run_pose_bg(od, fp, bb, cam_cfg)
                        except Exception as e:
                            print(f"[Pose] {tid} 실패, 30초 쿨다운: {e}")
                            if tid in registry:
                                registry[tid]["pose_retry_at"] = time.time() + 30
                        finally:
                            _busy[f"pose{tid}"] = False
                    threading.Thread(target=_pose_bg, daemon=True).start()

        (SCENE / "live_overlay.json").write_text(json.dumps(overlay))

        now = time.time()
        if now - _last_fusion >= 1.0:
            print(f"--- radar targets: {len(targets)} | registry {len(registry)} | graveyard {len(graveyard)} ---")
            for ln in fusion_lines:
                print(ln)
            _last_fusion = now

        for _tid in [k for k, v in graveyard.items() if now - v["evicted_at"] > GRAVEYARD_TTL]:
            del graveyard[_tid]

        time.sleep(0.3)


def _start_iperf_server():
    try:
        proc = subprocess.Popen(
            ["iperf3", "-s"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[BW] iperf3 서버 시작 (포트 5201) — 센더 mode 1 측정 대기")
        return proc
    except FileNotFoundError:
        print("[BW] iperf3 미설치 — 센더 mode 1 불가 (센더는 mode 0 사용 권장)")
        return None


def _start_matlab_cfar():
    import datetime as _dt
    matlab_dir = PROJECT_ROOT / "matlab"
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"radar_live_{_dt.datetime.now():%H%M%S}.log"
    try:
        log = open(log_path, "w")
        proc = subprocess.Popen(
            ["matlab", "-batch", "radar_live"],
            cwd=str(matlab_dir),
            stdout=log, stderr=subprocess.STDOUT)
        print(f"[Radar] MATLAB radar_live 자동 시작 (로그: logs/{log_path.name})")
        return proc
    except FileNotFoundError:
        print("[Radar] 'matlab' 명령 못 찾음 — radar_live.m 수동 실행 필요")
        return None


def _wait_meta_start_radar(no_radar):
    # 레이더 메타(.mat 설정)는 레이더의 필수 선행 — 받을 때까지 게이트로 대기한다.
    # 늦게 와도(sender 늦게 켜짐 등) 오는 순간 radar_live 시작. --no-radar 면 생략.
    if no_radar:
        print("[Init] --no-radar: 레이더 생략 (웹캠만)")
        return None
    print("[Init] 레이더 메타 수신 대기...")
    while not receiver._chirp_ready.wait(timeout=10):
        print("[Init] 메타 미수신 — sender/레이더 켜졌는지 확인. 계속 대기 (Ctrl+C 종료)")
    print("[Init] 메타 확정 — 2초 후 radar_live 시작")
    time.sleep(2)
    return _start_matlab_cfar()


def _verify_intake(secs=3.0):
    import time as _t
    targets_path = PROJECT_ROOT / "data" / "radar" / "targets.json"
    t0 = _t.time(); got_cam = got_radar = False
    while _t.time() - t0 < secs:
        if receiver.get_latest_frame() is not None:
            got_cam = True
        try:
            if targets_path.exists() and json.loads(targets_path.read_text()):
                got_radar = True
        except Exception:
            pass
        if got_cam and got_radar:
            break
        _t.sleep(0.2)
    print(f"[Init] 수신 확인({secs:.0f}s) — 웹캠:{'OK' if got_cam else '없음'} "
          f"레이더:{'OK' if got_radar else '아직(radar_live 준비 중일 수 있음 — 진행)'}")
    if not got_cam:
        print("[Init] ⚠ 웹캠 프레임 미수신 — 센더/네트워크 확인 필요")


class _ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stream":
            self._mjpeg()
        else:
            super().do_GET()

    def _mjpeg(self):
        import time as _t, cv2 as _cv
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
        except Exception:
            return
        while not receiver.shutdown_event.is_set():
            frame = receiver.get_latest_frame()
            if frame is None:
                _t.sleep(0.05); continue
            ok, buf = _cv.imencode(".jpg", frame, [_cv.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(buf.tobytes())
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            _t.sleep(0.1)

    def log_message(self, *args):
        pass


def _start_viewer(port=8000):
    export_camera_json(PROJECT_ROOT / "data" / "scene" / "camera.json")
    os.chdir(PROJECT_ROOT)
    from src.utils.config import load_network
    host = load_network().get("desktop_ip", "0.0.0.0")
    server = http.server.ThreadingHTTPServer((host, port), _ViewerHandler)
    url = f"http://{host}:{port}/src/viewer/viewer.html"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"\n=== 뷰어 시작 (통합 2D/3D + 콘솔) ===")
    print(f"[Server] {url}")
    print(f"[Server] 휴대폰: Tailscale 켜고 같은 주소 접속  (Ctrl+C로 종료)")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="main_r.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "RadYOLO 수신 파이프라인 (AWR1642 레이더 + 웹캠 융합)\n"
            "\n"
            "기본 흐름(플래그 없음):\n"
            "  1) data/ 아카이브 → 2) 레이더/웹캠 수신 → 3) 배경 선택(10s)\n"
            "  4) ESRGAN 업스케일 → 5) DA3 depth → 6) depth 보정 → 7) YOLO 마스크\n"
            "  8) 객체 크롭 → 9) Trellis 3D → 10) GigaPose 포즈 → 라이브 루프 + 뷰어\n"
        ),
        epilog=(
            "예시:\n"
            "  python main_r.py                  전체 파이프라인\n"
            "  python main_r.py --skip-calib     depth 보정만 건너뜀 (metric DA3 시험)\n"
            "  python main_r.py --skip-3d        3D/포즈 없이 거리 융합만\n"
            "  python main_r.py --verify         경량 검증(수신+박스+뷰어)\n"
            "  python main_r.py --calib          레이더↔카메라 외부 캘리브\n"
            "  python main_r.py --viewer-only    뷰어 서버만\n"
        ),
    )
    parser.add_argument("--skip-bg",    action="store_true", help="배경 촬영 건너뜀 (background_raw.jpg 이미 있을 때)")
    parser.add_argument("--skip-depth", action="store_true", help="DA3 건너뜀 (depth.png 이미 있을 때)")
    parser.add_argument("--skip-3d",    action="store_true", help="Trellis 3D 모델 생성 건너뜀")
    parser.add_argument("--skip-calib", action="store_true", help="레이더 기반 depth 보정 건너뜀")
    parser.add_argument("--viewer-only", action="store_true", help="뷰어만 시작")
    parser.add_argument("--no-radar",    action="store_true", help="레이더 생략 (메타 대기 안 함, 웹캠만)")
    parser.add_argument("--verify", action="store_true",
                        help="경량 검증: 수신+radar+YOLO박스+뷰어만 (배경/depth/crop/3D 전부 스킵)")
    parser.add_argument("--calib", action="store_true",
                        help="외부 캘리브: verify 경량 파이프라인 + 레이더↔카메라 yaw/baseline 추정 스레드")
    args = parser.parse_args()

    cam_cfg = load_camera()
    _http_server = None
    _iperf_proc  = None
    _matlab_proc = None

    if args.verify or args.calib:
        mode = "외부 캘리브" if args.calib else "경량 검증"
        print(f"=== {mode} 모드 (수신 + radar + YOLO박스 + 뷰어, 무거운 단계 전부 스킵) ===")
        archive_data()
        record_session_start()
        start_heartbeat_thread()
        _iperf_proc = _start_iperf_server()
        threading.Thread(target=meta_receive,   daemon=True).start()
        threading.Thread(target=radar_receive,  daemon=True).start()
        threading.Thread(target=webcam_receive, daemon=True).start()
        _matlab_proc = _wait_meta_start_radar(args.no_radar)
        _http_server = _start_viewer()
        t_live = threading.Thread(target=live_update_loop, args=(cam_cfg, False), daemon=True)
        t_live.start()
        print("[Live] YOLO 박스 + 레이더 거리 스레드 시작 (pose 끔)")

        if args.calib:
            from src.utils.radar_cam_calib import calibrate_loop
            threading.Thread(
                target=calibrate_loop,
                args=(cam_cfg, receiver.shutdown_event),
                daemon=True,
            ).start()
            print("[Calib] 외부 캘리브 스레드 시작 — 물체를 중앙에서 좌우로 움직이세요.")
            print("[Calib] 수렴 시 config/calib_radar_cam.json 자동 저장 (Ctrl+C 로 종료)")

    elif not args.viewer_only:
        print("=== 이전 data/ 아카이브 중 ===")
        archive_data()
        record_session_start()
        start_heartbeat_thread()

        _iperf_proc = _start_iperf_server()

        t_meta   = threading.Thread(target=meta_receive,   daemon=True)
        t_radar  = threading.Thread(target=radar_receive,  daemon=True)
        t_webcam = threading.Thread(target=webcam_receive, daemon=True)
        t_meta.start()
        t_radar.start()
        t_webcam.start()

        _matlab_proc = _wait_meta_start_radar(args.no_radar)

        print("\n=== Step 0.5: 레이더+웹캠 수신 확인 (3초) ===")
        _verify_intake(3.0)

        if not args.skip_bg:
            print("=== Step 1: 배경 프레임 선택 (10초) ===")
            select_background()
        else:
            print("=== Step 1: 건너뜀 (--skip-bg) ===")

        if not args.skip_bg:
            print("\n=== Step 2: ESRGAN 업스케일 ===")
            upscale_image()
        else:
            print("\n=== Step 2: 건너뜀 (--skip-bg) ===")

        if not args.skip_depth:
            print("\n=== Step 3: DA3 깊이 추정 ===")
            generate_depth()
        else:
            print("\n=== Step 3: 건너뜀 (--skip-depth) ===")

        if not args.skip_calib:
            print("\n=== Step 3.5: 레이더 기반 depth 보정 (계수 산출) ===")
            try:
                calibrate_depth()
            except Exception as e:
                print(f"[Calib] 보정 실패, 건너뜀: {e}")
        else:
            print("\n=== Step 3.5: 건너뜀 (--skip-calib) ===")

        print("\n=== Step 4: 첫 마스크 (초기 객체 식별 + 배경 구멍) ===")
        generate_mask()

        _http_server = _start_viewer()

        t_live = threading.Thread(target=live_update_loop,
                                  args=(cam_cfg, not args.skip_3d), daemon=True)
        t_live.start()
        print(f"[Live] 상태머신 스레드 시작 (pose={'켬' if not args.skip_3d else '끔'})")

        from src.utils.config import load_receiver
        if load_receiver().get("dynamic_bg", {}).get("enabled", True):
            from src.background.dynamic_bg_fill import main as _dynbg_main
            threading.Thread(target=_dynbg_main, daemon=True).start()
            print("[DynBG] 동적 배경 채우기 스레드 시작")

    else:
        _http_server = _start_viewer()

    try:
        _http_server.serve_forever()
    finally:
        record_session_end()
        receiver.close_bin_file()
        if _iperf_proc is not None:
            _iperf_proc.terminate()
        if _matlab_proc is not None:
            _matlab_proc.terminate()
