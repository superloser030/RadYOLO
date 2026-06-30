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
DB_DIR       = PROJECT_ROOT / "db"   # 세션 간 영구 오브젝트 DB (archive 대상 아님)

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
    """GLB 모델이 있는 객체에 대해 GigaPose로 포즈 추정 후 pose.json + manifest.json 저장."""
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

        bbox = meta.get("bbox")   # [cx1, cy1, cx2, cy2] in original image
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
    """HSV 64-bin 색 히스토그램 — re-ID 외형 특징. 마스크 영역만 사용(없으면 bbox 전체)."""
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
    """graveyard 에서 cls·위치·외형 매칭 → (tid, entry) or None.

    위치: 중심거리 < bbox 폭 × 1.5.  외형: HSV 히스토그램 상관계수 ≥ hist_th.
    점수 = 위치 40% + 외형 60% — 가중 합산 최고점 선택.
    """
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
        hist_score = 0.5   # 외형 특징 없을 때 중립값
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
    """data/objects/ 스캔 → manifest.json 갱신 (뷰어가 어떤 오브젝트를 로드할지 결정)."""
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
    """GLB 완성 오브젝트를 영구 DB에 저장. 세션 간 재사용을 위해."""
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
    """DB에서 같은 클래스 + HSV 유사도 ≥ threshold 인 항목 검색 → Path or None."""
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
    """DB 항목을 data/objects/<cls>_<tid>/ 로 복사 후 경로 반환."""
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
    """백그라운드 스레드: GigaPose 추론 → pose.json 갱신 (회전/깊이)."""
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
    """라이브 상태머신: YOLO-seg 추적 + 레이더거리 + 객체 상태(NEW→MODELING→READY).

    1단계: 단일뷰, re-ID 없음(IoU+miss 추적). crop/3D/pose 를 이벤트화 — 새 물체가
    obj_conf 로 등록되고 model_conf 를 넘으면 DA3 재추론(거리)+crop+fal.ai 3D(백그라운드)
    → READY 시 GigaPose pose. enable_pose=False 면 추적/거리만(3D/pose 안 함).
    """
    import time, cv2
    import numpy as np
    from ultralytics import YOLO
    from src.objects.radar_fusion import load_latest_targets, match_one, depth_to_range, iou
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
    OBJECTS_DIR  = PROJECT_ROOT / "data" / "objects"
    DA3_INPUT    = SCENE / "da3_frame.jpg"   # 새 물체 DA3 재추론 입력(배경 안 건드림)

    ycfg        = load_receiver().get("yolo", {})
    obj_conf    = float(ycfg.get("obj_conf",   0.72))   # 객체 등록 게이트
    model_conf  = float(ycfg.get("model_conf", 0.85))   # 3D+pose 게이트
    live_conf   = float(ycfg.get("live_conf",  0.5))
    detect_conf = min(live_conf, obj_conf)   # 검출은 낮게(추적 안정), 등록만 obj_conf

    # 배경 depth + 보정계수 — 새 물체 DA3 재추론에 같은 계수 재적용. 리스트로 nonlocal 흉내.
    _depth = [None]; _depth_calib = [None]
    def _reload_depth():
        try:
            if DEPTH_PATH.exists() and CALIB_PATH.exists():
                d = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
                if d is not None and d.ndim != 2:
                    d = d[:, :, 0]
                _depth[0] = d
                _depth_calib[0] = json.loads(CALIB_PATH.read_text())
        except Exception:
            pass
    _reload_depth()
    if _depth[0] is not None:
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
    _sam2_lock = threading.Lock()   # SAM2 단일 인스턴스 동시 호출 직렬화
    _da3_lock  = threading.Lock()   # DA3 재추론 공유(동시 MODELING 시 1회만)
    _da3_last  = [0.0]

    def _mask_dist(m_frame, cx, cy):
        """마스크 영역 DA3 거리 median(m). 마스크 없으면 (cx,cy) 1점. depth 없으면 None."""
        dep, cal = _depth[0], _depth_calib[0]
        if dep is None or cal is None:
            return None
        dh, dw = dep.shape[:2]
        if m_frame is not None:
            mm = cv2.resize(m_frame, (dw, dh)) if (dh, dw) != m_frame.shape else m_frame
            vals = dep[mm > 0.5]
            if len(vals) > 0:
                return depth_to_range(float(np.median(vals)) / 255.0, cal)
        bx = max(0, min(dw - 1, int(cx * dw / cam_w)))
        by = max(0, min(dh - 1, int(cy * dh / cam_h)))
        return depth_to_range(dep[by, bx] / 255.0, cal)

    def _da3_rerun(frame_raw):
        """새 물체용: 현재 프레임 → ESRGAN 업스케일 → DA3 (배경과 '동일' 파이프라인)
        → depth.png 갱신(같은 보정계수 재적용). background.jpg(씬 배경)는 안 건드림.
        ※ 배경도 ESRGAN 후 DA3 이므로, 재추론도 같은 화질 경로를 타야 depth 가 일관됨."""
        try:
            raw = SCENE / "da3_raw.jpg"
            up  = SCENE / "da3_up.jpg"
            cv2.imwrite(str(raw), frame_raw)
            upscale_image(input_path=str(raw), output_path=str(up))   # ESRGAN (배경과 동일)
            generate_depth(input_path=str(up))
            cal = _depth_calib[0]
            if cal is not None:
                d = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
                if d is not None:
                    if d.ndim != 2:
                        d = d[:, :, 0]
                    cn = apply_calib(d.astype(np.float32) / 255.0, cal)
                    cv2.imwrite(str(DEPTH_PATH), (cn * 255).astype(np.uint8))
            _reload_depth()
            print("[State] DA3 재추론 → depth 갱신")
        except Exception as e:
            print(f"[State] DA3 재추론 실패: {e}")

    def _da3_rerun_shared(frame_big):
        """동시 MODELING 여러 개여도 DA3 재추론 1회만(락 + 5초 디바운스). depth 는 전체
        프레임이라 객체마다 돌릴 필요 없음 → 중복/충돌 방지."""
        if not _da3_lock.acquire(blocking=False):
            return                          # 이미 누가 재추론 중 → 공유(스킵)
        try:
            if time.time() - _da3_last[0] < 5.0:
                return                      # 최근 5초 내 했으면 재사용
            _da3_rerun(frame_big)
            _da3_last[0] = time.time()
        finally:
            _da3_lock.release()

    registry  = {}      # tid -> {cls,bbox,conf,miss,state,last_range,glb,appearance}
    graveyard = {}      # tid -> {…, evicted_at} — re-ID 대기 (TTL 내 재등록 시 같은 tid 재사용)
    _busy     = {}      # tid / f"pose{tid}" -> bool (백그라운드 진행 가드)
    _next_tid = [0]
    MISS_MAX, IOU_TH = 10, 0.15
    GRAVEYARD_TTL  = 120.0   # 120초 후 graveyard 에서 영구 삭제
    MIN_VIEWS      = 3        # COLLECTING → MODELING 최소 뷰 수
    COLLECT_TIMEOUT = 30.0    # 30초 경과 시 보유 뷰로 강제 진입
    MOVE_TH        = 0.30     # bbox폭 대비 중심이동 임계값
    _last_fusion = 0.0

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

        # ── 1. 검출 정리 (마스크 tight bbox + DA3 거리) ──
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

        # ── 2. IoU+중심 매칭 → registry 갱신/등록 (re-ID 는 2단계) ──
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
            elif d["conf"] >= obj_conf:   # 새 객체는 obj_conf 통과 시만 등록
                reid = _try_reid(graveyard, d["name"], d["bbox"], d["appearance"],
                                 time.time(), GRAVEYARD_TTL)
                if reid:
                    rtid, rentry = reid
                    del graveyard[rtid]
                    st = rentry.get("state", "NEW")
                    if st == "MODELING":
                        st = "NEW"   # MODELING 스레드는 이미 종료됨
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
                                         "miss": 0, "state": "READY", "last_range": None,
                                         "glb": str(od / "model_trellis.glb"),
                                         "appearance": d["appearance"]}
                        _update_manifest()
                        print(f"[DB-ReID] {d['name']}_{tid} DB 복원 → READY 즉시 진입")
                    else:
                        registry[tid] = {"cls": d["name"], "bbox": d["bbox"], "conf": d["conf"],
                                         "miss": 0, "state": "NEW", "last_range": None, "glb": None,
                                         "appearance": d["appearance"]}

        # 미매칭 registry: miss++ → graveyard 이동 (re-ID 대기)
        for tid in list(registry):
            if tid not in matched:
                registry[tid]["miss"] += 1
                if registry[tid]["miss"] > MISS_MAX:
                    entry = registry.pop(tid)
                    entry["evicted_at"] = time.time()
                    graveyard[tid] = entry
                    _busy.pop(tid, None); _busy.pop(f"pose{tid}", None)
                    print(f"[State] {entry['cls']}_{tid} → graveyard")

        # ── 3. 거리 매칭 + overlay + 상태 전이 ──
        det_by_tid   = {d["tid"]: d for d in dets if d["tid"] is not None}
        fusion_lines = []; overlay = []
        for tid, r in list(registry.items()):
            d = det_by_tid.get(tid)
            if d is None:
                continue   # 이번 프레임 미검출(miss 중)
            inst = f"{r['cls']}_{tid}"
            radar = match_one(targets, r["bbox"], cam_cfg, bbox_dist=d["mdist"],
                              last_range=r.get("last_range"), last_az=r.get("last_az"),
                              mask=d["m_frame"])
            o = {"name": inst, "cls": r["cls"], "conf": round(r["conf"], 2),
                 "bbox": [round(v) for v in r["bbox"]], "state": r["state"]}
            if radar:
                # 레이더 실측(n>0)일 때만 추적 기준 갱신 — DA3/유지값 오염 방지
                if radar.get("n_points", 0) > 0:
                    r["last_range"] = radar["range_m"]
                    r["last_az"]    = radar["azimuth_deg"]
                o["range_m"] = radar["range_m"]; o["az"] = radar["azimuth_deg"]
                o["v"] = radar["velocity_mps"];  o["n"]  = radar.get("n_points")
                fusion_lines.append(f"[Fusion] {inst:14} {radar['range_m']:6.2f}m | "
                                    f"az {radar['azimuth_deg']:+6.1f} | n {radar.get('n_points')} | {r['state']}")
            else:
                fusion_lines.append(f"[Fusion] {inst:14} miss | {r['state']}")
            overlay.append(o)

            if not enable_pose:
                continue
            # NEW →(model_conf)→ COLLECTING: 멀티뷰 수집 시작
            if (r["state"] == "NEW" and r["conf"] >= model_conf
                    and r["cls"] not in SKIP_CLASSES):
                r["state"] = "COLLECTING"
                r["views"] = []; r["collect_start"] = time.time()
                r["last_view_cx"] = r["last_view_cy"] = None
                od = OBJECTS_DIR / f"{r['cls']}_{tid}"
                (od / "views").mkdir(parents=True, exist_ok=True)
                print(f"[State] {r['cls']}_{tid} NEW → COLLECTING")

            # COLLECTING: 움직임 감지 시 뷰 저장 → 조건 충족 시 MODELING 진입
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
                # 타임아웃 + 뷰 없음 → 현재 프레임 단일뷰로 강제 저장
                if elapsed > COLLECT_TIMEOUT and not r["views"] and d["m_frame"] is not None:
                    od = OBJECTS_DIR / f"{r['cls']}_{tid}"
                    vp = od / "views" / "view_000.jpg"
                    cv2.imwrite(str(vp), cv2.resize(frame, (cam_w, cam_h)))
                    r["views"].append({"cx": d["cx"], "cy": d["cy"],
                                       "bbox": list(r["bbox"]), "path": str(vp)})
                # 충분한 뷰 또는 타임아웃 → MODELING
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
                                # 첫 번째 뷰 → main cutout (obj_dir 구조 생성)
                                f0 = cv2.imread(views[0]["path"])
                                sam.set_image(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
                                crop_one(f0, views[0]["bbox"], cls_name, tid, sam)
                                # 추가 뷰 → views/ 디렉토리에 cutout 저장
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

            # READY → GigaPose pose (백그라운드)
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

        # graveyard TTL 정리
        for _tid in [k for k, v in graveyard.items() if now - v["evicted_at"] > GRAVEYARD_TTL]:
            del graveyard[_tid]

        time.sleep(0.3)


def _start_iperf_server():
    """센더 mode 1(자동 대역폭) 측정용 iperf3 -s 상시 서버. iperf3 없으면 None."""
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
    """레이더 실시간 추적(radar_live.m)을 MATLAB 배치로 자동 실행.

    예제 모듈 조립 파이프라인(readRadarCube→…→trackObjects). 옛 cfar_detect_live.m
    은 legacy_matlab/ 으로 archive 됨.
    .mat 생성(메타 수신) 후에 호출할 것. 출력은 logs/radar_live_*.log 로.
    matlab 명령이 PATH 에 없으면 None (수동 실행 필요).
    """
    import datetime as _dt
    matlab_dir = PROJECT_ROOT / "matlab"
    # 로그는 data/ 밖(logs/)에 타임스탬프로 — archive 이동/좀비 잠금과 무관하게
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


def _verify_intake(secs=3.0):
    """레이더 targets + 웹캠 프레임이 실제로 들어오는지 secs 초 확인(형식 검증)."""
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
    """정적 파일 + /stream (실시간 웹캠 MJPEG)."""
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
            _t.sleep(0.1)   # ~10fps

    def log_message(self, *args):   # 폴링/스트림 로그 도배 방지
        pass


def _start_viewer(port=8000):
    # sender.toml 의 intrinsic 을 viewer 가 fetch 할 JSON 으로 export
    # (archive_data 가 data/ 를 비운 뒤여야 하므로 viewer 시작 시점에 생성)
    export_camera_json(PROJECT_ROOT / "data" / "scene" / "camera.json")
    os.chdir(PROJECT_ROOT)
    # Tailscale IP 에 바인드 → 휴대폰 등 Tailscale 기기에서 외부 접근(외부인 차단).
    # network.toml 의 desktop_ip(데스크톱 Tailscale IP)를 사용.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bg",    action="store_true", help="배경 촬영 건너뜀 (background_raw.jpg 이미 있을 때)")
    parser.add_argument("--skip-depth", action="store_true", help="DA3 건너뜀 (depth.png 이미 있을 때)")
    parser.add_argument("--skip-3d",    action="store_true", help="Trellis 3D 모델 생성 건너뜀")
    parser.add_argument("--skip-calib", action="store_true", help="레이더 기반 depth 보정 건너뜀")
    parser.add_argument("--viewer-only", action="store_true", help="뷰어만 시작")
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
        # verify 와 calib 은 같은 경량 파이프라인(수신+radar+YOLO박스+뷰어).
        # calib 은 여기에 레이더↔카메라 yaw/baseline 추정 스레드만 추가로 얹는다.
        mode = "외부 캘리브" if args.calib else "경량 검증"
        print(f"=== {mode} 모드 (수신 + radar + YOLO박스 + 뷰어, 무거운 단계 전부 스킵) ===")
        archive_data()
        record_session_start()
        start_heartbeat_thread()
        _iperf_proc = _start_iperf_server()
        threading.Thread(target=meta_receive,   daemon=True).start()
        threading.Thread(target=radar_receive,  daemon=True).start()
        threading.Thread(target=webcam_receive, daemon=True).start()
        import time as _vt
        print("[Init] 레이더 메타 수신 대기...")
        if receiver._chirp_ready.wait(timeout=20):
            print("[Init] 메타 확정 — 2초 후 radar_live 시작")
            _vt.sleep(2)
            _matlab_proc = _start_matlab_cfar()
        else:
            print("[Init] 메타 타임아웃(20s) — 웹캠만 (레이더 미시작)")
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

        # 레이더 메타(.mat/chirp) 확정 후 안정화되면 배경 캡처 시작
        import time as _time
        print("[Init] 레이더 메타 수신 대기...")
        if receiver._chirp_ready.wait(timeout=20):
            print("[Init] 메타 확정 — 2초 후 시작")
            _time.sleep(2)
            _matlab_proc = _start_matlab_cfar()   # .mat 생성됨 → 실시간 CFAR 시작
        else:
            print("[Init] 메타 타임아웃(20s) — 웹캠만으로 진행 (레이더 CFAR 미시작)")

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
        # DA3 상주(release 안 함) — 라이브에서 새 물체 등장 시 즉시 재추론하기 위함.
        # (ESRGAN 도 같이 상주, VRAM 16GB 여유. 기존 GPUManager.release 제거)

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

        # 배경/깊이/마스크 준비 완료 → 뷰어 열기.
        # crop/3D/pose 는 시작 1회가 아니라 live_update_loop 상태머신이 이벤트로 처리
        # (새 물체 등장 → NEW→DA3재추론→crop→fal.ai 3D→READY→GigaPose pose).
        _http_server = _start_viewer()

        t_live = threading.Thread(target=live_update_loop,
                                  args=(cam_cfg, not args.skip_3d), daemon=True)
        t_live.start()
        print(f"[Live] 상태머신 스레드 시작 (pose={'켬' if not args.skip_3d else '끔'})")

        # 동적 배경 채우기 백그라운드 (receiver.toml [dynamic_bg] enabled)
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
