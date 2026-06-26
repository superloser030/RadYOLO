import os
import sys
import json
import shutil
import threading
import webbrowser
import http.server
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def archive_data():
    data_dir = PROJECT_ROOT / "data"
    archive_base = PROJECT_ROOT / "archive"
    archive_base.mkdir(exist_ok=True)
    if not data_dir.exists():
        return

    # model_trellis.glb + templates/ 는 보존 (Trellis/포즈 자산)
    KEEP = ["model_trellis.glb", "templates"]
    saved = {}   # rel_path → bytes (글로벌 파일) 또는 dir_path
    objects_dir = data_dir / "objects"
    if objects_dir.exists():
        for obj_dir in objects_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            glb = obj_dir / "model_trellis.glb"
            tmpl = obj_dir / "templates"
            if glb.exists():
                saved[obj_dir.name + "/model_trellis.glb"] = glb.read_bytes()
            if tmpl.exists():
                # templates/ 전체 복사 (in-memory 대신 임시 위치)
                tmp = archive_base / f"_tmp_templates_{obj_dir.name}"
                if tmp.exists():
                    shutil.rmtree(tmp)
                shutil.copytree(str(tmpl), str(tmp))
                saved[obj_dir.name + "/__templates__"] = tmp

    idx = 1
    while (archive_base / f"data_{idx:03d}").exists():
        idx += 1
    dest = archive_base / f"data_{idx:03d}"
    shutil.move(str(data_dir), str(dest))
    print(f"[Archive] data/ → archive/data_{idx:03d}")

    # 보존 파일 복원
    if saved:
        new_objects = data_dir / "objects"
        new_objects.mkdir(parents=True, exist_ok=True)
        for rel, payload in saved.items():
            obj_name, fname = rel.split("/", 1)
            obj_dir = new_objects / obj_name
            obj_dir.mkdir(exist_ok=True)
            if fname == "__templates__":
                shutil.copytree(str(payload), str(obj_dir / "templates"))
                shutil.rmtree(payload)
            else:
                (obj_dir / fname).write_bytes(payload)
        print(f"[Archive] GLB/templates 보존: {list(saved.keys())}")
sys.path.insert(0, str(PROJECT_ROOT / "src" / "transmission"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "background"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "objects"))

import receiver
from receiver  import radar_receive, webcam_receive
from bg_select import select_background
from upscale   import upscale_image
from depth     import generate_depth
from yolo_mask import generate_mask
from obj_crop  import crop_objects
from trellis_gen import generate_3d


def estimate_object_poses():
    """GLB 모델이 있는 객체에 대해 GigaPose로 포즈 추정 후 pose.json + manifest.json 저장."""
    from pose_estimator import prepare_templates, estimate_pose

    cam_cfg    = json.loads((PROJECT_ROOT / "config" / "camera.json").read_text())
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


def _run_pose_bg(obj_dir, frame_path, bbox, cam_cfg):
    """백그라운드 스레드: GigaPose 추론 → pose.json 갱신 (회전/깊이)."""
    from pose_estimator import estimate_pose
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
        print(f"[Live-R] {obj_dir.name}: score={pose.get('score',0):.3f}  t={[round(v,3) for v in pose['t']]}")


def live_update_loop(cam_cfg):
    """라이브 루프: YOLO 위치(0.3s) + GigaPose 회전(이전 추론 완료 즉시) 갱신."""
    import time, cv2
    from ultralytics import YOLO

    YOLO_MODEL = str(PROJECT_ROOT / "models" / "yolo11x-seg.pt")

    yolo = YOLO(YOLO_MODEL)
    objects_dir = PROJECT_ROOT / "data" / "objects"
    cam_w = cam_cfg.get("width",  1920)
    cam_h = cam_cfg.get("height", 1080)
    inferring = {}   # obj_name → bool

    while not receiver.shutdown_event.is_set():
        frame = receiver.get_latest_frame()
        if frame is None:
            time.sleep(0.2)
            continue

        fh, fw = frame.shape[:2]
        sx = cam_w / fw
        sy = cam_h / fh

        results = yolo(frame, verbose=False, conf=0.4)

        if not objects_dir.exists():
            time.sleep(0.3)
            continue

        for obj_dir in sorted(objects_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            meta_p = obj_dir / "meta.json"
            if not meta_p.exists():
                continue
            meta = json.loads(meta_p.read_text())
            cls  = meta.get("class", "")

            best = None
            for box in results[0].boxes:
                name = yolo.names[int(box.cls[0])]
                if name == cls:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    if best is None or conf > best["conf"]:
                        best = {
                            "bbox_cx": (x1 + x2) / 2 * sx,
                            "bbox_cy": (y1 + y2) / 2 * sy,
                            "bbox":    [x1*sx, y1*sy, x2*sx, y2*sy],
                            "conf":    conf,
                            "ts":      time.time(),
                        }

            if best:
                (obj_dir / "live.json").write_text(json.dumps(best))

                # 이전 추론이 끝난 즉시 다음 추론 시작
                key = obj_dir.name
                if not inferring.get(key, False):
                    inferring[key] = True
                    frame_path = str(obj_dir / "live_frame.jpg")
                    up = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(frame_path, up)

                    def _launch(od=obj_dir, fp=frame_path, bbox=best["bbox"], k=key):
                        try:
                            _run_pose_bg(od, fp, bbox, cam_cfg)
                        finally:
                            inferring[k] = False

                    threading.Thread(target=_launch, daemon=True).start()

        time.sleep(0.3)


def serve_and_open(port=8000):
    os.chdir(PROJECT_ROOT)
    server = http.server.HTTPServer(
        ("localhost", port),
        http.server.SimpleHTTPRequestHandler
    )
    url = f"http://localhost:{port}/src/viewer/viewer.html"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"[Server] {url}")
    print("[Server] Ctrl+C로 종료")
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bg",    action="store_true", help="배경 촬영 건너뜀 (background_raw.jpg 이미 있을 때)")
    parser.add_argument("--skip-depth", action="store_true", help="DA3 건너뜀 (depth.png 이미 있을 때)")
    parser.add_argument("--viewer-only", action="store_true", help="뷰어만 시작")
    args = parser.parse_args()

    if not args.viewer_only:
        print("=== 이전 data/ 아카이브 중 ===")
        archive_data()

        # receiver 스레드 시작 (sender에서 오는 프레임/레이더 수신)
        t_radar  = threading.Thread(target=radar_receive,  daemon=True)
        t_webcam = threading.Thread(target=webcam_receive, daemon=True)
        t_radar.start()
        t_webcam.start()

        if not args.skip_bg:
            print("=== Step 1: 배경 프레임 선택 (10초) ===")
            select_background()
        else:
            print("=== Step 1: 건너뜀 (--skip-bg) ===")

        print("\n=== Step 2: ESRGAN 업스케일 ===")
        upscale_image()

        if not args.skip_depth:
            print("\n=== Step 3: DA3 깊이 추정 ===")
            generate_depth()
        else:
            print("\n=== Step 3: 건너뜀 (--skip-depth) ===")

        print("\n=== Step 4: YOLO 마스크 생성 ===")
        generate_mask()

        print("\n=== Step 5: 객체 크롭 + depth 마스크 정제 ===")
        crop_objects()

        print("\n=== Step 5.3: Trellis 3D 모델 생성 ===")
        objects_dir = PROJECT_ROOT / "data" / "objects"
        if objects_dir.exists():
            for obj_dir in sorted(objects_dir.iterdir()):
                if obj_dir.is_dir() and (obj_dir / "cutout.jpg").exists():
                    generate_3d(obj_dir)

        print("\n=== Step 5.5: GLB 모델 포즈 추정 ===")
        estimate_object_poses()

    print("\n=== Step 6: 뷰어 시작 ===")
    cam_cfg = json.loads((PROJECT_ROOT / "config" / "camera.json").read_text())
    if not args.viewer_only:
        t_live = threading.Thread(target=live_update_loop, args=(cam_cfg,), daemon=True)
        t_live.start()
        print("[Live] YOLO 위치 / GigaPose 회전 갱신 스레드 시작")
    serve_and_open()
