import os
import sys
import json
import shutil
import threading
import subprocess
import webbrowser
import http.server
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from src.utils.archive import (
    archive_data,
    record_session_start,
    record_session_end,
    start_heartbeat_thread,
)
from src.utils.config import load_camera, export_camera_json
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


def _start_viewer(port=8000):
    # sender.toml 의 intrinsic 을 viewer 가 fetch 할 JSON 으로 export
    # (archive_data 가 data/ 를 비운 뒤여야 하므로 viewer 시작 시점에 생성)
    export_camera_json(PROJECT_ROOT / "data" / "scene" / "camera.json")
    os.chdir(PROJECT_ROOT)
    server = http.server.HTTPServer(
        ("localhost", port),
        http.server.SimpleHTTPRequestHandler
    )
    url = f"http://localhost:{port}/src/viewer/viewer.html"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"\n=== 뷰어 시작 ===")
    print(f"[Server] {url}  (Ctrl+C로 종료)")
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
    args = parser.parse_args()

    cam_cfg = load_camera()
    _http_server = None
    _iperf_proc  = None

    if not args.viewer_only:
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
        else:
            print("[Init] 메타 타임아웃(20s) — 웹캠만으로 진행")

        if not args.skip_bg:
            print("=== Step 1: 배경 프레임 선택 (10초) ===")
            select_background()
        else:
            print("=== Step 1: 건너뜀 (--skip-bg) ===")

        _http_server = _start_viewer()

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
            print("\n=== Step 3.5: 레이더 기반 depth 보정 ===")
            calibrate_depth()
        else:
            print("\n=== Step 3.5: 건너뜀 (--skip-calib) ===")

        print("\n=== Step 4: YOLO 마스크 생성 ===")
        generate_mask()

        print("\n=== Step 5: 객체 크롭 + depth 마스크 정제 ===")
        crop_objects()

        if not args.skip_3d:
            print("\n=== Step 5.3: Trellis 3D 모델 생성 ===")
            objects_dir = PROJECT_ROOT / "data" / "objects"
            if objects_dir.exists():
                for obj_dir in sorted(objects_dir.iterdir()):
                    if obj_dir.is_dir() and (obj_dir / "cutout.jpg").exists():
                        generate_3d(obj_dir)

            print("\n=== Step 5.5: GLB 모델 포즈 추정 ===")
            estimate_object_poses()
        else:
            print("\n=== Step 5.3/5.5: 건너뜀 (--skip-3d) ===")

        t_live = threading.Thread(target=live_update_loop, args=(cam_cfg,), daemon=True)
        t_live.start()
        print("[Live] YOLO 위치 / GigaPose 회전 갱신 스레드 시작")

    else:
        _http_server = _start_viewer()

    try:
        _http_server.serve_forever()
    finally:
        record_session_end()
        receiver.close_bin_file()
        if _iperf_proc is not None:
            _iperf_proc.terminate()
