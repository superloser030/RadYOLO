import json
import shutil
import time
import uuid
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
SCENE          = PROJECT_ROOT / "data" / "scene"
INPUT_PATH     = SCENE / "background.jpg"
OUTPUT_PATH    = SCENE / "depth.png"
METRIC_PATH    = SCENE / "depth_metric.npy"        # live: 현재 프레임 미터맵 (_mask_dist 객체거리용)
CALIB_PATH     = SCENE / "depth_calib.json"
BG_DEPTH_PATH  = SCENE / "depth_bg.png"            # bg: 배경 시각화 depth (뷰어 배경 포인트클라우드용)
BG_CALIB_PATH  = SCENE / "depth_bg_calib.json"
WORKFLOW_PATH  = PROJECT_ROOT / "workflows_comfyui" / "da3_depth.json"


def _write_metric_vis(metric_npy: Path, png_path: Path, calib_path: Path):
    m = np.load(str(metric_npy)).astype(np.float32)
    mn, mx = float(m.min()), float(m.max())
    vis = 1.0 - (m - mn) / (mx - mn + 1e-8)
    cv2.imwrite(str(png_path), (vis * 255).astype(np.uint8))
    calib_path.write_text(json.dumps(
        {"model": "metric_linear", "range_min_m": mn, "range_max_m": mx}, indent=2))

COMFYUI_URL    = "http://127.0.0.1:8188"
COMFYUI_INPUT  = Path("C:/dev/ComfyUI/input")
COMFYUI_OUTPUT = Path("C:/dev/ComfyUI/output")
METRIC_NPY     = COMFYUI_OUTPUT / "da3_metric_raw.npy"


def _queue(workflow):
    payload = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionRefusedError):
        raise RuntimeError("ComfyUI 미실행 상태. 먼저 ComfyUI를 실행해주세요: python C:/dev/ComfyUI/main.py")


def _wait(prompt_id):
    while True:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as r:
            history = json.loads(r.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)


def generate_depth(input_path=None, mode="both"):
    # mode: "live"=현재프레임 미터맵(depth_metric.npy, _mask_dist 객체거리)만
    #       "bg"=배경 시각화(depth_bg.png + depth_bg_calib.json, 뷰어 배경)만
    #       "both"=둘 다 (초기 Step3 — background.jpg 가 객체 포함이라 겸용)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    src = Path(input_path) if input_path else INPUT_PATH
    shutil.copy(src, COMFYUI_INPUT / "background.jpg")

    workflow = json.loads(WORKFLOW_PATH.read_text())

    print("[Depth] DA3 실행 중 (시간 소요)...")
    resp    = _queue(workflow)
    history = _wait(resp["prompt_id"])

    outputs = history.get("outputs", {})
    if METRIC_NPY.exists():
        if mode in ("live", "both"):
            shutil.copy(METRIC_NPY, METRIC_PATH)
        if mode in ("bg", "both"):
            _write_metric_vis(METRIC_NPY, BG_DEPTH_PATH, BG_CALIB_PATH)
        print(f"[Depth] metric 갱신 (mode={mode})")
        return

    # legacy(metric npy 없음): ComfyUI 가 만든 depth.png 그대로 (구 log_linear 경로)
    for node_out in outputs.values():
        if "images" in node_out:
            shutil.copy(COMFYUI_OUTPUT / node_out["images"][0]["filename"], OUTPUT_PATH)
            print(f"[Depth] 저장 완료: {OUTPUT_PATH}")
            return

    status = history.get("status", {})
    msgs   = status.get("messages", [])
    for msg in msgs:
        if msg[0] == "execution_error":
            print(f"[Depth] ComfyUI 에러: {msg[1]}")
    if not outputs:
        print(f"[Depth] outputs 비어있음. status={status}")
    raise RuntimeError("[Depth] DA3 결과 없음")


if __name__ == "__main__":
    generate_depth()
