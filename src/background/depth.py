import json
import shutil
import time
import uuid
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
INPUT_PATH     = PROJECT_ROOT / "data" / "scene" / "background.jpg"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "scene" / "depth.png"
METRIC_PATH    = PROJECT_ROOT / "data" / "scene" / "depth_metric.npy"   # 미터 무손실(Raw 모드)
CALIB_PATH     = PROJECT_ROOT / "data" / "scene" / "depth_calib.json"
WORKFLOW_PATH  = PROJECT_ROOT / "workflows" / "da3_depth.json"


def _write_metric_vis(metric_npy: Path, png_path: Path, calib_path: Path):
    """metric depth(미터, 멀수록 큰값) → 시각화 depth.png(near=bright 0~1 정규화)
    + 뷰어 미터 역산용 depth_calib.json(metric_linear). 8bit PNG 가 미터를 못 담아
    1m 넘으면 잘리는 문제를 정규화로 해결(거리값은 .npy 가 보존)."""
    m = np.load(str(metric_npy)).astype(np.float32)
    mn, mx = float(m.min()), float(m.max())
    vis = 1.0 - (m - mn) / (mx - mn + 1e-8)            # 가까움(작은 미터)=밝음(1)
    cv2.imwrite(str(png_path), (vis * 255).astype(np.uint8))
    calib_path.write_text(json.dumps(
        {"model": "metric_linear", "range_min_m": mn, "range_max_m": mx}, indent=2))

COMFYUI_URL    = "http://127.0.0.1:8188"
COMFYUI_INPUT  = Path("C:/dev/ComfyUI/input")
COMFYUI_OUTPUT = Path("C:/dev/ComfyUI/output")
METRIC_NPY     = COMFYUI_OUTPUT / "da3_metric_raw.npy"   # da3 노드가 덤프하는 미터맵


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


def generate_depth(input_path=None):
    """background.jpg(기본) 또는 input_path 의 이미지로 DA3 depth → depth.png.

    상태머신의 새 물체 DA3 재추론은 input_path 로 '현재 프레임'을 넘긴다(씬 배경
    background.jpg 를 덮어쓰지 않기 위함). 출력은 항상 depth.png(현 씬 depth 갱신).
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    src = Path(input_path) if input_path else INPUT_PATH
    shutil.copy(src, COMFYUI_INPUT / "background.jpg")

    workflow = json.loads(WORKFLOW_PATH.read_text())

    print("[Depth] DA3 실행 중 (시간 소요)...")
    resp    = _queue(workflow)
    history = _wait(resp["prompt_id"])

    outputs = history.get("outputs", {})
    for node_out in outputs.values():
        if "images" in node_out:
            src = COMFYUI_OUTPUT / node_out["images"][0]["filename"]
            shutil.copy(src, OUTPUT_PATH)
            print(f"[Depth] 저장 완료: {OUTPUT_PATH}")
            # Raw(metric) 모드: 미터 무손실 npy 복사 + 시각화 depth.png 재생성
            # (raw 미터를 8bit 로 저장하면 1m 넘는 게 다 흰색으로 잘려 뷰어가 깨짐)
            if METRIC_NPY.exists():
                shutil.copy(METRIC_NPY, METRIC_PATH)
                _write_metric_vis(METRIC_PATH, OUTPUT_PATH, CALIB_PATH)
                print(f"[Depth] metric npy 복사 + 시각화 depth.png 재생성")
            return

    # 에러 내용 출력
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
