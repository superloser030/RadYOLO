import json
import shutil
import time
import uuid
import urllib.request
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
INPUT_PATH     = PROJECT_ROOT / "data" / "scene" / "background.jpg"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "scene" / "depth.png"
WORKFLOW_PATH  = PROJECT_ROOT / "workflows" / "da3_depth.json"

COMFYUI_URL    = "http://127.0.0.1:8188"
COMFYUI_INPUT  = Path("C:/dev/ComfyUI/input")
COMFYUI_OUTPUT = Path("C:/dev/ComfyUI/output")


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
