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


def generate_depth():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(INPUT_PATH, COMFYUI_INPUT / "background.jpg")

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


def free_comfyui():
    """ComfyUI 가 캐시한 모델 언로드 + VRAM 반환 (ESRGAN/DA3 다 쓴 뒤 호출).

    ComfyUI 는 모델을 캐시해 안 내리므로, depth/upscale 끝나면 /free 로 비운다.
    """
    try:
        req = urllib.request.Request(
            f"{COMFYUI_URL}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=15)
        print("[VRAM] ComfyUI 모델 언로드 (ESRGAN/DA3 VRAM 반환)")
    except Exception as e:
        print(f"[VRAM] ComfyUI free 실패(무시): {e}")


if __name__ == "__main__":
    generate_depth()
