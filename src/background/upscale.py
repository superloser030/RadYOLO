import json
import shutil
import time
import uuid
import urllib.request
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
INPUT_PATH     = PROJECT_ROOT / "data" / "scene" / "background_raw.jpg"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "scene" / "background.jpg"

COMFYUI_URL    = "http://127.0.0.1:8188"
COMFYUI_INPUT  = Path("C:/dev/ComfyUI/input")
COMFYUI_OUTPUT = Path("C:/dev/ComfyUI/output")
ESRGAN_MODEL   = "RealESRGAN_x4.pth"


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
        raise RuntimeError("ComfyUI 미실행 상태. 먼저 ComfyUI를 실행해주세요.")


def _wait(prompt_id):
    while True:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as r:
            history = json.loads(r.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)


def upscale_image():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(INPUT_PATH, COMFYUI_INPUT / "background_raw.jpg")

    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "background_raw.jpg"}
        },
        "2": {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": ESRGAN_MODEL}
        },
        "3": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["2", 0],
                "image": ["1", 0]
            }
        },
        "4": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "width": 1920,
                "height": 1080,
                "crop": "disabled"
            }
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["4", 0],
                "filename_prefix": "background"
            }
        }
    }

    print(f"[Upscale] Real-ESRGAN x4 실행 중...")
    resp    = _queue(workflow)
    history = _wait(resp["prompt_id"])

    for node_out in history["outputs"].values():
        if "images" in node_out:
            src = COMFYUI_OUTPUT / node_out["images"][0]["filename"]
            shutil.copy(src, OUTPUT_PATH)
            print(f"[Upscale] 저장 완료: {OUTPUT_PATH}")
            return

    raise RuntimeError("[Upscale] 결과 없음")


if __name__ == "__main__":
    upscale_image()
