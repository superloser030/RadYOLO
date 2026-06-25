import json
import shutil
import time
import uuid
import random
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

IMAGE_PATH  = Path("D:/projects/RadYOLO/test.jpg")
OUTPUT_PATH = Path("D:/projects/RadYOLO/background2.jpg")
COMFYUI_URL    = "http://127.0.0.1:8188"
COMFYUI_INPUT  = Path("C:/dev/ComfyUI/input")
COMFYUI_OUTPUT = Path("C:/dev/ComfyUI/output")

BBOX_PADDING = 30
    

def queue_prompt(workflow):
    payload = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"ComfyUI 에러 ({e.code}): {e.read().decode()}")
        raise


def wait_for_result(prompt_id):
    while True:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as r:
            history = json.loads(r.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)


def flux_inpaint(image_path, mask_path):
    shutil.copy(image_path, COMFYUI_INPUT / "inpaint_image.png")
    shutil.copy(mask_path,  COMFYUI_INPUT / "inpaint_mask.png")

    workflow = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "flux1-fill-dev-Q8_0.gguf"}
        },
        "2": {
            "class_type": "DifferentialDiffusion",
            "inputs": {"model": ["1", 0]}
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux"
            }
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["3", 0]}
        },
        "6": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["5", 0], "guidance": 30.0}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["3", 0]}
        },
        "8": {
            "class_type": "LoadImage",
            "inputs": {"image": "inpaint_image.png"}
        },
        "9": {
            "class_type": "LoadImage",
            "inputs": {"image": "inpaint_mask.png"}
        },
        "10": {
            "class_type": "ImageToMask",
            "inputs": {"image": ["9", 0], "channel": "red"}
        },
        "11": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive":   ["6", 0],
                "negative":   ["7", 0],
                "vae":        ["4", 0],
                "pixels":     ["8", 0],
                "mask":       ["10", 0],
                "noise_mask": True
            }
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["2", 0],
                "positive":     ["11", 0],
                "negative":     ["11", 1],
                "latent_image": ["11", 2],
                "seed":         random.randint(0, 2**32 - 1),
                "steps":        20,
                "cfg":          1.0,
                "sampler_name": "euler",
                "scheduler":    "normal",
                "denoise":      1.0
            }
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["12", 0], "vae": ["4", 0]}
        },
        "14": {
            "class_type": "SaveImage",
            "inputs": {"images": ["13", 0], "filename_prefix": "background2"}
        }
    }

    result  = queue_prompt(workflow)
    history = wait_for_result(result["prompt_id"])

    for node_output in history["outputs"].values():
        if "images" in node_output:
            return COMFYUI_OUTPUT / node_output["images"][0]["filename"]
    return None


# ── 메인 ──────────────────────────────────────────────────────────────────
img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]

print("=== Step 1: YOLO 감지 ===")
yolo = YOLO("yolo11x-seg.pt")
det = yolo(str(IMAGE_PATH), conf=0.3)[0]

mask = np.zeros((h, w), dtype=np.uint8)

if det.boxes is not None:
    for box, cls, conf in zip(det.boxes.xyxy, det.boxes.cls, det.boxes.conf):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        label = yolo.names[int(cls)]
        print(f"  {label} ({conf:.2f})  bbox=[{x1},{y1},{x2},{y2}]")
        x1 = max(0, x1 - BBOX_PADDING)
        y1 = max(0, y1 - BBOX_PADDING)
        x2 = min(w, x2 + BBOX_PADDING)
        y2 = min(h, y2 + BBOX_PADDING)
        mask[y1:y2, x1:x2] = 255

print(f"총 {len(det.boxes) if det.boxes else 0}개 객체 → 마스크 합산 완료")

tmp_img  = Path("D:/projects/RadYOLO/tmp_current.png")
tmp_mask = Path("D:/projects/RadYOLO/tmp_mask.png")
cv2.imwrite(str(tmp_img),  img)
cv2.imwrite(str(tmp_mask), mask)

print("\n=== Step 2: FLUX 인페인팅 (1회) ===")
result_path = flux_inpaint(tmp_img, tmp_mask)

if result_path and result_path.exists():
    shutil.copy(result_path, OUTPUT_PATH)
    print(f"완료 → {OUTPUT_PATH}")
else:
    print("실패")
