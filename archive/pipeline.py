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
OUTPUT_PATH = Path("D:/projects/RadYOLO/background.jpg")
COMFYUI_URL    = "http://127.0.0.1:8188"
COMFYUI_INPUT  = Path("C:/dev/ComfyUI/input")
COMFYUI_OUTPUT = Path("C:/dev/ComfyUI/output")

BBOX_PADDING = 30


# ── ComfyUI 유틸 ───────────────────────────────────────────────────────────
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
        body = e.read().decode()
        print(f"ComfyUI 에러 ({e.code}): {body}")
        raise


def wait_for_result(prompt_id):
    while True:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as r:
            history = json.loads(r.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)


def free_vram():
    req = urllib.request.Request(
        f"{COMFYUI_URL}/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req)


# ── Step 0: 깊이맵 생성 ────────────────────────────────────────────────────
def generate_depth(image_path):
    shutil.copy(image_path, COMFYUI_INPUT / "inpaint_image.png")

    workflow = {
        "1": {
            "class_type": "DownloadAndLoadDepthAnythingV3Model",
            "inputs": {"model": "da3_large.safetensors"}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": "inpaint_image.png"}
        },
        "3": {
            "class_type": "DepthAnything_V3",
            "inputs": {
                "da3_model": ["1", 0],
                "images":    ["2", 0],
                "normalization_mode": "Standard",  # close=bright(높은값), far=dark(낮은값)
                "invert_depth": False
            }
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "depth_map"}
        }
    }

    result    = queue_prompt(workflow)
    history   = wait_for_result(result["prompt_id"])

    for node_output in history["outputs"].values():
        if "images" in node_output:
            return COMFYUI_OUTPUT / node_output["images"][0]["filename"]
    return None


# ── Step 1: YOLO 감지 ──────────────────────────────────────────────────────
def detect_objects(image_path, depth_map):
    yolo = YOLO("yolo11x-seg.pt")
    detections = yolo(str(image_path), conf=0.3)
    det = detections[0]

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    objects = []

    if det.boxes is not None:
        for box, cls, conf in zip(det.boxes.xyxy, det.boxes.cls, det.boxes.conf):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            name = yolo.names[int(cls)]
            # Standard 모드: 밝을수록 가까움 → 높은 값 = 전경
            depth_val = float(depth_map[y1:y2, x1:x2].mean())
            objects.append({
                "class": name,
                "conf":  float(conf),
                "bbox":  [x1, y1, x2, y2],
                "depth": depth_val
            })
            print(f"  {name} ({conf:.2f})  depth={depth_val:.1f}")

    # 가까운(전경, 높은 depth값) 객체부터 먼저 제거
    objects.sort(key=lambda o: -o["depth"])
    return objects, h, w


# ── Step 2: 순차 FLUX 인페인팅 ────────────────────────────────────────────
def make_bbox_mask(h, w, bbox):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - BBOX_PADDING)
    y1 = max(0, y1 - BBOX_PADDING)
    x2 = min(w, x2 + BBOX_PADDING)
    y2 = min(h, y2 + BBOX_PADDING)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def flux_inpaint(image_path, mask_path, step_prefix, label="object"):
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
            "inputs": {
                "text": "no objects, no people, clean background, empty room",
                "clip": ["3", 0]
            }
        },
        "6": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["5", 0], "guidance": 30.0}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": f"{label}",
                "clip": ["3", 0]
            }
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
            "inputs": {"images": ["13", 0], "filename_prefix": step_prefix}
        }
    }

    result    = queue_prompt(workflow)
    history   = wait_for_result(result["prompt_id"])

    for node_output in history["outputs"].values():
        if "images" in node_output:
            return COMFYUI_OUTPUT / node_output["images"][0]["filename"]
    return None


# ── 메인 ──────────────────────────────────────────────────────────────────
img_orig = cv2.imread(str(IMAGE_PATH))
tmp_img  = Path("D:/projects/RadYOLO/tmp_current.png")
tmp_mask = Path("D:/projects/RadYOLO/tmp_mask.png")
cv2.imwrite(str(tmp_img), img_orig)

print("=== Step 0: 깊이맵 생성 ===")
depth_result = generate_depth(tmp_img)
if depth_result is None:
    raise RuntimeError("깊이맵 생성 실패")
depth_raw = cv2.imread(str(depth_result), cv2.IMREAD_GRAYSCALE)
h, w = img_orig.shape[:2]
depth_map = cv2.resize(depth_raw, (w, h)).astype(float)
print(f"깊이맵 생성 완료: {depth_result.name}")

print("\n=== Step 1: YOLO 감지 ===")
objects, h, w = detect_objects(IMAGE_PATH, depth_map)
print(f"제거 순서: {[o['class'] for o in objects]}")

print("\n=== Step 2: 순차 인페인팅 ===")
current_img = tmp_img

for i, obj in enumerate(objects):
    label = obj["class"]
    print(f"\n[{i+1}/{len(objects)}] {label} 제거 중...")

    mask = make_bbox_mask(h, w, obj["bbox"])
    cv2.imwrite(str(tmp_mask), mask)

    result_path = flux_inpaint(current_img, tmp_mask, step_prefix=f"step{i+1}_{label}", label=label)
    if result_path is None or not result_path.exists():
        print(f"  실패 - 스킵")
        continue

    shutil.copy(result_path, tmp_img)
    print(f"  완료 → {result_path.name}")

final = cv2.imread(str(tmp_img))
cv2.imwrite(str(OUTPUT_PATH), final, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"\n=== 완료: {OUTPUT_PATH} ===")
