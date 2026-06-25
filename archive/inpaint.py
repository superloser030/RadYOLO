import json
import shutil
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_INPUT = Path("C:/dev/ComfyUI/input")
COMFYUI_OUTPUT = Path("C:/dev/ComfyUI/output")

IMAGE_PATH = Path("D:/projects/RadYOLO/test2.jpg")
MASK_PATH  = Path("D:/projects/RadYOLO/mask.png")
OUTPUT_PATH = Path("D:/projects/RadYOLO/background.jpg")

# ComfyUI input 폴더에 복사
shutil.copy(IMAGE_PATH, COMFYUI_INPUT / "inpaint_image.jpg")
shutil.copy(MASK_PATH,  COMFYUI_INPUT / "inpaint_mask.png")

CLIENT_ID = str(uuid.uuid4())

WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0_0.9vae.safetensors"}
    },
    "2": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "diffusion_pytorch_model.fp16.safetensors",
            "weight_dtype": "fp8_e4m3fn"
        }
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "empty room, clean background, no people, indoor room, high quality",
            "clip": ["1", 1]
        }
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "person, human, people, chair, furniture, objects",
            "clip": ["1", 1]
        }
    },
    "5": {
        "class_type": "LoadImage",
        "inputs": {"image": "inpaint_image.jpg"}
    },
    "6": {
        "class_type": "LoadImage",
        "inputs": {"image": "inpaint_mask.png"}
    },
    "7": {
        "class_type": "ImageToMask",
        "inputs": {"image": ["6", 0], "channel": "red"}
    },
    "8": {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": ["5", 0],
            "vae": ["1", 2],
            "mask": ["7", 0],
            "grow_mask_by": 40
        }
    },
    "9": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["2", 0],
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["8", 0],
            "seed": 42,
            "steps": 30,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["9", 0],
            "vae": ["1", 2]
        }
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["10", 0],
            "filename_prefix": "background"
        }
    }
}

def queue_prompt(workflow):
    payload = json.dumps({"prompt": workflow, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as r:
        return json.loads(r.read())

print("ComfyUI 인페인팅 요청 중...")
result = queue_prompt(WORKFLOW)
prompt_id = result["prompt_id"]
print(f"prompt_id: {prompt_id}")

print("완료 대기 중...")
while True:
    history = get_history(prompt_id)
    if prompt_id in history:
        break
    time.sleep(2)

outputs = history[prompt_id]["outputs"]
for node_id, node_output in outputs.items():
    if "images" in node_output:
        for img in node_output["images"]:
            src = COMFYUI_OUTPUT / img["filename"]
            shutil.copy(src, OUTPUT_PATH)
            print(f"배경 이미지 저장: {OUTPUT_PATH}")
