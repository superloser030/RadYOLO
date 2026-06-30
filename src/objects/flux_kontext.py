import json
import random
import shutil
import time
import uuid
import urllib.request
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
OBJECTS_DIR    = PROJECT_ROOT / "data" / "objects"
WORKFLOW_PATH  = PROJECT_ROOT / "workflows_comfyui" / "flux_kontext.json"

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
        raise RuntimeError("ComfyUI 미실행. python C:/dev/ComfyUI/main.py 먼저 실행하세요.")


def _wait(prompt_id):
    while True:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as r:
            history = json.loads(r.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)


def run_kontext(obj_dir: Path):
    meta       = json.loads((obj_dir / "meta.json").read_text())
    class_name = meta["class"]
    crop_path  = obj_dir / "crop.jpg"
    out_path   = obj_dir / "kontext.jpg"

    fname = f"kontext_input_{obj_dir.name}.jpg"
    shutil.copy(crop_path, COMFYUI_INPUT / fname)

    workflow = json.loads(WORKFLOW_PATH.read_text())
    workflow["4"]["inputs"]["image"] = fname
    workflow["7"]["inputs"]["clip_l"] = f"white background {class_name} product photo"
    workflow["7"]["inputs"]["t5xxl"] = (
        f"isolated {class_name} on pure white background, "
        "product photography, studio lighting, clean sharp edges, no shadows"
    )
    workflow["9"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
    workflow["11"]["inputs"]["filename_prefix"] = f"kontext_{obj_dir.name}"

    print(f"[Kontext] {class_name}_{meta['idx']} 처리 중...")
    resp    = _queue(workflow)
    history = _wait(resp["prompt_id"])

    for node_out in history.get("outputs", {}).values():
        if "images" in node_out:
            src = COMFYUI_OUTPUT / node_out["images"][0]["filename"]
            shutil.copy(src, out_path)
            print(f"[Kontext] 저장: {out_path.relative_to(PROJECT_ROOT)}")
            return

    raise RuntimeError(f"[Kontext] {obj_dir.name} 결과 없음")


def run_all():
    if not OBJECTS_DIR.exists():
        print("[Kontext] data/objects/ 없음. obj_crop.py 먼저 실행하세요.")
        return

    obj_dirs = sorted(d for d in OBJECTS_DIR.iterdir() if d.is_dir() and (d / "crop.jpg").exists())
    if not obj_dirs:
        print("[Kontext] 크롭된 객체 없음. obj_crop.py 먼저 실행하세요.")
        return

    for obj_dir in obj_dirs:
        run_kontext(obj_dir)


if __name__ == "__main__":
    run_all()
