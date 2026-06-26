import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
MODEL_PATH     = PROJECT_ROOT / "models" / "yolo11x-seg.pt"
INPUT_PATH     = PROJECT_ROOT / "data" / "scene" / "background.jpg"
OUTPUT_MASK    = PROJECT_ROOT / "data" / "scene" / "mask.png"
OUTPUT_PREVIEW = PROJECT_ROOT / "data" / "scene" / "mask_preview.jpg"


def generate_mask():
    OUTPUT_MASK.parent.mkdir(parents=True, exist_ok=True)
    model   = YOLO(str(MODEL_PATH))
    results = model(str(INPUT_PATH), conf=0.3)

    img = cv2.imread(str(INPUT_PATH))
    h, w = img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    result = results[0]
    if result.masks is not None:
        for mask, cls, conf in zip(result.masks.data, result.boxes.cls, result.boxes.conf):
            print(f"[Mask] {model.names[int(cls)]} ({conf:.2f})")
            mask_np = mask.cpu().numpy()
            mask_rs = cv2.resize(mask_np, (w, h))
            combined_mask = np.maximum(combined_mask, (mask_rs > 0.5).astype(np.uint8) * 255)
    else:
        print("[Mask] 감지된 객체 없음")

    cv2.imwrite(str(OUTPUT_MASK), combined_mask)

    preview = img.copy().astype(np.float32)
    preview[combined_mask > 0] = preview[combined_mask > 0] * 0.4 + np.array([0, 0, 255]) * 0.6
    cv2.imwrite(str(OUTPUT_PREVIEW), preview.astype(np.uint8))

    print(f"[Mask] 저장: {OUTPUT_MASK}")


if __name__ == "__main__":
    generate_mask()
