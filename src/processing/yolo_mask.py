import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_PATH = "D:/projects/RadYOLO/test.jpg"
OUTPUT_MASK = "D:/projects/RadYOLO/mask.png"
OUTPUT_PREVIEW = "D:/projects/RadYOLO/mask_preview.jpg"

model = YOLO("yolo11x-seg.pt")
results = model(IMAGE_PATH, conf=0.3)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
combined_mask = np.zeros((h, w), dtype=np.uint8)

result = results[0]
if result.masks is not None:
    for i, (mask, cls, conf) in enumerate(zip(result.masks.data, result.boxes.cls, result.boxes.conf)):
        class_name = model.names[int(cls)]
        print(f"감지: {class_name} ({conf:.2f})")
        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (w, h))
        combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8) * 255)
else:
    print("감지된 객체 없음")

cv2.imwrite(OUTPUT_MASK, combined_mask)

preview = img.copy()
preview[combined_mask > 0] = preview[combined_mask > 0] * 0.4 + np.array([0, 0, 255]) * 0.6
cv2.imwrite(OUTPUT_PREVIEW, preview.astype(np.uint8))

print(f"마스크 저장: {OUTPUT_MASK}")
print(f"프리뷰 저장: {OUTPUT_PREVIEW}")
