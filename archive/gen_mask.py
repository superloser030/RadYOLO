import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

IMAGE_PATH  = Path("D:/projects/RadYOLO/test.jpg")
MASK_PATH   = Path("D:/projects/RadYOLO/mask.png")
BBOX_PADDING = 30

img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]

print("=== YOLO 감지 ===")
yolo = YOLO("yolo11x-seg.pt")
det = yolo(str(IMAGE_PATH), conf=0.3)[0]

mask = np.zeros((h, w), dtype=np.uint8)

if det.boxes is not None:
    for box, cls, conf in zip(det.boxes.xyxy, det.boxes.cls, det.boxes.conf):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        label = yolo.names[int(cls)]
        print(f"  {label} ({conf:.2f})")
        x1 = max(0, x1 - BBOX_PADDING)
        y1 = max(0, y1 - BBOX_PADDING)
        x2 = min(w, x2 + BBOX_PADDING)
        y2 = min(h, y2 + BBOX_PADDING)
        mask[y1:y2, x1:x2] = 255

cv2.imwrite(str(MASK_PATH), mask)
print(f"\n마스크 저장 완료 → {MASK_PATH}")
print(f"총 {len(det.boxes) if det.boxes else 0}개 객체 마스킹")
