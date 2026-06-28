import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

from src.transmission.receiver import frame_queue

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH   = PROJECT_ROOT / "models" / "yolo11x-seg.pt"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "scene" / "background_raw.jpg"
CAPTURE_SECS = 10


def _boundary_sharpness(frame, result) -> float:
    """YOLO 마스크 경계선 위 color gradient 평균 → 선명도 점수."""
    if result.masks is None or result.boxes is None:
        return 0.0

    h, w = frame.shape[:2]

    # Lab color gradient
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad = np.zeros((h, w), np.float32)
    for c in range(3):
        ch = cv2.GaussianBlur(lab[:, :, c], (5, 5), 0)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.maximum(grad, np.sqrt(gx**2 + gy**2))

    scores = []
    for mask_data in result.masks.data:
        mask     = cv2.resize(mask_data.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        mask     = (mask > 0.5).astype(np.uint8)
        boundary = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - mask
        if boundary.sum() == 0:
            continue
        scores.append(float(grad[boundary > 0].mean()))

    return float(np.mean(scores)) if scores else 0.0


TS_PATH = PROJECT_ROOT / "data" / "scene" / "background_ts.txt"


def select_background():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    frames, scores, timestamps = [], [], []

    print("[BG] sender에서 프레임 수신 대기 중...")
    try:
        frame_queue.get(timeout=30)
    except Exception:
        raise RuntimeError("sender에서 프레임이 오지 않음. sender.py가 실행 중인지 확인하세요.")

    print(f"[BG] {CAPTURE_SECS}초 촬영 시작...")
    start = time.time()

    while time.time() - start < CAPTURE_SECS:
        try:
            frame, ts_ms = frame_queue.get(timeout=1.0)
        except Exception:
            continue

        result  = model(frame, conf=0.3, verbose=False)[0]
        score   = _boundary_sharpness(frame, result)

        frames.append(frame.copy())
        scores.append(score)
        timestamps.append(ts_ms)

        elapsed = time.time() - start
        print(f"\r[BG] {elapsed:.1f}s / {CAPTURE_SECS}s  프레임: {len(frames)}  선명도: {score:.1f}", end="")

    print()
    best_idx = int(np.argmax(scores))
    cv2.imwrite(str(OUTPUT_PATH), frames[best_idx])
    TS_PATH.write_text(str(timestamps[best_idx]))
    print(f"[BG] 저장 완료: {OUTPUT_PATH}  (score={scores[best_idx]:.1f}, frame={best_idx+1}/{len(frames)}, ts={timestamps[best_idx]})")
