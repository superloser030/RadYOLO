import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

from src.transmission.receiver import frame_queue
from src.utils.config import load_receiver, load_camera
from src.objects.radar_fusion import load_latest_targets, match_one

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH   = PROJECT_ROOT / "models" / "yolo11x-seg.pt"
_MASK_CONF   = load_receiver().get("yolo", {}).get("mask_conf", 0.4)
OUTPUT_PATH  = PROJECT_ROOT / "data" / "scene" / "background_raw.jpg"
TARGETS_PATH = PROJECT_ROOT / "data" / "radar" / "targets.json"
CAPTURE_SECS = 10

W_SHARP = 0.25
W_RADAR = 0.75


def _best_object_snr(frame, result, targets, cam) -> float:
    """이 프레임에서 레이더에 가장 또렷하게 잡힌 객체의 SNR(=신뢰도).

    각 YOLO 객체 마스크에 레이더 점을 매칭(match_one)해, 매칭된 점들의 최대 SNR
    중 가장 큰 값을 반환. 매칭 0이면 0. cam_w(1920) 좌표계로 bbox/mask 스케일."""
    if result.boxes is None or not targets:
        return 0.0
    h, w = frame.shape[:2]
    cam_w = cam.get("width", 1920)
    sx, sy = cam_w / w, cam.get("height", 1080) / h
    masks = result.masks
    best = 0.0
    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes[i].xyxy[0].tolist()
        bbox = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        m_frame = None
        if masks is not None and i < len(masks.data):
            m = masks.data[i].cpu().numpy()
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h))
            m_frame = m
        radar = match_one(targets, bbox, cam, mask=m_frame)
        if radar:
            best = max(best, float(radar.get("snr", 0.0)))
    return best


def _boundary_sharpness(frame, result) -> float:
    """YOLO 마스크 경계선 위 color gradient 평균 → 선명도 점수."""
    if result.masks is None or result.boxes is None:
        return 0.0

    h, w = frame.shape[:2]

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
    cam   = load_camera()

    frames, sharp_scores, radar_scores, timestamps = [], [], [], []

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

        result  = model(frame, conf=_MASK_CONF, verbose=False)[0]
        sharp   = _boundary_sharpness(frame, result)
        targets = load_latest_targets(TARGETS_PATH, ts_ms)
        radar   = _best_object_snr(frame, result, targets, cam)

        frames.append(frame.copy())
        sharp_scores.append(sharp)
        radar_scores.append(radar)
        timestamps.append(ts_ms)

        elapsed = time.time() - start
        print(f"\r[BG] {elapsed:.1f}s / {CAPTURE_SECS}s  프레임: {len(frames)}  "
              f"선명도: {sharp:.1f}  레이더SNR: {radar:.1f}", end="")

    print()
    sharp_arr = np.array(sharp_scores, dtype=np.float32)
    radar_arr = np.array(radar_scores, dtype=np.float32)
    sn = sharp_arr / (sharp_arr.max() + 1e-8)
    rn = radar_arr / (radar_arr.max() + 1e-8)
    if radar_arr.max() <= 0:
        final = sn
        print("[BG] 레이더 매칭 없음 — 선명도만으로 선정")
    else:
        final = W_SHARP * sn + W_RADAR * rn
    best_idx = int(np.argmax(final))
    cv2.imwrite(str(OUTPUT_PATH), frames[best_idx])
    TS_PATH.write_text(str(timestamps[best_idx]))
    print(f"[BG] 저장 완료: {OUTPUT_PATH}  (final={final[best_idx]:.2f}, "
          f"선명도={sharp_scores[best_idx]:.1f}, 레이더SNR={radar_scores[best_idx]:.1f}, "
          f"frame={best_idx+1}/{len(frames)}, ts={timestamps[best_idx]})")
