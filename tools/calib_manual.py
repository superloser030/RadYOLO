"""
수동 렌즈 왜곡 보정 도구
왼쪽: 보정된 카메라 이미지
오른쪽: depth.png 기반 top-down 포인트클라우드 (벽/바닥이 직선이 되면 OK)
s: 저장 / q: 종료
"""
import cv2
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_PATH     = PROJECT_ROOT / "data" / "scene" / "background.jpg"
DEPTH_PATH   = PROJECT_ROOT / "data" / "scene" / "depth.png"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "scene" / "background_undist.jpg"
CALIB_PATH   = PROJECT_ROOT / "config" / "calib.json"

img   = cv2.imread(str(IMG_PATH))
depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"background.jpg 없음: {IMG_PATH}")
if depth is None:
    raise FileNotFoundError(f"depth.png 없음: {DEPTH_PATH}")

h, w = img.shape[:2]
depth = cv2.resize(depth, (w, h)).astype(np.float32) / 255.0

STEP    = 3     # 픽셀 서브샘플링 (빠른 업데이트)
PC_SIZE = 600   # 포인트클라우드 뷰 크기

WIN = "캘리브레이션 (s=저장, q=종료)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1600, 650)

# k1, k2: -0.5 ~ +0.5  (슬라이더 0~1000, 기준=500)
# fx: 50~200%           (슬라이더 50~200, 기준=100 = 이미지 너비)
cv2.createTrackbar("k1 x1000", WIN, 500, 1000, lambda v: None)
cv2.createTrackbar("k2 x1000", WIN, 500, 1000, lambda v: None)
cv2.createTrackbar("fx  pct",  WIN, 100, 200,  lambda v: None)


def make_pointcloud_topdown(img_ud, depth_map, fx_px):
    cx, cy = w / 2.0, h / 2.0

    ys, xs = np.mgrid[0:h:STEP, 0:w:STEP]
    d      = depth_map[ys, xs]
    colors = img_ud[ys, xs]          # BGR

    mask = d > 0.02
    xs, ys, d, colors = xs[mask], ys[mask], d[mask], colors[mask]

    X = (xs - cx) * d / fx_px
    Z = d
    Y = (ys - cy) * d / fx_px

    if len(X) == 0:
        return np.zeros((PC_SIZE, PC_SIZE, 3), np.uint8)

    x_min, x_max = X.min(), X.max()
    z_min, z_max = Z.min(), Z.max()

    px = ((X - x_min) / (x_max - x_min + 1e-6) * (PC_SIZE - 1)).astype(int)
    pz = ((Z - z_min) / (z_max - z_min + 1e-6) * (PC_SIZE - 1)).astype(int)
    pz = PC_SIZE - 1 - pz   # 가까운 쪽이 아래

    view = np.full((PC_SIZE, PC_SIZE, 3), 20, dtype=np.uint8)
    for i in range(len(px)):
        cv2.circle(view, (px[i], pz[i]), 1, colors[i].tolist(), -1)

    cv2.putText(view, "top-down (X vs depth Z)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return view


prev   = (None, None, None)
frame  = None
pcview = None

while True:
    k1_raw = cv2.getTrackbarPos("k1 x1000", WIN)
    k2_raw = cv2.getTrackbarPos("k2 x1000", WIN)
    fx_raw = cv2.getTrackbarPos("fx  pct",  WIN)

    params = (k1_raw, k2_raw, fx_raw)
    if params != prev:
        k1    = (k1_raw - 500) / 1000.0
        k2    = (k2_raw - 500) / 1000.0
        fx_px = max(w * fx_raw / 100.0, 1.0)
        cx, cy = w / 2.0, h / 2.0

        K    = np.array([[fx_px, 0, cx], [0, fx_px, cy], [0, 0, 1]], np.float64)
        dist = np.array([k1, k2, 0, 0, 0], np.float64)

        img_ud = cv2.undistort(img, K, dist)
        pcview = make_pointcloud_topdown(img_ud, depth, fx_px)

        cv2.putText(img_ud, f"k1={k1:.3f}  k2={k2:.3f}  fx={fx_px:.0f}px ({fx_raw}%)",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        frame = img_ud
        prev  = params

    if frame is None:
        continue

    disp_h   = 590
    img_disp = cv2.resize(frame,  (int(w * disp_h / h), disp_h))
    pc_disp  = cv2.resize(pcview, (disp_h, disp_h))
    combined = np.hstack([img_disp, pc_disp])

    cv2.imshow(WIN, combined)
    key = cv2.waitKey(16) & 0xFF

    if key == ord('s'):
        k1    = (cv2.getTrackbarPos("k1 x1000", WIN) - 500) / 1000.0
        k2    = (cv2.getTrackbarPos("k2 x1000", WIN) - 500) / 1000.0
        fx_px = max(w * cv2.getTrackbarPos("fx  pct", WIN) / 100.0, 1.0)
        K    = np.array([[fx_px, 0, w/2.0], [0, fx_px, h/2.0], [0, 0, 1]], np.float64)
        dist = np.array([k1, k2, 0, 0, 0], np.float64)
        cv2.imwrite(str(OUTPUT_PATH), cv2.undistort(img, K, dist))
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIB_PATH.write_text(json.dumps(
            {"k1": k1, "k2": k2, "fx": fx_px, "fy": fx_px,
             "cx": w/2.0, "cy": h/2.0, "image_size": [w, h]}, indent=2))
        print(f"저장 완료  →  {OUTPUT_PATH}")
        print(f"캘리브레이션  →  {CALIB_PATH}")

    elif key in (ord('q'), 27):
        break

cv2.destroyAllWindows()
