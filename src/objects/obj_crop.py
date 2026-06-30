import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.utils.config import load_receiver

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
_MODEL_CONF   = load_receiver().get("yolo", {}).get("model_conf", 0.85)
YOLO_PATH     = PROJECT_ROOT / "models" / "yolo11x-seg.pt"
SAM2_CKPT     = PROJECT_ROOT / "models" / "sam2.1_hiera_large.pt"
SAM2_CFG      = "configs/sam2.1/sam2.1_hiera_l.yaml"
INPUT_PATH    = PROJECT_ROOT / "data" / "scene" / "background.jpg"
DEPTH_PATH    = PROJECT_ROOT / "data" / "scene" / "depth.png"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "objects"

SKIP_CLASSES  = {"person"}
BBOX_PAD      = 40


def _load_sam2():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(SAM2_CFG, str(SAM2_CKPT), device="cuda")
    return SAM2ImagePredictor(model)


def _build_gradient_map(img: np.ndarray, depth: np.ndarray):
    """debug용: color(Lab) + depth gradient 시각화."""
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_grad = np.zeros((h, w), np.float32)
    for c in range(3):
        ch = cv2.GaussianBlur(lab[:, :, c], (5, 5), 0)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        color_grad = np.maximum(color_grad, np.sqrt(gx**2 + gy**2))
    c_norm    = color_grad / (color_grad.max() + 1e-6)
    color_vis = (c_norm * 255).clip(0, 255).astype(np.uint8)

    depth_f   = cv2.GaussianBlur(depth.astype(np.float32), (5, 5), 0)
    gx        = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    gy        = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    depth_raw = np.sqrt(gx**2 + gy**2)
    d_norm    = depth_raw / (depth_raw.max() + 1e-6)
    depth_vis = (d_norm * 255).clip(0, 255).astype(np.uint8)

    return depth_vis, color_vis


def crop_objects():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # YOLO: 감지 + 클래스
    yolo    = YOLO(str(YOLO_PATH))
    results = yolo(str(INPUT_PATH), conf=_MODEL_CONF, verbose=False)
    result  = results[0]

    img     = cv2.imread(str(INPUT_PATH))
    h, w    = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
    if depth_raw is None:
        raise FileNotFoundError(f"depth.png 없음: {DEPTH_PATH}")
    depth = cv2.resize(depth_raw, (w, h))

    if result.boxes is None:
        print("[Crop] 감지된 객체 없음")
        return []

    # SAM2: 마스크 생성
    print("[Crop] SAM2 로드 중...")
    sam2 = _load_sam2()
    sam2.set_image(img_rgb)

    depth_vis, color_vis = _build_gradient_map(img, depth)
    combined_vis = np.maximum(depth_vis, color_vis)

    objects   = []
    cls_count = {}

    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        class_name = yolo.names[int(cls)]

        if class_name in SKIP_CLASSES:
            print(f"[Crop] {class_name} 스킵")
            continue

        idx = cls_count.get(class_name, 0)
        cls_count[class_name] = idx + 1

        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        # SAM2로 정밀 마스크 생성
        masks, scores, _ = sam2.predict(
            box=np.array([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=False
        )
        sam_mask = (masks[0] > 0).astype(np.uint8) * 255

        # bbox crop 범위
        cx1 = max(0, x1 - BBOX_PAD)
        cy1 = max(0, y1 - BBOX_PAD)
        cx2 = min(w, x2 + BBOX_PAD)
        cy2 = min(h, y2 + BBOX_PAD)

        def crop(arr):
            return arr[cy1:cy2, cx1:cx2]

        obj_dir = OUTPUT_DIR / f"{class_name}_{idx}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        dbg_dir = obj_dir / "debug"
        dbg_dir.mkdir(exist_ok=True)

        # ── 최종 결과물 ──────────────────────────────────────────────
        cv2.imwrite(str(obj_dir / "crop.jpg"), crop(img), [cv2.IMWRITE_JPEG_QUALITY, 95])

        white = np.full_like(img, 255)
        cutout = np.where(sam_mask[:, :, None] > 0, img, white)
        cv2.imwrite(str(obj_dir / "cutout.jpg"), crop(cutout), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(obj_dir / "mask.png"), crop(sam_mask))

        # ── 파이프라인 중간 단계 (debug/) ────────────────────────────
        cv2.imwrite(str(dbg_dir / "1_sam2_mask.png"),   crop(sam_mask))
        cv2.imwrite(str(dbg_dir / "2_depth_grad.png"),  crop(depth_vis))
        cv2.imwrite(str(dbg_dir / "3_color_grad.png"),  crop(color_vis))
        cv2.imwrite(str(dbg_dir / "4_combined.png"),    crop(combined_vis))

        meta = {
            "class": class_name, "idx": idx,
            "conf": float(conf), "sam2_score": float(scores[0]),
            "bbox": [cx1, cy1, cx2, cy2]
        }
        (obj_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"[Crop] {class_name}_{idx}  conf={conf:.2f}  sam2={scores[0]:.3f}  → {obj_dir.relative_to(PROJECT_ROOT)}")
        objects.append((class_name, idx, obj_dir))

    # SAM2 는 crop 단계에서만 쓰므로 GPU 에서 내림 (VRAM 반환)
    del sam2
    from src.utils.gpu_scheduler import GPUManager
    GPUManager.release("SAM2")

    return objects


def load_sam2():
    """SAM2 predictor 로드 — 상태머신 이벤트 루프가 lazy 로드/재사용."""
    return _load_sam2()


def crop_one_extra(img, bbox, sam2):
    """추가 뷰 cutout ndarray 반환 (디렉토리 생성 없음). SAM2 set_image는 호출자 책임."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    masks, _, _ = sam2.predict(
        box=np.array([x1, y1, x2, y2], dtype=np.float32), multimask_output=False)
    sam_mask = (masks[0] > 0).astype(np.uint8) * 255
    cx1, cy1 = max(0, x1 - BBOX_PAD), max(0, y1 - BBOX_PAD)
    cx2, cy2 = min(w, x2 + BBOX_PAD), min(h, y2 + BBOX_PAD)
    white  = np.full_like(img, 255)
    cutout = np.where(sam_mask[:, :, None] > 0, img, white)
    return cutout[cy1:cy2, cx1:cx2]


def crop_one(img, bbox, class_name, tid, sam2):
    """현재 프레임의 한 객체 → data/objects/<class>_<tid>/ (cutout/crop/mask/meta).

    상태머신 MODELING 단계용(이벤트). crop_objects(배경 1장 배치)와 달리 현재 프레임의
    특정 bbox 하나만 처리한다. sam2 는 set_image 된 predictor(호출자가 프레임마다
    set_image 호출). bbox=(x1,y1,x2,y2). 반환: obj_dir(Path).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    masks, scores, _ = sam2.predict(
        box=np.array([x1, y1, x2, y2], dtype=np.float32), multimask_output=False)
    sam_mask = (masks[0] > 0).astype(np.uint8) * 255

    cx1, cy1 = max(0, x1 - BBOX_PAD), max(0, y1 - BBOX_PAD)
    cx2, cy2 = min(w, x2 + BBOX_PAD), min(h, y2 + BBOX_PAD)
    def crop(arr):
        return arr[cy1:cy2, cx1:cx2]

    obj_dir = OUTPUT_DIR / f"{class_name}_{tid}"
    obj_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(obj_dir / "crop.jpg"),  crop(img),    [cv2.IMWRITE_JPEG_QUALITY, 95])
    white  = np.full_like(img, 255)
    cutout = np.where(sam_mask[:, :, None] > 0, img, white)
    cv2.imwrite(str(obj_dir / "cutout.jpg"), crop(cutout), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(obj_dir / "mask.png"),   crop(sam_mask))
    meta = {"class": class_name, "idx": tid, "sam2_score": float(scores[0]),
            "bbox": [cx1, cy1, cx2, cy2]}
    (obj_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return obj_dir


if __name__ == "__main__":
    crop_objects()
