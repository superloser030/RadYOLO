# 동적 배경 채우기: 객체가 비켜나 드러난 영역(480p)으로 배경 구멍을 메우고,
# 진행 단계마다 ESRGAN 으로 업스케일해 background.jpg 갱신. 설정은 receiver.toml [dynamic_bg].
import json
import threading
import time
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.config import load_receiver

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
SCENE          = PROJECT_ROOT / "data" / "scene"
WEBCAM         = PROJECT_ROOT / "data" / "webcam"
DYN            = SCENE / "dynamic_bg"
CHOSEN         = DYN / "chosen"
RAW_PATH       = SCENE / "background_raw.jpg"     # 저해상 원본(시작점)
FILLED_PATH    = DYN / "filled_raw.jpg"           # 채워지는 중인 저해상 배경
REMAINING_PATH = DYN / "remaining.png"            # 남은 가려진 영역(진행 확인)
OUT_BG         = SCENE / "background.jpg"          # 최종 업스케일 결과(뷰어가 읽음)
YOLO_MODEL     = PROJECT_ROOT / "models" / "yolo11x-seg.pt"


def _esrgan(src: Path, dst: Path):
    """ComfyUI Real-ESRGAN x4 → 1920x1080 (upscale.py 헬퍼/워크플로 재사용)."""
    from src.background.upscale import _queue, _wait, COMFYUI_INPUT, COMFYUI_OUTPUT, ESRGAN_MODEL
    name = Path(src).name
    shutil.copy(src, COMFYUI_INPUT / name)
    wf = {
        "1": {"class_type": "LoadImage",            "inputs": {"image": name}},
        "2": {"class_type": "UpscaleModelLoader",   "inputs": {"model_name": ESRGAN_MODEL}},
        "3": {"class_type": "ImageUpscaleWithModel","inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "ImageScale",           "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                                               "width": 1920, "height": 1080, "crop": "disabled"}},
        "5": {"class_type": "SaveImage",            "inputs": {"images": ["4", 0], "filename_prefix": "dynbg"}},
    }
    resp = _queue(wf)
    hist = _wait(resp["prompt_id"])
    for node_out in hist["outputs"].values():
        if "images" in node_out:
            shutil.copy(COMFYUI_OUTPUT / node_out["images"][0]["filename"], dst)
            return
    raise RuntimeError("ESRGAN 결과 없음")


_depth_lock = threading.Lock()   # DynBG DA3 재추론 직렬화 (concurrent 방지)


def _rerun_depth(bg_path: Path):
    """background.jpg 갱신 후 DA3 재추론 + 보정계수 재적용 → depth.png.
    비동기(daemon thread) — DynBG 루프 블로킹 없이 뷰어 포인트클라우드 갱신 신호 제공."""
    if not _depth_lock.acquire(blocking=False):
        return   # 이미 재추론 중 → 최신 배경으로 곧 덮어써질 것이므로 스킵
    def _work():
        try:
            from src.background.depth import generate_depth
            from src.background.depth_calibration import apply_calib
            generate_depth(input_path=str(bg_path))
            calib_p = SCENE / "depth_calib.json"
            if calib_p.exists():
                cal = json.loads(calib_p.read_text())
                d = cv2.imread(str(SCENE / "depth.png"), cv2.IMREAD_GRAYSCALE)
                if d is not None:
                    if d.ndim != 2:
                        d = d[:, :, 0]
                    cn = apply_calib(d.astype(np.float32) / 255.0, cal)
                    cv2.imwrite(str(SCENE / "depth.png"), (cn * 255).astype(np.uint8))
            (SCENE / "bg_ts.txt").write_text(str(time.time()))   # 뷰어 재빌드 신호
            print("[DynBG] depth.png 갱신 완료 → 뷰어 포인트클라우드 재빌드 신호")
        except Exception as e:
            print(f"[DynBG] depth 재추론 실패: {e}")
        finally:
            _depth_lock.release()
    threading.Thread(target=_work, daemon=True).start()


def _upscale(src: Path, dst: Path, use_esrgan: bool):
    if use_esrgan:
        try:
            _esrgan(src, dst)
            print("[DynBG] ESRGAN 업스케일 → background.jpg")
            return
        except Exception as e:
            print(f"[DynBG] ESRGAN 실패({e}) → lanczos fallback")
    img = cv2.imread(str(src))
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(dst), big)
    print("[DynBG] lanczos 업스케일 → background.jpg")


def _object_mask(yolo, img, dilate_px=0, conf=0.4):
    """사람+사물 전부의 합집합 마스크(255=객체). img 와 같은 크기.
    dilate_px>0 이면 마스크를 팽창 — 손가락/머리카락 등 빡빡한 경계 밖 잔상 제거."""
    h, w = img.shape[:2]
    res = yolo(img, verbose=False, conf=conf)[0]
    m = np.zeros((h, w), np.uint8)
    if res.masks is not None:
        for md in res.masks.data:
            mm = cv2.resize(md.cpu().numpy(), (w, h))
            m = np.maximum(m, (mm > 0.5).astype(np.uint8) * 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m


def main():
    cfg = load_receiver().get("dynamic_bg", {})
    if not cfg.get("enabled", True):
        print("[DynBG] disabled (receiver.toml)")
        return
    cover_min  = float(cfg.get("cover_min", 0.005))
    steps      = list(cfg.get("upscale_steps", [0.20, 0.25, 0.30]))
    use_esrgan = bool(cfg.get("upscale", True))
    poll       = float(cfg.get("poll_sec", 1.0))
    dilate_px  = int(cfg.get("mask_dilate_px", 0))
    mask_conf  = float(load_receiver().get("yolo", {}).get("mask_conf", 0.4))

    DYN.mkdir(parents=True, exist_ok=True)
    CHOSEN.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        print(f"[DynBG] {RAW_PATH.name} 없음 — bg_select 를 먼저 실행하세요.")
        return

    filled = cv2.imread(str(RAW_PATH))
    H, W = filled.shape[:2]
    total = H * W
    thr_px = cover_min * total                      # 선정 최소(전체 화면 %)

    yolo = YOLO(str(YOLO_MODEL))
    remaining = _object_mask(yolo, filled, dilate_px, mask_conf)  # 가려진 영역 = 원본의 객체 자리
    initial = int((remaining > 0).sum())
    if initial == 0:
        print("[DynBG] 가려진 영역 없음(객체 미검출). 종료.")
        return
    cv2.imwrite(str(FILLED_PATH), filled)
    cv2.imwrite(str(REMAINING_PATH), remaining)
    print(f"[DynBG] 시작 — 가려진 {initial}px ({initial/total*100:.1f}%) | "
          f"cover_min={cover_min*100:.2f}% upscale_steps={steps}")

    processed = -1
    sel = 0
    up_idx = 0
    last_up_remaining = initial

    try:
        while True:
            frames = sorted(WEBCAM.glob("frame_*.jpg"))
            new = [f for f in frames if int(f.stem.split("_")[1]) > processed]
            for f in new:
                processed = int(f.stem.split("_")[1])
                if int((remaining > 0).sum()) == 0:
                    break
                img = cv2.imread(str(f))
                if img is None:
                    continue
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H))

                obj       = _object_mask(yolo, img, dilate_px, mask_conf)   # 팽창 → 손가락/머리카락 잔상 제거
                revealed  = cv2.bitwise_not(obj)                 # 객체 없는(드러난) 영역
                new_cover = cv2.bitwise_and(remaining, revealed) # 새로 채울 수 있는 곳
                cover_px  = int((new_cover > 0).sum())

                if cover_px >= thr_px:                           # 전체의 cover_min% 이상이면 선정
                    sel_mask = new_cover > 0
                    filled[sel_mask] = img[sel_mask]             # 480p 합성 (픽셀 복사)
                    remaining[sel_mask] = 0
                    sel += 1
                    cv2.imwrite(str(FILLED_PATH), filled)
                    cv2.imwrite(str(REMAINING_PATH), remaining)
                    shutil.copy(f, CHOSEN / f.name)
                    cur = int((remaining > 0).sum())
                    print(f"[DynBG] frame {processed} 선정#{sel} | +{cover_px/total*100:.2f}% | "
                          f"남음 {cur/total*100:.1f}% ({cur/initial*100:.0f}% of init)")

                    # 점진 업스케일: 마지막 업스케일 이후 '남은 영역'이 step 만큼 줄면 ESRGAN
                    step = steps[min(up_idx, len(steps) - 1)]
                    if last_up_remaining - cur >= step * last_up_remaining:
                        _upscale(FILLED_PATH, OUT_BG, use_esrgan)
                        _rerun_depth(OUT_BG)
                        up_idx += 1
                        last_up_remaining = cur

            if int((remaining > 0).sum()) == 0:
                print("[DynBG] 가려진 영역 모두 채움 — 최종 업스케일")
                _upscale(FILLED_PATH, OUT_BG, use_esrgan)
                _rerun_depth(OUT_BG)
                break

            time.sleep(poll)   # 새 프레임 대기 (영원히 — Ctrl+C 또는 완료까지)

    except KeyboardInterrupt:
        print(f"\n[DynBG] 종료 — {sel}프레임 선정, 최종 업스케일")
        if FILLED_PATH.exists():
            _upscale(FILLED_PATH, OUT_BG, use_esrgan)
            _rerun_depth(OUT_BG)


if __name__ == "__main__":
    main()
