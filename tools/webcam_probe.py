"""웹캠 종합 probe — 노트북에서 실행.

usage: python tools/webcam_probe.py

DSHOW(idx 0~3) + MSMF(idx 0~1) 를 열어 fps/해상도 측정 + 첫 프레임을
tools/probe_{backend}_idx{N}.jpg 로 저장. 저장 이미지로 외장 웹캠이 어느
backend/index 에서 보이는지 확인 후 sender.webcam_send 에 그 조합을 고정.
(MSMF 는 없는 index 를 열면 hang 하므로 0~1 만 시도)
"""
import cv2
import time
from pathlib import Path

W, H    = 1920, 1080
SECS    = 1.5
OUT_DIR = Path(__file__).resolve().parent


def test(bname, backend, idx):
    # open 타임아웃 3초 — MSMF 가 없는 index 에서 무한 대기(hang)하는 것 방지
    cap = cv2.VideoCapture(idx, backend, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000])
    if not cap.isOpened():
        print(f"{bname:5} idx{idx}: 없음")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(3):
        cap.read()

    n = 0
    t0 = time.time()
    last = None
    while time.time() - t0 < SECS:
        ret, frame = cap.read()
        if ret:
            n += 1
            last = frame

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if last is not None:
        p = OUT_DIR / f"probe_{bname}_idx{idx}.jpg"
        cv2.imwrite(str(p), last)
        print(f"{bname:5} idx{idx}: {n/SECS:5.1f} fps | {aw}x{ah} | 저장 {p.name}")
    else:
        print(f"{bname:5} idx{idx}: 열렸으나 프레임 못 읽음")
    cap.release()


if __name__ == "__main__":
    print(f"=== 웹캠 종합 probe ({W}x{H}) ===")
    for idx in range(4):
        try:
            test("DSHOW", cv2.CAP_DSHOW, idx)
        except Exception as e:
            print(f"DSHOW idx{idx}: 오류 {e}")
        time.sleep(0.3)
    for idx in range(5):
        try:
            test("MSMF", cv2.CAP_MSMF, idx)
        except Exception as e:
            print(f"MSMF  idx{idx}: 오류 {e}")
        time.sleep(0.3)
    print("\n→ probe_*.jpg 들을 열어 '외장'이 찍힌 backend/index 확인.")
    print("  그 조합을 sender.webcam_send 에 적용 (backend 와 device_index).")
