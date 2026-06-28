"""웹캠 인덱스별 확인 — 노트북에서 실행.

usage: python tools/webcam_probe.py

MSMF 백엔드로 index 0~3 을 열어 해상도/fps 측정 + 첫 프레임을
tools/probe_idx{N}.jpg 로 저장한다. 저장된 이미지를 보고 외장 웹캠의
index 를 확인한 뒤, config/sender.toml 의 [camera] device_index 를 그 번호로.
(DSHOW 와 MSMF 는 장치 열거 순서가 다르므로 MSMF 기준으로 찾아야 함)
"""
import cv2
import time
from pathlib import Path

W, H    = 1920, 1080
SECS    = 2.0
OUT_DIR = Path(__file__).resolve().parent


def probe(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"index {idx}: 없음")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(3):
        cap.read()

    n  = 0
    t0 = time.time()
    last = None
    while time.time() - t0 < SECS:
        ret, frame = cap.read()
        if ret:
            n += 1
            last = frame
    dt = time.time() - t0

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if last is not None:
        out = OUT_DIR / f"probe_idx{idx}.jpg"
        cv2.imwrite(str(out), last)
        print(f"index {idx}: {n/dt:5.1f} fps | {aw}x{ah} | 저장 {out.name}")
    else:
        print(f"index {idx}: 열렸으나 프레임 못 읽음")
    cap.release()


if __name__ == "__main__":
    print(f"=== 웹캠 index probe (MSMF, {W}x{H}) ===")
    for idx in range(4):
        try:
            probe(idx)
        except Exception as e:
            print(f"index {idx}: 오류 {e}")
        time.sleep(0.3)
    print("\n→ tools/probe_idx*.jpg 를 열어 외장 웹캠 번호 확인 후")
    print("  config/sender.toml 의 [camera] device_index 를 그 번호로 바꾸세요.")
