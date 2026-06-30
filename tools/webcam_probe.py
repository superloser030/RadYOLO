"""외장 웹캠(DSHOW idx0) 해상도별 fps probe — 노트북에서 실행.

usage: python tools/webcam_probe.py

외장 웹캠은 OpenCV 와 MJPG 협상이 안 돼 1080p 에서 YUY2 4fps 로 떨어진다.
해상도를 낮추면 YUY2 여도 fps 가 오르므로(USB 대역폭), 각 해상도/포맷의
실제 fps 를 측정해 sender 의 캡처 해상도를 결정한다.
"""
import cv2
import time
from pathlib import Path

IDX     = 0
SECS    = 2.0
OUT_DIR = Path(__file__).resolve().parent

RES = [(1920, 1080), (1280, 720), (854, 480)]
FCC = ["MJPG", "YUY2"]


def test(fcc, w, h):
    cap = cv2.VideoCapture(IDX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"{fcc} {w}x{h}: 열기 실패")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
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
    dt = time.time() - t0

    aw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got = int(cap.get(cv2.CAP_PROP_FOURCC))
    gs  = "".join(chr((got >> 8 * i) & 0xFF) for i in range(4))
    tag = f"{fcc}_{w}x{h}"
    if last is not None:
        p = OUT_DIR / f"probe_{tag}.jpg"
        cv2.imwrite(str(p), last)
        print(f"{fcc} 요청 {w}x{h:<4} → {n/dt:5.1f} fps | 실제 {aw}x{ah} fmt={gs} | {p.name}")
    else:
        print(f"{fcc} {w}x{h}: 프레임 못 읽음")
    cap.release()


if __name__ == "__main__":
    print(f"=== 외장(DSHOW idx{IDX}) 해상도별 probe ===")
    for fcc in FCC:
        for (w, h) in RES:
            try:
                test(fcc, w, h)
            except Exception as e:
                print(f"{fcc} {w}x{h}: 오류 {e}")
            time.sleep(0.3)
    print("\n→ fps 가 쓸 만한(>=10) 가장 높은 해상도를 골라 sender.toml 레벨 해상도에 반영.")
    print("  probe_*.jpg 로 화질도 확인. (ESRGAN 업스케일하므로 720p 면 충분)")
