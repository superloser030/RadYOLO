"""웹캠 백엔드 x 포맷 조합별 실제 fps 측정 — 노트북에서 실행.

usage: python tools/webcam_probe.py

1080p 에서 어떤 조합이 가장 빠른지 확인한 뒤, 그 값을 sender.webcam_send 에
고정한다. (예: MSMF + MJPG 가 30fps 나오면 그걸로)
"""
import cv2
import time

DEVICE = 0
W, H   = 1920, 1080
SECS   = 3.0

BACKENDS = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
FOURCCS  = ["MJPG", "YUY2"]


def probe(backend, bname, fcc):
    cap = cv2.VideoCapture(DEVICE, backend)
    if not cap.isOpened():
        print(f"{bname:6} {fcc}: 열기 실패")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 워밍업 (첫 프레임 초기화 지연 제외)
    for _ in range(3):
        cap.read()

    n  = 0
    t0 = time.time()
    while time.time() - t0 < SECS:
        ret, _ = cap.read()
        if ret:
            n += 1
    dt = time.time() - t0

    got = int(cap.get(cv2.CAP_PROP_FOURCC))
    gs  = "".join(chr((got >> 8 * i) & 0xFF) for i in range(4))
    aw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{bname:6} {fcc} 요청 → {n/dt:5.1f} fps | 실제 fmt={gs} | {aw}x{ah}")
    cap.release()


if __name__ == "__main__":
    print(f"=== 웹캠 probe ({W}x{H}, 각 {SECS:.0f}초) ===")
    for backend, bname in BACKENDS:
        for fcc in FOURCCS:
            try:
                probe(backend, bname, fcc)
            except Exception as e:
                print(f"{bname:6} {fcc}: 오류 {e}")
            time.sleep(0.5)
    print("\n→ 가장 높은 fps 조합(특히 fmt=MJPG)을 sender.webcam_send 에 적용하면 됩니다.")
