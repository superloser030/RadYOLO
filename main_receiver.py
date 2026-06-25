import sys
sys.path.insert(0, "src/transmission")

from receiver import radar_receive, webcam_receive
import threading
import time
import cv2

if __name__ == "__main__":
    t1 = threading.Thread(target=radar_receive, daemon=True)
    t2 = threading.Thread(target=webcam_receive, daemon=True)

    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("종료.")
