import sys
sys.path.insert(0, "src/transmission")

import threading
import time
import receiver
from receiver import radar_receive, webcam_receive

if __name__ == "__main__":
    t1 = threading.Thread(target=radar_receive, daemon=True)
    t2 = threading.Thread(target=webcam_receive, daemon=True)

    t1.start()
    t2.start()

    try:
        while not receiver.shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료 중...")
        receiver.shutdown_event.set()

    t1.join(timeout=2)
    t2.join(timeout=2)
    print("종료.")
