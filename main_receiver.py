import sys
sys.path.insert(0, "src/transmission")

from receiver import radar_receive, webcam_receive
import threading

if __name__ == "__main__":
    t1 = threading.Thread(target=radar_receive, daemon=True)
    t2 = threading.Thread(target=webcam_receive)

    t1.start()
    t2.start()
    t2.join()
