import sys
sys.path.insert(0, "src/transmission")

import argparse
import threading
import time
import sender

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, choices=[0, 1],
                        help="0: 웹캠만  1: 레이더 + 웹캠")
    args = parser.parse_args()

    threads = []

    if args.mode == 1:
        sender.dca_cli("fpga")
        sender.dca_cli("record")
        sender.send_radar_config()
        sender.start_record_with_monitor()
        threads.append(threading.Thread(target=sender.radar_forward, daemon=True))

    threads.append(threading.Thread(target=sender.webcam_send, daemon=True))

    for t in threads:
        t.start()

    mode_str = "웹캠만" if args.mode == 0 else "레이더 + 웹캠"
    print(f"실행 중 [{mode_str}]. Ctrl+C로 종료.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n종료 중...")
        if args.mode == 1:
            with sender.record_proc_lock:
                if sender.current_record_proc is not None:
                    sender.current_record_proc.terminate()
            try:
                sender.dca_cli("stop_record")
            except RuntimeError:
                pass
        print("종료.")
