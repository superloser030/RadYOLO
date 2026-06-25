import socket
import cv2
import threading
import struct
import numpy as np
from collections import defaultdict
import time

RADAR_PORT  = 5006
WEBCAM_PORT = 5007

shutdown_event = threading.Event()


def radar_receive():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RADAR_PORT))
    sock.settimeout(0.5)
    print(f"[Radar] 수신 대기 중 0.0.0.0:{RADAR_PORT}")

    total_bytes = 0
    count = 0
    start = time.time()

    while not shutdown_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue

        seq = struct.unpack('>I', data[:4])[0]
        payload = data[4:]
        print(f"[Radar] seq={seq:>6}  size={len(payload):>5}B  {payload[:16].hex()}")
        count += 1
        total_bytes += len(data)

        if count % 500 == 0:
            elapsed = time.time() - start
            mbps = (total_bytes * 8) / elapsed / 1e6
            print(f"[Radar] ── {count}개 패킷 | {total_bytes/1024:.1f} KB | {mbps:.2f} Mbps ──")

    sock.close()


def webcam_receive():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", WEBCAM_PORT))
    sock.settimeout(0.5)
    print(f"[Webcam] 수신 대기 중 0.0.0.0:{WEBCAM_PORT}")

    frames = defaultdict(dict)

    while not shutdown_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue

        hdr_size = struct.calcsize('>IHH')
        frame_id, chunk_id, total = struct.unpack('>IHH', data[:hdr_size])
        frames[frame_id][chunk_id] = data[hdr_size:]

        if len(frames[frame_id]) == total:
            frame_data = b''.join(frames[frame_id][i] for i in range(total))

            for fid in [fid for fid in list(frames.keys()) if fid < frame_id - 5]:
                del frames[fid]
            del frames[frame_id]

            arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Webcam Live', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shutdown_event.set()
                    break

    sock.close()
    cv2.destroyAllWindows()
