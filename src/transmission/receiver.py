import socket
import cv2
import threading
import struct
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import queue
import time

RADAR_PORT  = 5006
WEBCAM_PORT = 5007

shutdown_event     = threading.Event()
frame_queue        = queue.Queue(maxsize=30)
_latest_frame      = None
_latest_frame_ts   = 0   # ms since midnight (sender 기준)
_latest_frame_lock = threading.Lock()

_latest_radar      = None   # (payload_bytes, ts_ms)
_latest_radar_lock = threading.Lock()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config() -> dict:
    path = _PROJECT_ROOT / "config" / "receiver.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"pipeline_delay_ms": 700, "matlab_radar_port": 5009}


_cfg = load_config()
pipeline_delay_ms: int = _cfg["pipeline_delay_ms"]
_MATLAB_PORT: int      = _cfg["matlab_radar_port"]

_matlab_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def _ms_to_timestr(ms: int) -> str:
    h   = ms // 3600000
    m   = (ms % 3600000) // 60000
    s   = (ms % 60000)   // 1000
    ms3 = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms3:03d}"


def get_latest_frame():
    """라이브 루프용: 가장 최근 수신 프레임 반환 (None이면 아직 수신 안됨)"""
    with _latest_frame_lock:
        return _latest_frame


def get_latest_frame_ts() -> int:
    """가장 최근 웹캠 프레임의 타임스탬프 (ms since midnight, sender 시계 기준)"""
    with _latest_frame_lock:
        return _latest_frame_ts


def get_latest_radar():
    """가장 최근 레이더 패킷: (payload_bytes, ts_ms) 또는 None"""
    with _latest_radar_lock:
        return _latest_radar


def radar_receive():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RADAR_PORT))
    sock.settimeout(0.5)
    print(f"[Radar] 수신 대기 중 0.0.0.0:{RADAR_PORT} → MATLAB port {_MATLAB_PORT}")

    total_bytes = 0
    count = 0
    start = time.time()

    while not shutdown_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue

        hdr_size = struct.calcsize('>II')
        seq, ts_ms = struct.unpack('>II', data[:hdr_size])
        payload = data[hdr_size:]

        global _latest_radar
        with _latest_radar_lock:
            _latest_radar = (payload, ts_ms)

        # 커스텀 헤더 제거 후 원본 DCA1000 패킷을 MATLAB으로 포워딩
        _matlab_sock.sendto(payload, ("127.0.0.1", _MATLAB_PORT))

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

    frames    = defaultdict(dict)
    frame_ts  = {}   # frame_id → ts_ms

    while not shutdown_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue

        hdr_size = struct.calcsize('>IHHI')
        frame_id, chunk_id, total, ts_ms = struct.unpack('>IHHI', data[:hdr_size])
        frames[frame_id][chunk_id] = data[hdr_size:]
        frame_ts[frame_id] = ts_ms

        if len(frames[frame_id]) == total:
            frame_data = b''.join(frames[frame_id][i] for i in range(total))
            ts = frame_ts.pop(frame_id, 0)

            for fid in [fid for fid in list(frames.keys()) if fid < frame_id - 5]:
                frames.pop(fid, None)
                frame_ts.pop(fid, None)
            del frames[frame_id]

            arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                global _latest_frame, _latest_frame_ts
                with _latest_frame_lock:
                    _latest_frame    = frame
                    _latest_frame_ts = ts
                if not frame_queue.full():
                    frame_queue.put(frame)
                cv2.imshow('Webcam Live', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shutdown_event.set()
                    break

    sock.close()
    cv2.destroyAllWindows()
