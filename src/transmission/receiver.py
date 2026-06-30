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

from src.utils.config import load_network, load_receiver

_net = load_network()
RADAR_PORT  = _net["radar_port"]
WEBCAM_PORT = _net["webcam_port"]
META_PORT   = _net["meta_port"]

shutdown_event     = threading.Event()
_chirp_ready       = threading.Event()
frame_queue        = queue.Queue(maxsize=30)
_latest_frame      = None
_latest_frame_ts   = 0
_latest_frame_lock = threading.Lock()

_latest_radar      = None
_latest_radar_lock = threading.Lock()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


_cfg = load_receiver()
pipeline_delay_ms: int = _cfg["pipeline"]["delay_ms"]
_MATLAB_PORT: int      = _net["matlab_port"]

_SAVE_WEBCAM = _cfg["storage"]["save_webcam"]
_SAVE_RADAR  = _cfg["storage"]["save_radar"]
_SHOW_LIVE   = _cfg["storage"]["show_live"]

_matlab_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

_DEFAULT_CHIRP  = _cfg["radar"]["default_chirp"]
_BIN_FRAME_SIZE = 256 * 4 * _DEFAULT_CHIRP * 4
_BIN_FILE_MAX   = 100 * 1024 * 1024
_DCA_HDR_SIZE   = 10
_RADAR_DIR      = _PROJECT_ROOT / "data" / "radar"
_WEBCAM_DIR     = _PROJECT_ROOT / "data" / "webcam"

_adc_buf        = bytearray()
_bin_frame_idx  = 0
_bin_file_idx   = 0
_bin_file       = None
_ts_file        = None
_current_ts_ms  = 0
_bin_lock       = threading.Lock()

_webcam_frame_idx = 0
_webcam_ts_file   = None
_webcam_lock      = threading.Lock()


def _write_webcam_frame(jpeg_bytes: bytes, ts_ms: int):
    """수신한 JPEG 바이트를 재인코딩 없이 그대로 저장 + 타임스탬프 기록."""
    global _webcam_frame_idx, _webcam_ts_file
    _WEBCAM_DIR.mkdir(parents=True, exist_ok=True)
    if _webcam_ts_file is None:
        _webcam_ts_file = open(_WEBCAM_DIR / "webcam_timestamps.csv", "w")
        _webcam_ts_file.write("frame_idx,ts_ms\n")
    path = _WEBCAM_DIR / f"frame_{_webcam_frame_idx:06d}.jpg"
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    _webcam_ts_file.write(f"{_webcam_frame_idx},{ts_ms}\n")
    _webcam_ts_file.flush()
    _webcam_frame_idx += 1


def _write_bin_frame(frame_bytes: bytes, ts_ms: int):
    global _bin_file, _bin_file_idx, _bin_frame_idx, _ts_file
    _RADAR_DIR.mkdir(parents=True, exist_ok=True)
    if _ts_file is None:
        _ts_file = open(_RADAR_DIR / "iqData_timestamps.csv", "w")
        _ts_file.write("frame_idx,ts_ms\n")
    if _bin_file is None or _bin_file.tell() >= _BIN_FILE_MAX:
        if _bin_file:
            _bin_file.close()
        path = _RADAR_DIR / f"iqData_Raw_{_bin_file_idx}.bin"
        _bin_file = open(path, "ab")
        _bin_file_idx += 1
    _bin_file.write(frame_bytes)
    _bin_file.flush()
    _ts_file.write(f"{_bin_frame_idx},{ts_ms}\n")
    _ts_file.flush()
    _bin_frame_idx += 1


def close_bin_file():
    global _bin_file, _ts_file, _webcam_ts_file
    if _bin_file:
        _bin_file.close()
        _bin_file = None
    if _ts_file:
        _ts_file.close()
        _ts_file = None
    if _webcam_ts_file:
        _webcam_ts_file.close()
        _webcam_ts_file = None


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

        _matlab_sock.sendto(payload, ("127.0.0.1", _MATLAB_PORT))

        if _SAVE_RADAR and _chirp_ready.is_set():
            adc_bytes = payload[_DCA_HDR_SIZE:]
            with _bin_lock:
                global _current_ts_ms
                _current_ts_ms = ts_ms
                _adc_buf.extend(adc_bytes)
                while len(_adc_buf) >= _BIN_FRAME_SIZE:
                    frame = bytes(_adc_buf[:_BIN_FRAME_SIZE])
                    del _adc_buf[:_BIN_FRAME_SIZE]
                    _write_bin_frame(frame, _current_ts_ms)

    sock.close()


def webcam_receive():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", WEBCAM_PORT))
    sock.settimeout(0.5)
    print(f"[Webcam] 수신 대기 중 0.0.0.0:{WEBCAM_PORT}")

    frames    = defaultdict(dict)
    frame_ts  = {}

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
                if _SAVE_WEBCAM:
                    with _webcam_lock:
                        _write_webcam_frame(frame_data, ts)

                global _latest_frame, _latest_frame_ts
                with _latest_frame_lock:
                    _latest_frame    = frame
                    _latest_frame_ts = ts
                if not frame_queue.full():
                    frame_queue.put((frame, ts))
                if _SHOW_LIVE:
                    cv2.imshow('Webcam Live', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shutdown_event.set()
                        break

    sock.close()
    cv2.destroyAllWindows()


def set_chirp(num_chirps: int, samples_per_chirp: int = 256, num_receivers: int = 4):
    """chirp 수 변경에 맞춰 .bin 프레임 크기 갱신 (samples×rx×chirp×4byte)."""
    global _BIN_FRAME_SIZE
    _BIN_FRAME_SIZE = samples_per_chirp * num_receivers * num_chirps * 4
    _chirp_ready.set()
    print(f"[Meta] chirp={num_chirps} → bin frame size={_BIN_FRAME_SIZE}B")


def _write_mat(meta: dict):
    """센더 메타로 iqData_RecordingParameters.mat 생성 (MATLAB dca1000FileReader 용)."""
    import scipy.io as sio
    _RADAR_DIR.mkdir(parents=True, exist_ok=True)
    rp = {
        "ADCSampleRate":   float(meta["ADCSampleRate"]),
        "SweepSlope":      float(meta["SweepSlope"]),
        "SamplesPerChirp": float(meta["SamplesPerChirp"]),
        "CenterFrequency": float(meta["CenterFrequency"]),
        "ChirpCycleTime":  float(meta["ChirpCycleTime"]),
        "NumReceivers":    float(meta["NumReceivers"]),
        "NumChirps":       float(meta["NumChirps"]),
    }
    path = _RADAR_DIR / "iqData_RecordingParameters.mat"
    sio.savemat(str(path), {"RecordingParameters": rp})
    print(f"[Meta] .mat 생성: {path.name}  NumChirps={rp['NumChirps']:.0f}")


_last_meta = None


def meta_receive():
    """센더가 보낸 레벨 메타(TCP)를 받아 .mat 생성 + .bin 프레임 크기 갱신.

    센더는 메타를 주기적으로 재전송하므로, 값이 직전과 같으면 무시한다.
    """
    global _last_meta
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", META_PORT))
    sock.listen(1)
    sock.settimeout(0.5)
    print(f"[Meta] 수신 대기 중 0.0.0.0:{META_PORT}")

    while not shutdown_event.is_set():
        try:
            conn, _ = sock.accept()
        except socket.timeout:
            continue
        with conn:
            conn.settimeout(2.0)
            buf = b""
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            except socket.timeout:
                pass
        if not buf:
            continue
        try:
            meta = json.loads(buf.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            print(f"[Meta] 파싱 실패: {e}")
            continue
        if meta == _last_meta:
            continue
        _last_meta = meta
        set_chirp(int(meta["NumChirps"]),
                  int(meta["SamplesPerChirp"]),
                  int(meta["NumReceivers"]))
        _write_mat(meta)

    sock.close()


