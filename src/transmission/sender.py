import socket
import cv2
import threading
import struct
import time
import datetime
import json
import subprocess
import serial
import os
from pathlib import Path

DESKTOP_IP  = "100.126.220.19"
RADAR_PORT  = 5006
WEBCAM_PORT = 5007
CHUNK_SIZE  = 1100

DCA_DATA_IP   = "192.168.33.30"
DCA_DATA_PORT = 4098

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent


def load_config():
    path = PROJECT_ROOT / "config" / "sender.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"webcam_fps": 10, "webcam_quality": 20, "radar_frame_period_ms": 1000}
DCA_ROOT       = PROJECT_ROOT / "tools" / "dca1000"
CLI_EXE        = DCA_ROOT / "DCA1000EVM_CLI_Control.exe"
CLI_RECORD_EXE = DCA_ROOT / "DCA1000EVM_CLI_Record.exe"
CLI_CONFIG     = str(DCA_ROOT / "chirp_configs" / "datacard_config.json")
RADAR_CFG      = DCA_ROOT / "config" / "awr1642_raw_data.cfg"

CLI_PORT = "COM3"
CLI_BAUD = 115200

RESTART_AT_SEQ = 70000


def _ts_ms():
    """자정 기준 밀리초 (uint32, 4바이트)."""
    n = datetime.datetime.now()
    return (n.hour * 3600 + n.minute * 60 + n.second) * 1000 + n.microsecond // 1000


record_proc_lock = threading.Lock()
current_record_proc = None
restart_event = threading.Event()


def dca_cli(cmd):
    print(f"[DCA] {cmd} 실행 중...")
    result = subprocess.run(
        [str(CLI_EXE), cmd, CLI_CONFIG],
        capture_output=True, text=True,
        cwd=str(DCA_ROOT)
    )
    output = (result.stdout + result.stderr).strip()
    print(f"[DCA] {cmd} 완료: {output}")
    if result.returncode != 0:
        raise RuntimeError(f"DCA CLI '{cmd}' 실패 (returncode={result.returncode})")

def dca_cli_background(cmd):
    print(f"[DCA] {cmd} 실행 중... (백그라운드)")
    return subprocess.Popen(
        [str(CLI_RECORD_EXE), cmd, CLI_CONFIG],
        cwd=str(DCA_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )


def monitor_record_proc(proc):
    for line in proc.stdout:
        line = line.strip()
        if line:
            print(f"[Record] {line}")
        if "Record is completed" in line or "Timeout Error" in line:
            print("[Record] 종료 감지 → 즉시 재시작 트리거")
            restart_event.set()


def uart_send_commands(commands):
    with serial.Serial(CLI_PORT, CLI_BAUD, timeout=1) as ser:
        for command in commands:
            ser.write((command + '\r\n').encode('utf-8'))
            time.sleep(0.05)
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"  Send: {command:<30} | {response.strip()}")


def send_radar_config(frame_period_ms=1000):
    print(f"[UART] {CLI_PORT} 연결 중...")
    with serial.Serial(CLI_PORT, CLI_BAUD, timeout=1) as ser:
        print(f"[UART] 연결 완료. .cfg 전송 시작... (frame period: {frame_period_ms}ms)")
        with open(RADAR_CFG, 'r') as f:
            lines = f.readlines()
        for line in lines:
            command = line.strip()
            if command.startswith('%') or not command:
                continue
            if command.startswith('frameCfg'):
                parts = command.split()
                parts[5] = str(frame_period_ms)
                command = ' '.join(parts)
            ser.write((command + '\r\n').encode('utf-8'))
            time.sleep(0.05)
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"  Send: {command:<30} | {response.strip()}")
    print("[UART] 레이더 설정 완료.")


def start_record_with_monitor():
    global current_record_proc
    proc = dca_cli_background("start_record")
    with record_proc_lock:
        current_record_proc = proc
    t = threading.Thread(target=monitor_record_proc, args=(proc,), daemon=True)
    t.start()


def restart_radar():
    global current_record_proc
    restart_event.clear()
    print("[Radar] 재시작 중...")

    with record_proc_lock:
        if current_record_proc is not None:
            current_record_proc.terminate()
            current_record_proc.wait()
            current_record_proc = None

    try:
        dca_cli("stop_record")
    except RuntimeError:
        pass

    print("[UART] sensorStop → sensorStart 0 전송 중...")
    uart_send_commands(["sensorStop", "sensorStart 0"])

    start_record_with_monitor()
    print("[Radar] 재시작 완료.")


def radar_forward():
    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.bind((DCA_DATA_IP, DCA_DATA_PORT))
    data_sock.settimeout(0.5)

    fwd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[Radar] 수신 대기 중 → {DESKTOP_IP}:{RADAR_PORT}")
    seq = 0
    try:
        while True:
            if restart_event.is_set():
                print("[Radar] 재시작 이벤트 감지. 즉시 재시작...")
                data_sock.close()
                restart_radar()
                data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                data_sock.bind((DCA_DATA_IP, DCA_DATA_PORT))
                data_sock.settimeout(0.5)
                seq = 0
                continue

            try:
                chunk, _ = data_sock.recvfrom(65535)
            except socket.timeout:
                continue

            header = struct.pack('>II', seq, _ts_ms())
            fwd_sock.sendto(header + chunk, (DESKTOP_IP, RADAR_PORT))
            print(f"[Radar] seq={seq:>6}  size={len(chunk):>5}B  {chunk[:16].hex()}")
            seq += 1

            if seq == RESTART_AT_SEQ:
                print(f"[Radar] seq {RESTART_AT_SEQ} 도달. 선제적 재시작...")
                data_sock.close()
                restart_radar()
                data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                data_sock.bind((DCA_DATA_IP, DCA_DATA_PORT))
                data_sock.settimeout(0.5)
                seq = 0
    finally:
        data_sock.close()
        fwd_sock.close()


def webcam_send(fps=10, quality=20):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[Webcam] 카메라 열기 실패")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame_id = 0
    interval = 1 / fps
    print(f"[Webcam] 전송 시작 → {DESKTOP_IP}:{WEBCAM_PORT}  ({fps}fps, quality={quality})")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        data = buf.tobytes()

        chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
        total = len(chunks)

        ts = _ts_ms()
        for i, chunk in enumerate(chunks):
            header = struct.pack('>IHHI', frame_id % 65536, i, total, ts)
            sock.sendto(header + chunk, (DESKTOP_IP, WEBCAM_PORT))

        frame_id += 1
        time.sleep(interval)

    cap.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, choices=[0, 1],
                        help="0: 웹캠만  1: 레이더 + 웹캠")
    args = parser.parse_args()

    cfg = load_config()
    print(f"[Config] webcam {cfg['webcam_fps']}fps / quality {cfg['webcam_quality']} / radar {cfg['radar_frame_period_ms']}ms")

    threads = []

    if args.mode == 1:
        dca_cli("fpga")
        dca_cli("record")
        send_radar_config(frame_period_ms=cfg["radar_frame_period_ms"])
        start_record_with_monitor()
        threads.append(threading.Thread(target=radar_forward, daemon=True))

    threads.append(threading.Thread(target=webcam_send,
                                    args=(cfg["webcam_fps"], cfg["webcam_quality"]),
                                    daemon=True))

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
            with record_proc_lock:
                if current_record_proc is not None:
                    current_record_proc.terminate()
            try:
                dca_cli("stop_record")
            except RuntimeError:
                pass
        print("종료.")
