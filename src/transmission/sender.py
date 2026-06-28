import socket
import cv2
import threading
import struct
import time
import datetime
import json
import subprocess
from pathlib import Path
# serial(pyserial)은 레이더(UART) 송신에만 필요 → 함수 내부에서 지연 import
# (웹캠 전용 모드/데스크톱에서는 미설치여도 동작)

from src.utils.config import load_network, load_sender, resolve_level

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── config 로드 ──────────────────────────────────
_net    = load_network()
_sender = load_sender()
_dca    = _sender["dca"]
_cam    = _sender["camera"]
_radar  = _sender["radar"]

DESKTOP_IP  = _net["desktop_ip"]
RADAR_PORT  = _net["radar_port"]
WEBCAM_PORT = _net["webcam_port"]
META_PORT   = _net["meta_port"]
CHUNK_SIZE  = 1100

DCA_DATA_IP    = _dca["data_ip"]
DCA_DATA_PORT  = _dca["data_port"]
CLI_PORT       = _dca["cli_port"]
CLI_BAUD       = _dca["cli_baud"]
RESTART_AT_SEQ = _dca["restart_at_seq"]

DCA_ROOT       = PROJECT_ROOT / "tools" / "dca1000"
CLI_EXE        = DCA_ROOT / "DCA1000EVM_CLI_Control.exe"
CLI_RECORD_EXE = DCA_ROOT / "DCA1000EVM_CLI_Record.exe"
CLI_CONFIG     = str(DCA_ROOT / "chirp_configs" / "datacard_config.json")
RADAR_CFG      = DCA_ROOT / "config" / "awr1642_raw_data.cfg"


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
    import serial
    with serial.Serial(CLI_PORT, CLI_BAUD, timeout=1) as ser:
        for command in commands:
            ser.write((command + '\r\n').encode('utf-8'))
            time.sleep(0.05)
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"  Send: {command:<30} | {response.strip()}")


def send_radar_config(num_loops, frame_period_01ms):
    """레벨에 맞춰 .cfg 의 frameCfg(numLoops, framePeriodicity)를 동적 수정 후 UART 전송.

    frameCfg <chirpStartIdx> <chirpEndIdx> <numLoops> <numFrames>
             <framePeriodicity(0.1ms)> <triggerSelect> <frameTriggerDelay>
    """
    import serial
    print(f"[UART] {CLI_PORT} 연결 중...")
    with serial.Serial(CLI_PORT, CLI_BAUD, timeout=1) as ser:
        print(f"[UART] .cfg 전송 (numLoops={num_loops}, framePeriod={frame_period_01ms*0.1:.1f}ms)")
        with open(RADAR_CFG, 'r') as f:
            lines = f.readlines()
        for line in lines:
            command = line.strip()
            if command.startswith('%') or not command:
                continue
            if command.startswith('frameCfg'):
                parts = command.split()
                parts[3] = str(num_loops)          # numLoops
                parts[5] = str(frame_period_01ms)  # framePeriodicity (0.1ms 단위)
                command = ' '.join(parts)
            ser.write((command + '\r\n').encode('utf-8'))
            time.sleep(0.05)
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"  Send: {command:<30} | {response.strip()}")
    print("[UART] 레이더 설정 완료.")


def send_meta(level: dict):
    """레벨 메타를 데스크톱(META_PORT)으로 TCP 전송 → 데스크톱이 .mat 생성.

    데스크톱은 이 값으로 iqData_RecordingParameters.mat 의 NumChirps 와
    .bin 프레임 크기를 맞춘다.
    """
    meta = {
        "ADCSampleRate":   _radar["adc_sample_rate"],
        "SweepSlope":      _radar["sweep_slope"],
        "SamplesPerChirp": _radar["samples_per_chirp"],
        "CenterFrequency": _radar["center_freq_ghz"],
        "ChirpCycleTime":  _radar["chirp_cycle_us"],
        "NumReceivers":    _radar["num_receivers"],
        "NumChirps":       level["chirp"],
        "level":           level["level"],
    }
    try:
        with socket.create_connection((DESKTOP_IP, META_PORT), timeout=5) as s:
            s.sendall(json.dumps(meta).encode("utf-8"))
        print(f"[Meta] 전송 완료 → {DESKTOP_IP}:{META_PORT}  NumChirps={meta['NumChirps']}")
    except OSError as e:
        print(f"[Meta] 전송 실패 ({e}) — 데스크톱이 default_chirp 로 동작")


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


def webcam_send(fps, quality, width, height):
    cap = cv2.VideoCapture(_cam["device_index"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("[Webcam] 카메라 열기 실패")
        return

    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame_id = 0
    interval = 1 / fps
    print(f"[Webcam] 전송 시작 → {DESKTOP_IP}:{WEBCAM_PORT}  "
          f"({fps}fps, {act_w}x{act_h} 요청 {width}x{height}, quality={quality})")

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


def measure_bandwidth_mbps():
    """iperf3 -c 로 데스크톱까지 상행(uplink) 대역폭 측정. 실패 시 0.0 반환.

    데스크톱(main_r.py)이 iperf3 -s 를 상시 띄워두어야 함.
    """
    try:
        out = subprocess.run(
            ["iperf3", "-c", DESKTOP_IP, "-t", "3", "-J"],
            capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        print("[BW] iperf3 미설치 — mode 1 측정 불가")
        return 0.0
    except subprocess.TimeoutExpired:
        print("[BW] iperf3 타임아웃")
        return 0.0
    if out.returncode != 0:
        print(f"[BW] iperf3 실패: {out.stderr.strip()[:120]}")
        return 0.0
    try:
        data = json.loads(out.stdout)
        bps  = data["end"]["sum_sent"]["bits_per_second"]
        return bps / 1e6
    except (ValueError, KeyError) as e:
        print(f"[BW] iperf3 결과 파싱 실패: {e}")
        return 0.0


def select_level():
    """모드에 따라 사용할 레벨 dict 결정."""
    mode   = _sender["mode"]
    levels = _sender["level"]

    if mode["sender_mode"] == 0:
        return resolve_level(_sender, mode["fixed_level"])

    # mode 1: 자동 대역폭 — 측정 × target% 예산 이하인 최고 레벨 선택
    mbps = measure_bandwidth_mbps()
    if mbps <= 0:
        print("[Level] 대역폭 측정 실패 → 최소 레벨 1 사용")
        return resolve_level(_sender, 1)

    budget = mbps * mode["bw_target_pct"] / 100.0
    chosen = 1
    for i, lv in enumerate(levels, 1):
        if lv["est_mbps"] <= budget:
            chosen = i
    print(f"[Level] 측정 {mbps:.1f}Mbps × {mode['bw_target_pct']}% "
          f"= 예산 {budget:.1f}Mbps → 레벨 {chosen}")
    return resolve_level(_sender, chosen)


def run(transmit_mode: int):
    """송신 진입점.

    transmit_mode: 0 = 웹캠만, 1 = 레이더 + 웹캠
    """
    lv = select_level()
    print(f"[Level] {lv['level']}: chirp={lv['chirp']} fps={lv['fps']} "
          f"{lv['width']}x{lv['height']} q{lv['quality']}  (~{lv['est_mbps']}Mbps)")

    threads = []

    if transmit_mode == 1:
        send_meta(lv)                       # 데스크톱이 .mat 생성하도록 메타 먼저 전송
        dca_cli("fpga")
        dca_cli("record")
        send_radar_config(lv["num_loops"], lv["frame_period_01ms"])
        start_record_with_monitor()
        threads.append(threading.Thread(target=radar_forward, daemon=True))

    threads.append(threading.Thread(
        target=webcam_send,
        args=(lv["fps"], lv["quality"], lv["width"], lv["height"]),
        daemon=True))

    for t in threads:
        t.start()

    mode_str = "웹캠만" if transmit_mode == 0 else "레이더 + 웹캠"
    print(f"실행 중 [{mode_str}]. Ctrl+C로 종료.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n종료 중...")
        if transmit_mode == 1:
            with record_proc_lock:
                if current_record_proc is not None:
                    current_record_proc.terminate()
            try:
                dca_cli("stop_record")
            except RuntimeError:
                pass
        print("종료.")
