import serial 
import time
import os
config_path = os.path.join(os.path.dirname(__file__), 'awr1642_raw_data.cfg')

#CLI_PORT = '/dev/ttyACM0'
CLI_PORT = 'COM3'
CLI_BAUD = 115200

try:
    cli_serial = serial.Serial(CLI_PORT, CLI_BAUD, timeout=1)   
    print(f"[{CLI_PORT}] Connected!")

    with open(config_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        command = line.strip()
        if command.startswith('%') or not command:
            continue

        cli_serial.write((command + '\r\n').encode('utf-8'))
        time.sleep(0.05)

        response = cli_serial.read(cli_serial.in_waiting).decode('utf-8', errors='ignore')
        print(f"Send: {command:<30} | Response: {response.strip()}")

    print("Radar Configuration Completed!")

except Exception as e:
    print(f"error: {e}")
finally:
    if 'cli_serial' in locals() and cli_serial.is_open:
        cli_serial.close()
        