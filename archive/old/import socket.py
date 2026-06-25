import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 4098))
print("수신 대기 중...")
while True:
    data, addr = sock.recvfrom(65535)
    print(f"수신: {addr} | {len(data)} bytes")