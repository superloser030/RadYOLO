import socket

UDP_IP = "0.0.0.0"      # 모든 인터페이스에서 수신
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT} ...")

count = 0
try:
    while True:
        data, addr = sock.recvfrom(1024)
        count += 1
        print(f"[{count}] Received from {addr}: {data.decode()}")
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    sock.close()