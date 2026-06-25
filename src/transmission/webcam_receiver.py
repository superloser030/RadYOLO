import cv2
import socket
import struct
import numpy as np

PORT = 5007

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)
print(f"대기 중... (포트 {PORT})")

conn, addr = server.accept()
print(f"연결됨: {addr}")

def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

try:
    while True:
        raw_len = recv_exact(conn, 4)
        if raw_len is None:
            print("연결 끊김")
            break
        msg_len = struct.unpack('>I', raw_len)[0]

        data = recv_exact(conn, msg_len)
        if data is None:
            print("연결 끊김")
            break

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"디코드 실패 (데이터 크기: {len(data)})")
            continue
        print(f"프레임 수신: {frame.shape}, 데이터 {len(data)} bytes")
        cv2.imshow("webcam", frame)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\n종료")
finally:
    conn.close()
    server.close()
    cv2.destroyAllWindows()
