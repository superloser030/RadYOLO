import cv2
import socket
import struct

TARGET_IP = "100.126.220.19"  # fordev-kjh (데스크톱)
TARGET_PORT = 5007
JPEG_QUALITY = 80

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠 열기 실패")
    exit(1)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TARGET_IP, TARGET_PORT))
print(f"연결됨: {TARGET_IP}:{TARGET_PORT}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        data = buf.tobytes()

        # 4바이트 길이 헤더 + 프레임 데이터
        sock.sendall(struct.pack('>I', len(data)) + data)

except KeyboardInterrupt:
    print("\n종료")
finally:
    cap.release()
    sock.close()
