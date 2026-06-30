"""
렌즈 보정 웹 서버
python tools/calib_server.py
→ http://localhost:8765/tools/calib_web.html
"""
import json
import threading
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH   = PROJECT_ROOT / "config" / "calib.json"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "scene" / "background_undist.jpg"
PORT         = 8765


class CalibHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_POST(self):
        if self.path == "/api/calib/save":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            calib  = json.loads(body)

            CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
            CALIB_PATH.write_text(json.dumps(calib, indent=2))

            try:
                import cv2, numpy as np
                img = cv2.imread(str(PROJECT_ROOT / "data" / "scene" / "background.jpg"))
                if img is not None:
                    w, h = calib["image_size"]
                    fx = calib["fx"]
                    K    = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], np.float64)
                    dist = np.array([calib["k1"], calib["k2"], 0, 0, 0], np.float64)
                    cv2.imwrite(str(OUTPUT_PATH), cv2.undistort(img, K, dist))
            except Exception as e:
                print(f"[경고] undistort 저장 실패: {e}")

            msg = f"저장 완료 → config/calib.json"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(msg.encode())
            print(f"[저장] {calib}")
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), CalibHandler)
    url    = f"http://localhost:{PORT}/tools/calib_web.html"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"[서버] {url}")
    print("[서버] Ctrl+C로 종료")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
