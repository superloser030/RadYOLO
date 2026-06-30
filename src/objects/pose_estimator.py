import json
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
GIGAPOSE_PY   = r"C:\Users\loser\anaconda3\envs\gigapose\python.exe"
SERVER_SCRIPT = PROJECT_ROOT / "tools" / "gigapose" / "inference_server.py"
RENDER_SCRIPT = PROJECT_ROOT / "tools" / "gigapose" / "prepare_templates.py"

_server       = None
_server_lock  = threading.Lock()
_server_retry_at = 0.0



def prepare_templates(mesh_path: str, template_dir: str, level: int = 1):
    if (Path(template_dir) / "meta.json").exists():
        return
    cmd = [GIGAPOSE_PY, str(RENDER_SCRIPT),
           "--mesh", mesh_path, "--out", template_dir, "--level", str(level)]
    print(f"[Pose] 템플릿 렌더링: {mesh_path}")
    subprocess.run(cmd, check=True)
    print(f"[Pose] 렌더링 완료 → {template_dir}")



def _get_server(camera_k):
    import time
    global _server, _server_retry_at
    with _server_lock:
        if _server and _server["proc"].poll() is None:
            return _server

        if time.time() < _server_retry_at:
            return None

        k_str = ",".join(map(str, map(float, camera_k)))
        cmd   = [GIGAPOSE_PY, str(SERVER_SCRIPT), "--camera-k", k_str]
        proc  = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        print("[Pose] 추론 서버 시작 (모델 1회 로드)...")
        ready_line = proc.stdout.readline()
        try:
            if json.loads(ready_line).get("status") == "ready":
                print("[Pose] 추론 서버 준비 완료")
            else:
                raise ValueError(ready_line)
        except Exception as e:
            print(f"[Pose] 서버 시작 실패: {e}  (60초 쿨다운)")
            proc.kill()
            _server_retry_at = time.time() + 60
            return None

        _server = {"proc": proc, "lock": threading.Lock()}
        return _server



def estimate_pose(image_path: str, bbox, template_dir: str, camera_k):
    info = _get_server(camera_k)
    if info is None:
        return None

    req = json.dumps({"image": image_path, "bbox": list(map(float, bbox)),
                      "template": str(template_dir)})
    with info["lock"]:
        try:
            info["proc"].stdin.write(req + "\n")
            info["proc"].stdin.flush()
            line = info["proc"].stdout.readline()
            if not line:
                return None
            result = json.loads(line.strip())
            if "error" in result:
                print(f"[Pose] 추론 오류: {result['error']}")
                return None
            return result
        except Exception as e:
            print(f"[Pose] 통신 예외: {e}")
            return None


def shutdown_servers():
    global _server
    with _server_lock:
        if _server:
            try:
                _server["proc"].terminate()
            except Exception:
                pass
            _server = None
