"""
GigaPose 래퍼 — 영구 추론 서버(inference_server.py) 방식
첫 호출 시 gigapose 환경에서 서버 프로세스를 시작하고 모델을 로드한다.
이후 요청은 stdin/stdout JSON 통신으로 처리 → 모델 재로드 없음.
"""
import json
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
GIGAPOSE_PY   = r"C:\Users\loser\anaconda3\envs\gigapose\python.exe"
SERVER_SCRIPT = PROJECT_ROOT / "tools" / "gigapose" / "inference_server.py"
RENDER_SCRIPT = PROJECT_ROOT / "tools" / "gigapose" / "prepare_templates.py"

_server       = None           # 단일 추론 서버 {"proc", "lock"} — 모델 1번만 로드
_server_lock  = threading.Lock()
_server_retry_at = 0.0        # 서버 재시작 가능 시각 (실패 시 60초 쿨다운)


# ── 템플릿 렌더링 ────────────────────────────────────────────────────

def prepare_templates(mesh_path: str, template_dir: str, level: int = 1):
    """GLB 메쉬 → 템플릿 이미지 렌더링 (이미 있으면 스킵)."""
    if (Path(template_dir) / "meta.json").exists():
        return
    cmd = [GIGAPOSE_PY, str(RENDER_SCRIPT),
           "--mesh", mesh_path, "--out", template_dir, "--level", str(level)]
    print(f"[Pose] 템플릿 렌더링: {mesh_path}")
    subprocess.run(cmd, check=True)
    print(f"[Pose] 렌더링 완료 → {template_dir}")


# ── 영구 서버 관리 ───────────────────────────────────────────────────

def _get_server(camera_k):
    """단일 추론 서버 시작/재사용. 모델은 1번만 로드, 템플릿은 서버가 요청별 캐시."""
    import time
    global _server, _server_retry_at
    with _server_lock:
        if _server and _server["proc"].poll() is None:
            return _server         # 살아있는 서버 재사용 (객체 수 무관)

        if time.time() < _server_retry_at:
            return None            # 60초 쿨다운 중 — 재시작 시도 안 함

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


# ── 포즈 추정 ────────────────────────────────────────────────────────

def estimate_pose(image_path: str, bbox, template_dir: str, camera_k):
    """
    6DoF 포즈 추정.
    bbox: (x1, y1, x2, y2)  camera_k: (fx, fy, cx, cy)
    Returns: {"R": 3x3, "t": [x,y,z], "score": float}  or  None
    """
    info = _get_server(camera_k)
    if info is None:
        return None

    req = json.dumps({"image": image_path, "bbox": list(map(float, bbox)),
                      "template": str(template_dir)})
    with info["lock"]:      # 동시 요청 직렬화
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
    """추론 서버 종료."""
    global _server
    with _server_lock:
        if _server:
            try:
                _server["proc"].terminate()
            except Exception:
                pass
            _server = None
