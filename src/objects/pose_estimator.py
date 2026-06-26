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

_servers      = {}             # template_dir → {"proc", "lock"}
_servers_lock = threading.Lock()


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

def _get_server(template_dir: str, camera_k):
    """서버 프로세스가 없거나 죽어있으면 새로 시작해서 반환."""
    key = template_dir
    with _servers_lock:
        info = _servers.get(key)
        if info and info["proc"].poll() is None:
            return info            # 살아있는 서버 재사용

        # 시작
        k_str = ",".join(map(str, map(float, camera_k)))
        cmd   = [GIGAPOSE_PY, str(SERVER_SCRIPT),
                 "--template", template_dir, "--camera-k", k_str]
        proc  = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        print("[Pose] 추론 서버 시작 (모델 로드 중)...")
        ready_line = proc.stdout.readline()
        try:
            if json.loads(ready_line).get("status") == "ready":
                print("[Pose] 추론 서버 준비 완료")
            else:
                raise ValueError(ready_line)
        except Exception as e:
            print(f"[Pose] 서버 시작 실패: {e}")
            proc.kill()
            return None

        info = {"proc": proc, "lock": threading.Lock()}
        _servers[key] = info
        return info


# ── 포즈 추정 ────────────────────────────────────────────────────────

def estimate_pose(image_path: str, bbox, template_dir: str, camera_k):
    """
    6DoF 포즈 추정.
    bbox: (x1, y1, x2, y2)  camera_k: (fx, fy, cx, cy)
    Returns: {"R": 3x3, "t": [x,y,z], "score": float}  or  None
    """
    info = _get_server(template_dir, camera_k)
    if info is None:
        return None

    req = json.dumps({"image": image_path, "bbox": list(map(float, bbox))})
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
    """모든 영구 서버 종료."""
    with _servers_lock:
        for info in _servers.values():
            try:
                info["proc"].terminate()
            except Exception:
                pass
        _servers.clear()
