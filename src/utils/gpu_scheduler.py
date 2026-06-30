import queue
import threading
import itertools
from enum import IntEnum
from concurrent.futures import Future


class Priority(IntEnum):
    REALTIME    = 0
    INTERACTIVE = 1
    BACKGROUND  = 2
    OFFLINE     = 3


class GPUScheduler:

    def __init__(self):
        self._q       = queue.PriorityQueue()
        self._counter = itertools.count()
        self._worker  = None
        self._stop    = threading.Event()

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True, name="gpu-scheduler")
        self._worker.start()

    def stop(self):
        self._stop.set()

    def submit(self, fn, *args, priority: Priority = Priority.BACKGROUND, **kwargs) -> Future:
        fut  = Future()
        item = (int(priority), next(self._counter), fn, args, kwargs, fut)
        self._q.put(item)
        return fut

    def _run(self):
        while not self._stop.is_set():
            try:
                _, _, fn, args, kwargs, fut = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn(*args, **kwargs))
                except Exception as e:
                    fut.set_exception(e)
            self._q.task_done()


def vram_free_mb():
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
    except Exception:
        pass
    return None


class GPUManager:
    COMFYUI_URL = "http://127.0.0.1:8188"

    @staticmethod
    def empty_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def free_comfyui(cls):
        import json
        import urllib.request
        import subprocess
        try:
            req = urllib.request.Request(
                f"{cls.COMFYUI_URL}/free",
                data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=15)
            print("[VRAM] ComfyUI 본체 모델 언로드")
        except Exception as e:
            print(f"[VRAM] ComfyUI free 실패(무시): {e}")
        try:
            subprocess.run(
                ["powershell", "-Command",
                 "Get-Process python -ErrorAction SilentlyContinue | "
                 "Where-Object { $_.Path -like '*depthanything*' } | Stop-Process -Force"],
                timeout=10, capture_output=True)
            print("[VRAM] DA3 추론 프로세스 종료 (~6GB 반환)")
        except Exception as e:
            print(f"[VRAM] DA3 종료 실패(무시): {e}")

    @classmethod
    def release(cls, label="", comfyui=False):
        if comfyui:
            cls.free_comfyui()
        cls.empty_cache()
        free = vram_free_mb()
        tag  = f"{label} " if label else ""
        if free is not None:
            print(f"[VRAM] {tag}반환 (여유 {free/1024:.1f}GB)")
        elif label:
            print(f"[VRAM] {tag}반환")


scheduler = GPUScheduler()
