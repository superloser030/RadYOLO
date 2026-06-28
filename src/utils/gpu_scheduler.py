"""GPU 작업 스케줄러 — VRAM 경합 방지용 우선순위 큐 + 단일 워커.

⚠ 아직 파이프라인에 연결되지 않은 골격(skeleton)이다.
VRAM 무거운 작업이 늘어나면(실시간 메시 렌더링, 동적 배경 갱신 등) 기존 GPU
작업(live YOLO, GigaPose 등)을 submit() 으로 통과시켜 GPU 를 직렬화한다.

사용 예 (나중에):
    from src.utils.gpu_scheduler import scheduler, Priority
    scheduler.start()
    fut = scheduler.submit(run_yolo, frame, priority=Priority.REALTIME)
    result = fut.result()
"""
import queue
import threading
import itertools
from enum import IntEnum
from concurrent.futures import Future


class Priority(IntEnum):
    """숫자가 작을수록 먼저 실행."""
    REALTIME    = 0   # 실시간 추적 (지연 민감, 최우선)
    INTERACTIVE = 1   # 포즈 추정 (GigaPose)
    BACKGROUND  = 2   # 동적 배경 갱신
    OFFLINE     = 3   # 3D 생성(Trellis) 등 무거운 1회성


class GPUScheduler:
    """GPU 작업을 단일 워커 스레드에서 우선순위 순으로 직렬 실행.

    한 번에 하나의 작업만 GPU 를 점유하게 해 OOM/병목을 막고,
    실시간 작업을 무거운 배치 작업보다 우선시킨다.
    """

    def __init__(self):
        self._q       = queue.PriorityQueue()
        self._counter = itertools.count()   # 같은 우선순위 내 FIFO tiebreaker
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
        """GPU 작업 제출 → Future 반환. priority 숫자 작을수록 먼저 실행."""
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
                except Exception as e:        # 작업 실패가 워커를 죽이지 않도록
                    fut.set_exception(e)
            self._q.task_done()


def vram_free_mb():
    """현재 GPU 여유 VRAM(MB). torch/CUDA 없으면 None.

    워커에서 무거운 작업 전 여유 확인용 (연결 시 활용 예정).
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
    except Exception:
        pass
    return None


# 모듈 전역 싱글턴 — 필요한 곳에서 import 해서 submit
scheduler = GPUScheduler()
