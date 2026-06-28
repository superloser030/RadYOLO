import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"
_ARCHIVE_DIR  = _PROJECT_ROOT / "archive"

_F_START = _DATA_DIR / ".session_start"
_F_END   = _DATA_DIR / ".session_end"
_F_HB    = _DATA_DIR / ".session_heartbeat"

_HB_INTERVAL = 10  # seconds


def record_session_start():
    _DATA_DIR.mkdir(exist_ok=True)
    _F_START.write_text(datetime.now().strftime("%Y%m%d_%H%M"))
    _F_END.unlink(missing_ok=True)
    _F_HB.unlink(missing_ok=True)


def update_heartbeat():
    _F_HB.write_text(datetime.now().strftime("%H%M"))


def record_session_end():
    _F_END.write_text(datetime.now().strftime("%H%M"))


def start_heartbeat_thread():
    def _loop():
        while True:
            update_heartbeat()
            time.sleep(_HB_INTERVAL)
    threading.Thread(target=_loop, daemon=True).start()


def _save_preserved(archive_base: Path) -> dict:
    """model_trellis.glb / templates/ / iqData_RecordingParameters.mat 를 임시 위치에 보존."""
    saved = {}

    # GLB + templates
    objects_dir = _DATA_DIR / "objects"
    if objects_dir.exists():
        for obj_dir in objects_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            glb  = obj_dir / "model_trellis.glb"
            tmpl = obj_dir / "templates"
            if glb.exists():
                saved[obj_dir.name + "/model_trellis.glb"] = glb.read_bytes()
            if tmpl.exists():
                tmp = archive_base / f"_tmp_templates_{obj_dir.name}"
                if tmp.exists():
                    shutil.rmtree(tmp)
                shutil.copytree(str(tmpl), str(tmp))
                saved[obj_dir.name + "/__templates__"] = tmp

    # RecordingParameters.mat
    mat = _DATA_DIR / "radar" / "iqData_RecordingParameters.mat"
    if mat.exists():
        saved["__recording_params_mat__"] = mat.read_bytes()

    return saved


def _restore_preserved(saved: dict):
    """보존했던 파일들을 data/ 로 복원."""
    if not saved:
        return
    new_objects = _DATA_DIR / "objects"
    new_objects.mkdir(parents=True, exist_ok=True)
    for rel, payload in saved.items():
        if rel == "__recording_params_mat__":
            radar_dir = _DATA_DIR / "radar"
            radar_dir.mkdir(parents=True, exist_ok=True)
            (radar_dir / "iqData_RecordingParameters.mat").write_bytes(payload)
            continue
        obj_name, fname = rel.split("/", 1)
        obj_dir = new_objects / obj_name
        obj_dir.mkdir(exist_ok=True)
        if fname == "__templates__":
            shutil.copytree(str(payload), str(obj_dir / "templates"))
            shutil.rmtree(payload)
        else:
            (obj_dir / fname).write_bytes(payload)
    print(f"[Archive] 보존 복원: {list(saved.keys())}")


def archive_data():
    """이전 세션 data/ 내용을 archive/YYYYMMDD_HHMM~HHMM/ 으로 이동.
    model_trellis.glb 와 templates/ 는 이동 후 복원.
    """
    _SESSION_FILES = {_F_START.name, _F_END.name, _F_HB.name}
    _ARCHIVE_DIR.mkdir(exist_ok=True)

    if not _DATA_DIR.exists():
        return

    items = [
        p for p in _DATA_DIR.iterdir()
        if p.name not in _SESSION_FILES
        and (p.is_file() or (p.is_dir() and any(p.rglob("*"))))
    ]
    if not items:
        return

    # GLB / templates 보존
    saved = _save_preserved(_ARCHIVE_DIR)

    # 시작 시각
    if _F_START.exists():
        start_str = _F_START.read_text().strip()
    else:
        all_files = [f for item in items for f in (list(item.rglob("*")) if item.is_dir() else [item])]
        if all_files:
            oldest = min(all_files, key=lambda f: f.stat().st_mtime)
            start_str = datetime.fromtimestamp(oldest.stat().st_mtime).strftime("%Y%m%d_%H%M")
        else:
            start_str = datetime.now().strftime("%Y%m%d_%H%M")

    # 종료 시각
    if _F_END.exists():
        end_str = _F_END.read_text().strip()
    elif _F_HB.exists():
        end_str = _F_HB.read_text().strip()
    else:
        end_str = "xxxx"

    folder_name = f"{start_str}~{end_str}"
    dest = _ARCHIVE_DIR / folder_name
    dest.mkdir(parents=True, exist_ok=True)

    for item in items:
        shutil.move(str(item), str(dest / item.name))

    for f in (_F_START, _F_END, _F_HB):
        f.unlink(missing_ok=True)

    print(f"[Archive] → archive/{folder_name}/")

    # GLB / templates 복원
    _restore_preserved(saved)
