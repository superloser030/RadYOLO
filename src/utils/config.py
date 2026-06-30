import math
import tomllib
from pathlib import Path

_ROOT   = Path(__file__).resolve().parent.parent.parent
_CONFIG = _ROOT / "config"


def load_toml(name: str) -> dict:
    with open(_CONFIG / name, "rb") as f:
        return tomllib.load(f)


def load_network() -> dict:
    return load_toml("network.toml")["network"]


def load_sender() -> dict:
    return load_toml("sender.toml")


def load_receiver() -> dict:
    return load_toml("receiver.toml")


def load_camera() -> dict:
    return load_sender()["camera"]


_CAMERA_VIEW_KEYS = ["model", "width", "height", "fx", "fy", "cx", "cy",
                     "dfov_deg", "hfov_deg", "vfov_deg"]


def export_camera_json(dest) -> dict:
    import json
    cam  = load_camera()
    view = {k: cam[k] for k in _CAMERA_VIEW_KEYS if k in cam}
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(view, indent=2))
    return view


_DCA_PAYLOAD = 1456
_RECORD_SAFE_SECS = 70


def resolve_level(sender_cfg: dict, level_idx: int) -> dict:
    levels = sender_cfg["level"]
    if not (1 <= level_idx <= len(levels)):
        raise ValueError(f"level {level_idx} 범위 밖 (1~{len(levels)})")
    lv    = dict(levels[level_idx - 1])
    radar = sender_cfg["radar"]

    lv["num_loops"]       = lv["chirp"] // 2
    lv["frame_period_ms"] = round(1000 / lv["fps"])
    lv["bin_frame_size"]  = (radar["samples_per_chirp"]
                             * radar["num_receivers"]
                             * lv["chirp"] * 4)
    frame_pkts = math.ceil(lv["bin_frame_size"] / _DCA_PAYLOAD)
    lv["restart_at_seq"] = int(lv["fps"] * frame_pkts * _RECORD_SAFE_SECS)
    lv["level"] = level_idx
    return lv
