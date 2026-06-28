"""TOML 설정 로더 — 센더/리시버 공통.

config/
  network.toml   IP/포트 (양쪽 공유)
  sender.toml    노트북: 카메라 캡처 + 레이더 + 레벨 테이블 + 모드
  receiver.toml  데스크톱: 파이프라인/저장 옵션
"""
import tomllib
from pathlib import Path

_ROOT   = Path(__file__).resolve().parent.parent.parent
_CONFIG = _ROOT / "config"


def load_toml(name: str) -> dict:
    with open(_CONFIG / name, "rb") as f:
        return tomllib.load(f)


def load_network() -> dict:
    """network.toml 의 [network] 섹션 (desktop_ip, *_port)."""
    return load_toml("network.toml")["network"]


def load_sender() -> dict:
    """sender.toml 전체 (level / camera / dca / radar / mode)."""
    return load_toml("sender.toml")


def load_receiver() -> dict:
    """receiver.toml 전체."""
    return load_toml("receiver.toml")


def load_camera() -> dict:
    """sender.toml [camera] 섹션 (캡처 설정 + intrinsic)."""
    return load_sender()["camera"]


# viewer.html / pose 가 쓰는 intrinsic 키
_CAMERA_VIEW_KEYS = ["model", "width", "height", "fx", "fy", "cx", "cy",
                     "dfov_deg", "hfov_deg", "vfov_deg"]


def export_camera_json(dest) -> dict:
    """sender.toml [camera] 의 intrinsic 을 viewer 용 JSON 으로 export.

    브라우저는 TOML 파싱이 안 되므로 main_r 이 시작 시 호출해
    viewer.html 이 fetch 할 JSON 을 생성한다.
    """
    import json
    cam  = load_camera()
    view = {k: cam[k] for k in _CAMERA_VIEW_KEYS if k in cam}
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(view, indent=2))
    return view


def resolve_level(sender_cfg: dict, level_idx: int) -> dict:
    """1-based 레벨 번호 → 레벨 dict.

    chirp/fps 로부터 레이더 파생값(num_loops, frame_period_01ms, bin_frame_size)도
    계산해 함께 반환한다. 센더(.cfg 생성)·리시버(.bin 프레임 크기) 양쪽이 공유.
    """
    levels = sender_cfg["level"]
    if not (1 <= level_idx <= len(levels)):
        raise ValueError(f"level {level_idx} 범위 밖 (1~{len(levels)})")
    lv    = dict(levels[level_idx - 1])
    radar = sender_cfg["radar"]

    lv["num_loops"]         = lv["chirp"] // 2          # TDM 2TX
    lv["frame_period_01ms"] = round(10000 / lv["fps"])  # frameCfg 5번째 인자
    lv["bin_frame_size"]    = (radar["samples_per_chirp"]
                               * radar["num_receivers"]
                               * lv["chirp"] * 4)        # 4 bytes/sample (I+Q int16)
    lv["level"] = level_idx
    return lv
