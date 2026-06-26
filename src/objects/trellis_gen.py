"""
fal.ai Trellis 3D 생성
cutout.jpg (흰 배경 오브젝트) → model_trellis.glb
"""
import os
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def generate_3d(obj_dir: Path) -> bool:
    """
    cutout.jpg → fal.ai Trellis → model_trellis.glb
    Returns True on success.
    """
    _load_env()

    cutout = obj_dir / "cutout.jpg"
    out    = obj_dir / "model_trellis.glb"

    if not cutout.exists():
        print(f"[Trellis] cutout.jpg 없음: {obj_dir.name}")
        return False

    try:
        import fal_client
    except ImportError:
        print("[Trellis] fal-client 미설치. pip install fal-client")
        return False

    print(f"[Trellis] {obj_dir.name}: 이미지 업로드 중...")
    image_url = fal_client.upload_file(str(cutout))

    print(f"[Trellis] {obj_dir.name}: 3D 생성 중 (30~60초)...")
    result = fal_client.run(
        "fal-ai/trellis",
        arguments={
            "image_url": image_url,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 12,
            "slat_guidance_strength": 3.0,
            "slat_sampling_steps": 12,
            "mesh_simplify": 0.95,
            "texture_size": 1024,
        },
    )

    glb_url = (result.get("model_mesh") or {}).get("url")
    if not glb_url:
        print(f"[Trellis] {obj_dir.name}: GLB URL 없음 — {result}")
        return False

    print(f"[Trellis] {obj_dir.name}: GLB 다운로드 중...")
    urllib.request.urlretrieve(glb_url, str(out))
    print(f"[Trellis] {obj_dir.name}: 저장 완료 → {out.relative_to(PROJECT_ROOT)}")
    return True


def generate_all():
    objects_dir = PROJECT_ROOT / "data" / "objects"
    if not objects_dir.exists():
        print("[Trellis] data/objects/ 없음")
        return
    for obj_dir in sorted(objects_dir.iterdir()):
        if obj_dir.is_dir() and (obj_dir / "cutout.jpg").exists():
            generate_3d(obj_dir)


if __name__ == "__main__":
    generate_all()
