# Order for commands in powershell:
# ffmpeg -i test.mp4 -vf fps=5 images/frame_%04d.png
# python test.py
# .\run-colmap-geometric.ps1 -WorkspacePath .\dense_model
# python watertightness.py

import io
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap


USE_MASKING = True
MASK_MODE = "ai"
MASK_BACKGROUND_TO = (255, 255, 255)
IMAGE_PATH = Path("images")
MASKED_IMAGE_PATH = Path("masked_images")
DATABASE_PATH = Path("database.db")
SPARSE_PATH = Path("sparse_model")
DENSE_PATH = Path("dense_model")

def mask_background_ai(input_dir: Path, output_dir: Path) -> Path:
    try:
        from rembg import new_session, remove
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AI masking requires `rembg` and `onnxruntime`. Install them with "
            "`pip install rembg onnxruntime` and rerun."
        ) from exc

    output_dir.mkdir(exist_ok=True)
    session = new_session("u2net")

    for image_path in sorted(input_dir.glob("*.png")):
        image = Image.open(image_path).convert("RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = remove(
            buffer.getvalue(),
            session=session,
            only_mask=False,
            post_process_mask=True,
        )

        segmented = Image.open(io.BytesIO(result)).convert("RGBA")
        background = Image.new("RGBA", segmented.size, MASK_BACKGROUND_TO + (255,))
        composited = Image.alpha_composite(background, segmented).convert("RGB")
        composited.save(output_dir / image_path.name)

    return output_dir


def mask_background(input_dir: Path, output_dir: Path) -> Path:
    if MASK_MODE == "ai":
        return mask_background_ai(input_dir, output_dir)
    raise ValueError(f"Unsupported MASK_MODE: {MASK_MODE}")


def run_colmap(image_path: Path) -> None:
    pycolmap.extract_features(DATABASE_PATH, image_path)
    pycolmap.match_sequential(DATABASE_PATH)

    SPARSE_PATH.mkdir(exist_ok=True)
    DENSE_PATH.mkdir(exist_ok=True)

    pycolmap.incremental_mapping(DATABASE_PATH, image_path, SPARSE_PATH)
    pycolmap.undistort_images(DENSE_PATH, SPARSE_PATH / "0", image_path)


def main() -> None:
    source_path = IMAGE_PATH
    if USE_MASKING:
        source_path = mask_background(IMAGE_PATH, MASKED_IMAGE_PATH)

    run_colmap(source_path)


if __name__ == "__main__":
    main()
