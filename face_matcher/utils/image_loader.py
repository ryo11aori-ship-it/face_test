"""
Image loader utilities. Supports jpg/jpeg/png/heif.

Functions:
- list_image_files(dirpath)
- load_image_pil(path) -> PIL.Image
"""
from pathlib import Path
from typing import List
from PIL import Image
import os

_VALID_EXT = {".jpg", ".jpeg", ".png", ".heif", ".heic"}

def list_image_files(dirpath: Path) -> List[Path]:
    dirpath = Path(dirpath)
    files = []
    if not dirpath.is_dir():
        return []
    for p in sorted(dirpath.iterdir()):
        if p.suffix.lower() in _VALID_EXT:
            files.append(p)
    return files

def load_image_pil(path: Path):
    """
    Load image and return PIL.Image (RGB).
    Uses pillow-heif plugin if available for HEIF/HEIC.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".heif", ".heic"):
        # attempt to use pillow_heif if installed
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            # if pillow_heif not available, PIL may still fail
            pass
    img = Image.open(path).convert("RGB")
    return img
