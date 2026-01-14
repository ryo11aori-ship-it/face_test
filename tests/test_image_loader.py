from pathlib import Path
from PIL import Image
import tempfile
from face_matcher.utils.image_loader import load_image_pil, list_image_files

def test_list_and_load(tmp_path):
    # create sample png
    p = tmp_path / "a.png"
    img = Image.new("RGB", (64,64), color=(123,222,111))
    img.save(p)
    files = list_image_files(tmp_path)
    assert len(files) == 1
    loaded = load_image_pil(files[0])
    assert loaded.size == (64,64)
