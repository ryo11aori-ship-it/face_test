"""
Face detector utilities.

This module performs lazy import of facenet-pytorch's MTCNN to avoid heavy startup
on unit tests that do not require model loading.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

@dataclass
class DetectedFace:
    image: any            # PIL Image of aligned face (or cropped face)
    bbox: Tuple[float,float,float,float]  # x1,y1,x2,y2
    confidence: float
    landmarks: List[Tuple[float,float]]   # list of landmarks (x,y)

# Lazy import helper
_mtcnn = None
def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
        except Exception as e:
            raise RuntimeError("facenet-pytorch is required for face detection. Install via pip: pip install facenet-pytorch") from e
        # create MTCNN object; keep device automatic
        _mtcnn = MTCNN(keep_all=True)
    return _mtcnn

def detect_best_face(pil_image, return_aligned=False, return_landmarks=False):
    """
    Detects faces and returns the best detected face (highest probability).
    If no face, returns None.

    If return_aligned=True, returns the aligned face tensor/PIL ready for embedding extractor.
    Here we return the cropped face PIL image by default to keep things simple.
    """
    mtcnn = _get_mtcnn()
    boxes, probs, points = mtcnn.detect(pil_image, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None

    # select highest probability
    idx = int(probs.argmax())
    bbox = boxes[idx]
    confidence = float(probs[idx])
    lm = points[idx].tolist() if points is not None else []
    x1,y1,x2,y2 = map(float, bbox)

    # crop face region with margin
    w = x2 - x1
    h = y2 - y1
    margin = 0.2
    x1m = max(0, int(x1 - w*margin))
    y1m = max(0, int(y1 - h*margin))
    x2m = int(x2 + w*margin)
    y2m = int(y2 + h*margin)

    from PIL import Image
    img = pil_image
    cropped = img.crop((x1m, y1m, x2m, y2m))

    # if caller wants aligned face, we can call mtcnn.extract or mtcnn(img, save_path=...). For simplicity we return cropped.
    det = DetectedFace(image=cropped, bbox=(x1m,y1m,x2m,y2m), confidence=confidence, landmarks=lm)
    return det
