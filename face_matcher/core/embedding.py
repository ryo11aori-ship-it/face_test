"""
Embedding extraction utilities using facenet-pytorch's InceptionResnetV1.

Design: lazy-load model to avoid heavy import on tests.
"""
import numpy as np

_model = None
_transform = None

def _get_model():
    global _model, _transform
    if _model is None:
        try:
            from facenet_pytorch import InceptionResnetV1
        except Exception as e:
            raise RuntimeError("facenet-pytorch is required for embeddings. Install via pip: pip install facenet-pytorch") from e
        import torch
        _model = InceptionResnetV1(pretrained='vggface2').eval()
        # basic transform: resize to 160x160 and normalization expected by model
        from torchvision import transforms
        _transform = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    return _model, _transform

def get_embedding_for_face(pil_face_image):
    """
    Returns a 1-D numpy array embedding (float32) for the given PIL image (face crop).
    """
    model, transform = _get_model()
    import torch
    tensor = transform(pil_face_image).unsqueeze(0)  # 1 x C x H x W
    with torch.no_grad():
        emb = model(tensor)
    emb = emb.cpu().numpy().reshape(-1).astype(np.float32)
    # L2-normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb
