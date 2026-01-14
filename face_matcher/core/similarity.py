"""
Similarity utilities.
All functions here are pure numpy to allow unit-testing without heavy model dependencies.
"""
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.
    Returns float in [-1,1]
    """
    a = a.astype(float)
    b = b.astype(float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def pairwise_similarities(mat: np.ndarray) -> np.ndarray:
    """
    Given mat shape (N, D), returns NxN matrix of cosine similarities.
    """
    if mat.ndim != 2:
        raise ValueError("mat must be 2-D")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    normalized = mat / norms
    return normalized @ normalized.T
