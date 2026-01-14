import numpy as np
from face_matcher.core.similarity import cosine_similarity, pairwise_similarities

def test_cosine_same_vector():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    s = cosine_similarity(a,b)
    assert abs(s - 1.0) < 1e-6

def test_cosine_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    s = cosine_similarity(a,b)
    assert abs(s - 0.0) < 1e-6

def test_pairwise():
    mat = np.array([[1.,0.],[0.,1.],[1.,1.]])
    sims = pairwise_similarities(mat)
    assert sims.shape == (3,3)
    assert abs(sims[0,1] - 0.0) < 1e-6
