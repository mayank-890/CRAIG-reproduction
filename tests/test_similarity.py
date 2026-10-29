import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest, torch
from craig.similarity import pairwise_distances, distance_to_similarity, similarity_matrix


def test_euclidean_matches_cdist_and_blockwise():
    X = torch.randn(97, 20)
    ref = torch.cdist(X.double(), X.double()).float()
    for bs in (7, 32, 1000):
        torch.testing.assert_close(pairwise_distances(X, block_size=bs), ref, atol=1e-4, rtol=1e-4)


def test_cosine():
    X = torch.randn(30, 8)
    Xn = torch.nn.functional.normalize(X, dim=1)
    torch.testing.assert_close(pairwise_distances(X, metric="cosine", block_size=4), (1 - Xn @ Xn.T).clamp_min(0).fill_diagonal_(0), atol=1e-5, rtol=1e-5)


def test_similarity_properties():
    X = torch.randn(40, 5)
    D = pairwise_distances(X); S = distance_to_similarity(D)
    assert (S >= 0).all() and torch.allclose(S, S.T, atol=1e-5)
    assert torch.allclose(S.diagonal(), D.max().expand(40))
    assert torch.isclose(S.min(), torch.tensor(0.0), atol=1e-5)


def test_memory_guard():
    with pytest.raises(MemoryError):
        similarity_matrix(torch.randn(100, 3), max_bytes=1000)
