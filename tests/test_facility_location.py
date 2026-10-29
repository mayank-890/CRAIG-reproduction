import os, sys, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from craig.similarity import similarity_matrix
from craig.facility_location import greedy_facility_location, facility_location_objective


def test_lazy_equals_naive():
    S = similarity_matrix(torch.randn(80, 6))
    a, ga = greedy_facility_location(S, 10, "naive"); b, gb = greedy_facility_location(S, 10, "lazy")
    assert np.isclose(facility_location_objective(S, a), facility_location_objective(S, b))


def test_gains_diminishing_and_unique():
    S = similarity_matrix(torch.randn(60, 4))
    sel, gains = greedy_facility_location(S, 15)
    assert len(set(sel.tolist())) == 15
    assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))
    assert np.isclose(sum(gains), facility_location_objective(S, sel))


def test_approximation_bound_vs_bruteforce():
    rng = torch.Generator().manual_seed(1)
    S = similarity_matrix(torch.randn(12, 3, generator=rng))
    k = 3
    opt = max(facility_location_objective(S, c) for c in itertools.combinations(range(12), k))
    sel, _ = greedy_facility_location(S, k)
    assert facility_location_objective(S, sel) >= (1 - 1 / np.e) * opt - 1e-9


def test_clusters_pick_one_per_cluster():
    centers = torch.tensor([[0., 0.], [100., 0.], [0., 100.]])
    X = torch.cat([c + 0.1 * torch.randn(20, 2) for c in centers])
    sel, _ = greedy_facility_location(similarity_matrix(X), 3)
    assert sorted((sel // 20).tolist()) == [0, 1, 2]
