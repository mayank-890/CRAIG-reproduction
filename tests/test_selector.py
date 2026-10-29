import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset
from craig.selector import CraigSelector
from craig.similarity import similarity_matrix
from craig.weights import compute_weights
from analysis.gradient_error import gradient_error
from models.logistic_regression import LogisticRegression


def test_weights_sum_to_n():
    S = similarity_matrix(torch.randn(50, 4))
    w = compute_weights(S, [0, 5, 9])
    assert w.sum() == 50 and (w >= 1).all()


def _ds(n=300):
    g = torch.Generator().manual_seed(0)
    return TensorDataset(torch.randn(n, 784, generator=g), torch.randint(0, 10, (n,), generator=g))


def test_selector_end_to_end():
    torch.manual_seed(0)
    res = CraigSelector(per_class=True).select(LogisticRegression(), _ds(), nn.CrossEntropyLoss(), subset_ratio=0.1)
    assert abs(len(res.indices) - 30) <= 10
    assert len(set(res.indices.tolist())) == len(res.indices)
    assert res.weights.sum() == 300
    t = res.timings
    assert np.isclose(t["selection_time"], t["gradient_time"] + t["similarity_time"] + t["facility_location_time"] + t["weight_time"])


def test_gradient_error_metrics_and_craig_beats_random():
    torch.manual_seed(0)
    sel = CraigSelector(per_class=False, keep_gradients=True)
    # clustered data (structure that CRAIG's bound exploits); pure isotropic noise has none
    g = torch.Generator().manual_seed(1)
    centers = 3 * torch.randn(10, 784, generator=g)
    y = torch.arange(300) % 10
    ds = TensorDataset(centers[y] + 0.3 * torch.randn(300, 784, generator=g), y)
    res = sel.select(LogisticRegression(), ds, nn.CrossEntropyLoss(), subset_ratio=0.1)
    G = res.gradients
    full = gradient_error(G, np.arange(G.shape[0]))
    assert full["relative_error"] < 1e-9 and abs(full["cosine_similarity"] - 1) < 1e-9
    craig = gradient_error(G, res.indices, res.weights)["relative_error"]
    rng = np.random.RandomState(0)
    rand = np.mean([gradient_error(G, rng.choice(300, len(res.indices), replace=False),
                                   np.full(len(res.indices), 300 / len(res.indices)))["relative_error"] for _ in range(20)])
    assert craig < rand
