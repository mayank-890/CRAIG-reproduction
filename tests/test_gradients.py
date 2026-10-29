import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from craig.gradients import compute_per_sample_gradients, select_param_names
from craig.representations import GradientRepresentation
from models.logistic_regression import LogisticRegression
from models.timeseries_cnn import TimeSeriesCNN


def _data(n=40, d=784, c=10):
    g = torch.Generator().manual_seed(0)
    return TensorDataset(torch.randn(n, d, generator=g), torch.randint(0, c, (n,), generator=g))


def _reference(model, ds, names):
    model.eval()
    pm = dict(model.named_parameters())
    rows = []
    for x, y in ds:
        model.zero_grad()
        nn.CrossEntropyLoss()(model(x[None]), y[None]).backward()
        rows.append(torch.cat([pm[n].grad.reshape(-1).clone() for n in names]))
    return torch.stack(rows)


def test_per_sample_gradient_shape():
    torch.manual_seed(0); m = LogisticRegression(); ds = _data()
    G = compute_per_sample_gradients(m, DataLoader(ds, batch_size=16), nn.CrossEntropyLoss(), "cpu")
    assert G.shape == (40, sum(p.numel() for p in m.parameters()))


def test_per_sample_gradient_correctness():
    torch.manual_seed(0); m = LogisticRegression(); ds = _data()
    G, meta = compute_per_sample_gradients(m, DataLoader(ds, batch_size=16), nn.CrossEntropyLoss(), "cpu", return_meta=True)
    assert meta["method"] == "vmap"
    torch.testing.assert_close(G, _reference(m, ds, meta["param_names"]), atol=1e-5, rtol=1e-4)


def test_loop_fallback_matches_vmap():
    torch.manual_seed(0); m = LogisticRegression(); ds = _data()
    a = compute_per_sample_gradients(m, DataLoader(ds, batch_size=16), nn.CrossEntropyLoss(), "cpu", method="vmap")
    b = compute_per_sample_gradients(m, DataLoader(ds, batch_size=16), nn.CrossEntropyLoss(), "cpu", method="loop")
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_cnn_with_batchnorm_correctness():
    torch.manual_seed(0); m = TimeSeriesCNN(); g = torch.Generator().manual_seed(1)
    ds = TensorDataset(torch.randn(12, 1, 50, generator=g), torch.randint(0, 2, (12,), generator=g))
    G, meta = compute_per_sample_gradients(m, DataLoader(ds, batch_size=5), nn.CrossEntropyLoss(), "cpu", return_meta=True)
    torch.testing.assert_close(G, _reference(m, ds, meta["param_names"]), atol=1e-5, rtol=1e-4)


def test_layer_selection():
    m = LogisticRegression()
    last = select_param_names(m, "last")
    assert set(last) == set(n for n, _ in m.named_parameters())
    c = TimeSeriesCNN()
    assert select_param_names(c, "last") == ["fc.weight", "fc.bias"]
    ds = TensorDataset(torch.randn(6, 1, 30), torch.randint(0, 2, (6,)))
    G = compute_per_sample_gradients(c, DataLoader(ds, batch_size=6), nn.CrossEntropyLoss(), "cpu", layers="last")
    assert G.shape == (6, 64 * 2 + 2)


def test_chunking_invariance():
    torch.manual_seed(0); m = LogisticRegression(); ds = _data(n=200)
    a = compute_per_sample_gradients(m, DataLoader(ds, batch_size=200), nn.CrossEntropyLoss(), "cpu", chunk_size=32)
    b = compute_per_sample_gradients(m, DataLoader(ds, batch_size=200), nn.CrossEntropyLoss(), "cpu", chunk_size=128)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_random_projection():
    rep = GradientRepresentation(projection_dim=64, projection_seed=3)
    G = torch.randn(50, 1000)
    P = rep.project(G)
    assert P.shape == (50, 64)
    torch.testing.assert_close(P, GradientRepresentation(projection_dim=64, projection_seed=3).project(G))
