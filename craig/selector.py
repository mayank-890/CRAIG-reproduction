"""Unified CRAIG selector: gradients -> similarity -> greedy facility location -> weights.

All stage timings are measured with time.perf_counter and reported separately so the
selection overhead is never hidden (selection_time = gradient + similarity + FL + weights).
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .representations import GradientRepresentation
from .similarity import similarity_matrix, DEFAULT_MAX_BYTES
from .facility_location import greedy_facility_location
from .weights import compute_weights


@dataclass
class CraigResult:
    indices: np.ndarray
    weights: np.ndarray
    timings: Dict[str, float] = field(default_factory=dict)
    gradients: Optional[torch.Tensor] = None  # full [N, D] representation (optional)

    @property
    def selection_time(self):
        return self.timings.get("selection_time", 0.0)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class CraigSelector:
    def __init__(self, metric: str = "euclidean", layers: str = "all", per_class: bool = True,
                 projection_dim: Optional[int] = None, chunk_size: int = 128, fl_method: str = "lazy",
                 max_bytes: int = DEFAULT_MAX_BYTES, device="cpu", keep_gradients: bool = False):
        self.metric = metric
        self.per_class = per_class
        self.fl_method = fl_method
        self.max_bytes = max_bytes
        self.device = torch.device(device)
        self.keep_gradients = keep_gradients
        self.rep = GradientRepresentation("last_layer" if layers in ("last", "last_layer") else "full_gradient",
                                          projection_dim=projection_dim, chunk_size=chunk_size)

    def select_from_gradients(self, G: torch.Tensor, labels: torch.Tensor, subset_ratio=None, subset_size=None):
        n = G.shape[0]
        k_total = subset_size if subset_size is not None else max(1, int(round(subset_ratio * n)))
        groups = [np.arange(n)]
        if self.per_class:
            lab = labels.numpy()
            groups = [np.where(lab == c)[0] for c in np.unique(lab)]
        t_sim = t_fl = t_w = 0.0
        idx_all, w_all = [], []
        for g in groups:
            k = max(1, int(round(k_total * len(g) / n)))
            t0 = time.perf_counter()
            S = similarity_matrix(G[g], metric=self.metric, max_bytes=self.max_bytes)
            t1 = time.perf_counter()
            sel, _ = greedy_facility_location(S, k, method=self.fl_method)
            t2 = time.perf_counter()
            w = compute_weights(S, sel)
            t3 = time.perf_counter()
            t_sim += t1 - t0; t_fl += t2 - t1; t_w += t3 - t2
            idx_all.append(g[sel]); w_all.append(w)
        return np.concatenate(idx_all), np.concatenate(w_all), {
            "similarity_time": t_sim, "facility_location_time": t_fl, "weight_time": t_w}

    def select(self, model, dataset, loss_fn, subset_ratio=None, subset_size=None, batch_size=512) -> CraigResult:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        _sync(self.device)
        t0 = time.perf_counter()
        G, meta = self.rep.compute(model, loader, loss_fn, self.device, return_meta=True)
        _sync(self.device)
        t_grad = time.perf_counter() - t0
        idx, w, tm = self.select_from_gradients(G, meta["labels"], subset_ratio, subset_size)
        tm["gradient_time"] = t_grad
        tm["selection_time"] = t_grad + tm["similarity_time"] + tm["facility_location_time"] + tm["weight_time"]
        tm["gradient_method"] = meta["method"]
        tm["gradient_dim"] = int(G.shape[1])
        return CraigResult(idx, w, tm, G if self.keep_gradients else None)
