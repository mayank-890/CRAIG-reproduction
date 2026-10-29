"""Greedy maximisation of facility location F(S) = sum_i max_{j in S} s(i, j).

F is monotone submodular for s >= 0, so greedy gives a (1 - 1/e) approximation.
'lazy' (Minoux) greedy returns the same set as naive greedy (up to ties) much faster.
"""
import heapq
from typing import List, Tuple
import numpy as np
import torch


def facility_location_objective(S: np.ndarray, selected) -> float:
    S = _np(S)
    if len(selected) == 0:
        return 0.0
    return float(S[:, list(selected)].max(axis=1).sum())


def _np(S):
    return S.numpy() if isinstance(S, torch.Tensor) else np.asarray(S)


def greedy_facility_location(S, k: int, method: str = "lazy") -> Tuple[np.ndarray, List[float]]:
    """Select k columns (medoids) of similarity matrix S [N, N].

    Returns (selected indices in greedy order, marginal gains).
    """
    S = np.ascontiguousarray(_np(S), dtype=np.float64)
    n = S.shape[0]
    k = int(min(k, n))
    cur = np.zeros(n)
    selected, gains = [], []
    if method == "naive":
        avail = np.ones(n, dtype=bool)
        for _ in range(k):
            g = np.maximum(S - cur[:, None], 0).sum(axis=0)
            g[~avail] = -np.inf
            j = int(np.argmax(g))
            selected.append(j); gains.append(float(g[j]))
            avail[j] = False
            cur = np.maximum(cur, S[:, j])
        return np.array(selected, dtype=np.int64), gains
    # lazy greedy: initial gains computed vectorised
    init = S.sum(axis=0)
    heap = [(-init[j], j) for j in range(n)]
    heapq.heapify(heap)
    while len(selected) < k and heap:
        neg, j = heapq.heappop(heap)
        g = float(np.maximum(S[:, j] - cur, 0).sum())
        if not heap or g >= -heap[0][0] - 1e-12:
            selected.append(j); gains.append(g)
            cur = np.maximum(cur, S[:, j])
        else:
            heapq.heappush(heap, (-g, j))
    return np.array(selected, dtype=np.int64), gains
