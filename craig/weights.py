"""Coreset weights: Voronoi assignment of every point to its most similar medoid; w_j = |C_j|."""
import numpy as np
import torch


def voronoi_assign(S, selected) -> np.ndarray:
    """Return, for each i, the position (0..k-1) in `selected` of argmax_{j in S} s(i, j)."""
    S = S.numpy() if isinstance(S, torch.Tensor) else np.asarray(S)
    return np.argmax(S[:, np.asarray(selected)], axis=1)


def compute_weights(S, selected) -> np.ndarray:
    """w_j = number of points whose nearest medoid is j. sum_j w_j = N."""
    assign = voronoi_assign(S, selected)
    return np.bincount(assign, minlength=len(selected)).astype(np.float32)
