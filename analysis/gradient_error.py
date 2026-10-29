"""Gradient approximation quality of a weighted subset.

G_full = sum_i g_i ; G_subset = sum_{j in S} w_j g_j
relative error = ||G_full - G_subset|| / ||G_full|| ; cosine = <G_full, G_subset> / (||.|| ||.||)
"""
import numpy as np
import torch


def gradient_error(G: torch.Tensor, indices, weights=None):
    G = G.double()
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    w = torch.ones(len(idx), dtype=torch.float64) if weights is None else torch.as_tensor(np.asarray(weights), dtype=torch.float64)
    g_full = G.sum(0)
    g_sub = (G[idx] * w[:, None]).sum(0)
    nf = g_full.norm()
    rel = float((g_full - g_sub).norm() / nf) if nf > 0 else float("nan")
    cos = float(torch.nn.functional.cosine_similarity(g_full, g_sub, dim=0))
    return {"relative_error": rel, "cosine_similarity": cos}
