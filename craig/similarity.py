"""Pairwise gradient distances and submodular similarity s(i,j) = max(D) - d(i,j)."""
import torch

DEFAULT_MAX_BYTES = 4 * 1024 ** 3  # 4 GiB guard for the dense N x N matrix


def _check_memory(n, m, max_bytes):
    need = n * m * 4
    if max_bytes is not None and need > max_bytes:
        raise MemoryError(f"Distance matrix {n}x{m} needs {need/1e9:.2f} GB > guard {max_bytes/1e9:.2f} GB. "
                          "Use per-class selection or raise max_bytes.")


def pairwise_distances(X: torch.Tensor, Y: torch.Tensor = None, metric: str = "euclidean",
                       block_size: int = 1024, max_bytes: int = DEFAULT_MAX_BYTES) -> torch.Tensor:
    """Blockwise pairwise distance matrix [N, M] (float32).

    euclidean: ||x_i - y_j||_2 ; cosine: 1 - cos(x_i, y_j) (in [0, 2]).
    Rows are processed in blocks so temporaries are O(block_size * M).
    """
    X = X.float()
    Y = X if Y is None else Y.float()
    n, m = X.shape[0], Y.shape[0]
    _check_memory(n, m, max_bytes)
    out = torch.empty(n, m, dtype=torch.float32)
    if metric == "euclidean":
        y2 = (Y * Y).sum(1)
        for s in range(0, n, block_size):
            xb = X[s:s + block_size]
            d2 = (xb * xb).sum(1, keepdim=True) + y2[None, :] - 2.0 * xb @ Y.T
            out[s:s + block_size] = d2.clamp_min_(0).sqrt_()
    elif metric == "cosine":
        Yn = torch.nn.functional.normalize(Y, dim=1, eps=1e-12)
        for s in range(0, n, block_size):
            xb = torch.nn.functional.normalize(X[s:s + block_size], dim=1, eps=1e-12)
            out[s:s + block_size] = (1.0 - xb @ Yn.T).clamp_min_(0)
    else:
        raise ValueError(f"Unknown metric {metric}")
    if Y is X or Y.data_ptr() == X.data_ptr():
        out.fill_diagonal_(0.0)
    return out


def distance_to_similarity(D: torch.Tensor) -> torch.Tensor:
    """s(i,j) = M - d(i,j) with M = max_{i,j} d(i,j); non-negative, monotone FL objective."""
    return D.max() - D


def similarity_matrix(G: torch.Tensor, metric: str = "euclidean", block_size: int = 1024,
                      max_bytes: int = DEFAULT_MAX_BYTES) -> torch.Tensor:
    return distance_to_similarity(pairwise_distances(G, metric=metric, block_size=block_size, max_bytes=max_bytes))
