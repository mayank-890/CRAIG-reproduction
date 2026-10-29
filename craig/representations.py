"""Gradient representations: full, layer-filtered, and JL random projection."""
from typing import Optional
import math
import torch
from .gradients import compute_per_sample_gradients


class GradientRepresentation:
    """kind: 'full_gradient' | 'last_layer' ; projection_dim: optional JL projection d << D."""

    def __init__(self, kind: str = "full_gradient", projection_dim: Optional[int] = None,
                 projection: str = "gaussian", projection_seed: int = 0, chunk_size: int = 128):
        self.kind = kind
        self.projection_dim = projection_dim
        self.projection = projection
        self.projection_seed = projection_seed
        self.chunk_size = chunk_size
        self._P = None

    @property
    def layers(self):
        return "last" if self.kind in ("last_layer", "last", "classifier") else "all"

    def projection_matrix(self, D: int) -> torch.Tensor:
        if self._P is None or self._P.shape[0] != D:
            gen = torch.Generator().manual_seed(self.projection_seed)
            d = self.projection_dim
            if self.projection == "rademacher":
                P = torch.randint(0, 2, (D, d), generator=gen).float() * 2 - 1
            else:
                P = torch.randn(D, d, generator=gen)
            self._P = P / math.sqrt(d)
        return self._P

    def project(self, G: torch.Tensor) -> torch.Tensor:
        if not self.projection_dim or self.projection_dim >= G.shape[1]:
            return G
        return G @ self.projection_matrix(G.shape[1])

    def compute(self, model, dataloader, loss_fn, device="cpu", max_samples=None, return_meta=False):
        G, meta = compute_per_sample_gradients(model, dataloader, loss_fn, device, layers=self.layers,
                                               chunk_size=self.chunk_size, max_samples=max_samples,
                                               return_meta=True)
        G = self.project(G)
        return (G, meta) if return_meta else G
