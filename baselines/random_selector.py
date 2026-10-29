"""Random subset baseline selector."""

import time
from typing import Tuple, Optional
import numpy as np


class RandomSubsetSelector:
    """Selects a random subset of training data without replacement."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        num_samples: int,
        subset_ratio: Optional[float] = None,
        subset_size: Optional[int] = None,
        weighted: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Select random subset indices and assign weights.
        
        Args:
            num_samples: Total training samples N.
            subset_ratio: Fraction of dataset to select (e.g. 0.10).
            subset_size: Exact number of samples k to select.
            weighted: Whether to assign coreset weights (N/k) or unit weights (1.0).
            
        Returns:
            Tuple of:
                selected_indices: np.ndarray of shape [k]
                weights: np.ndarray of shape [k]
                selection_time: Elapsed selection time in seconds.
        """
        start = time.perf_counter()
        
        if subset_size is None:
            if subset_ratio is None:
                raise ValueError("Either subset_ratio or subset_size must be specified.")
            k = max(1, int(round(subset_ratio * num_samples)))
        else:
            k = min(num_samples, max(1, subset_size))

        selected_indices = self.rng.choice(num_samples, size=k, replace=False)
        
        if weighted:
            # Each selected sample represents N / k samples, so sum(weights) == N
            weight_val = float(num_samples / k)
            weights = np.full(k, weight_val, dtype=np.float32)
        else:
            weights = np.ones(k, dtype=np.float32)

        selection_time = time.perf_counter() - start
        return selected_indices, weights, selection_time
