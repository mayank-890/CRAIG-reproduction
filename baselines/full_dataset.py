"""Full dataset baseline selector."""

import time
from typing import Tuple, Optional
import numpy as np
import torch


class FullDatasetSelector:
    """Selector representing 100% of the training dataset."""

    def __init__(self):
        pass

    def select(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Select all indices in the dataset.
        
        Args:
            num_samples: Total number of samples N.
            
        Returns:
            Tuple of:
                selected_indices: np.ndarray of shape [N]
                weights: np.ndarray of shape [N] (all 1.0)
                selection_time: float (near 0)
        """
        start = time.perf_counter()
        indices = np.arange(num_samples)
        weights = np.ones(num_samples, dtype=np.float32)
        selection_time = time.perf_counter() - start
        return indices, weights, selection_time
