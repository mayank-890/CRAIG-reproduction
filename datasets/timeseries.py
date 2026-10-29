"""Time-series dataset loader and synthetic generation for scaling analysis."""

import os
from typing import Tuple, Optional
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from .mnist import TensorSubsetDataset


def create_synthetic_timeseries(
    num_train: int = 500,
    num_test: int = 100,
    seq_length: int = 100,
    num_classes: int = 2,
    num_channels: int = 1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Generate synthetic time-series data with distinctive class frequency signatures.
    
    Args:
        num_train: Number of training sequences.
        num_test: Number of test sequences.
        seq_length: Length of each sequence.
        num_classes: Number of distinct classes.
        num_channels: Number of channels per sequence.
        seed: Random seed.
        
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    rng = np.random.RandomState(seed)
    
    def generate_data(n_samples: int):
        t = np.linspace(0, 1, seq_length)
        X = np.zeros((n_samples, num_channels, seq_length), dtype=np.float32)
        y = rng.randint(0, num_classes, size=n_samples).astype(np.int64)
        
        for i in range(n_samples):
            cls = y[i]
            freq = (cls + 1) * 2.0
            phase = rng.uniform(0, 2 * np.pi)
            signal = np.sin(2 * np.pi * freq * t + phase)
            noise = rng.normal(0, 0.2, size=seq_length)
            for c in range(num_channels):
                X[i, c] = signal + noise
                
        return torch.from_numpy(X), torch.from_numpy(y)

    x_train, y_train = generate_data(num_train)
    x_test, y_test = generate_data(num_test)
    
    return TensorSubsetDataset(x_train, y_train), TensorSubsetDataset(x_test, y_test)


def get_timeseries_datasets(
    dataset_name: str = "synthetic",
    data_dir: str = "./data/timeseries",
    seq_length: int = 100,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Retrieve time-series datasets."""
    if dataset_name == "synthetic" or not os.path.exists(os.path.join(data_dir, dataset_name)):
        return create_synthetic_timeseries(
            num_train=max_train_samples or 1000,
            num_test=200,
            seq_length=seq_length,
            num_classes=2,
            seed=seed,
        )
        
    raise NotImplementedError(f"Dataset {dataset_name} download requires external dataset files.")
