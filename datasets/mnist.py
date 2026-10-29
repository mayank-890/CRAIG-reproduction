"""MNIST dataset loader and synthetic dataset generation for tests."""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


class FlattenTransform:
    """Transform to flatten 2D image tensor to 1D vector of shape [784]."""
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(-1)


class TensorSubsetDataset(Dataset):
    """Simple in-memory dataset wrapping input and target tensors, with optional weights."""
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None):
        self.inputs = inputs
        self.targets = targets
        self.weights = weights

    def __len__(self) -> int:
        return self.targets.size(0)

    def __getitem__(self, idx: int):
        if self.weights is not None:
            return self.inputs[idx], self.targets[idx], self.weights[idx]
        return self.inputs[idx], self.targets[idx]


def get_mnist_datasets(
    data_dir: str = "./data",
    flatten: bool = True,
    max_train_samples: Optional[int] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load MNIST train and test datasets with standard normalization.
    
    Args:
        data_dir: Directory where MNIST raw files are stored.
        flatten: Whether to flatten images from [1, 28, 28] to [784].
        max_train_samples: If provided, slice the training set to this size (useful for debugging).
        download: Whether to download if missing.
        
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if flatten:
        transform_list.append(FlattenTransform())
        
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(max_train_samples)))

    return train_dataset, test_dataset


def create_synthetic_mnist(
    num_train: int = 128,
    num_test: int = 64,
    num_classes: int = 10,
    input_dim: int = 784,
    flatten: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Generate tiny synthetic datasets matching MNIST shapes for unit testing without downloads."""
    if seed is not None:
        torch.manual_seed(seed)
    shape_train = (num_train, input_dim) if flatten else (num_train, 1, 28, 28)
    shape_test = (num_test, input_dim) if flatten else (num_test, 1, 28, 28)
    
    x_train = torch.randn(*shape_train)
    y_train = torch.randint(0, num_classes, (num_train,))
    
    x_test = torch.randn(*shape_test)
    y_test = torch.randint(0, num_classes, (num_test,))
    
    return TensorSubsetDataset(x_train, y_train), TensorSubsetDataset(x_test, y_test)
