"""CIFAR-10 dataset loader and synthetic dataset generation."""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from .mnist import TensorSubsetDataset


def get_cifar10_datasets(
    data_dir: str = "./data",
    max_train_samples: Optional[int] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 train and test datasets with standard normalization."""
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(max_train_samples)))

    return train_dataset, test_dataset


def create_synthetic_cifar10(
    num_train: int = 128,
    num_test: int = 64,
    num_classes: int = 10,
) -> Tuple[Dataset, Dataset]:
    """Generate tiny synthetic datasets matching CIFAR-10 shapes [B, 3, 32, 32]."""
    torch.manual_seed(42)
    x_train = torch.randn(num_train, 3, 32, 32)
    y_train = torch.randint(0, num_classes, (num_train,))
    x_test = torch.randn(num_test, 3, 32, 32)
    y_test = torch.randint(0, num_classes, (num_test,))
    return TensorSubsetDataset(x_train, y_train), TensorSubsetDataset(x_test, y_test)
