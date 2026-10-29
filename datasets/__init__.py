"""Dataset loaders and registry."""

from typing import Tuple, Any
from torch.utils.data import Dataset
from .mnist import get_mnist_datasets, create_synthetic_mnist, TensorSubsetDataset
from .cifar10 import get_cifar10_datasets, create_synthetic_cifar10
from .timeseries import get_timeseries_datasets, create_synthetic_timeseries


def get_dataset(name: str, **kwargs: Any) -> Tuple[Dataset, Dataset]:
    """Factory to load train and test datasets by name."""
    name_clean = name.lower()
    if name_clean == "mnist":
        return get_mnist_datasets(**kwargs)
    elif name_clean == "cifar10":
        return get_cifar10_datasets(**kwargs)
    elif name_clean in ("timeseries", "synthetic_timeseries"):
        return get_timeseries_datasets(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name '{name}'.")


__all__ = [
    "get_dataset",
    "get_mnist_datasets",
    "create_synthetic_mnist",
    "get_cifar10_datasets",
    "create_synthetic_cifar10",
    "get_timeseries_datasets",
    "create_synthetic_timeseries",
    "TensorSubsetDataset",
]
