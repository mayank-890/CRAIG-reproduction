"""Model factory and registry."""

from typing import Any
import torch.nn as nn
from .logistic_regression import LogisticRegression
from .mnist_cnn import MNISTCNN, CIFARCNN
from .timeseries_cnn import TimeSeriesCNN

MODELS = {
    "logistic_regression": LogisticRegression,
    "mnist_cnn": MNISTCNN,
    "cifar_cnn": CIFARCNN,
    "timeseries_cnn": TimeSeriesCNN,
}


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """Factory to instantiate models by name."""
    name_clean = name.lower().replace("-", "_")
    if name_clean not in MODELS:
        raise ValueError(f"Unknown model name '{name}'. Available: {list(MODELS.keys())}")
    return MODELS[name_clean](**kwargs)


__all__ = [
    "LogisticRegression",
    "MNISTCNN",
    "CIFARCNN",
    "TimeSeriesCNN",
    "get_model",
]
