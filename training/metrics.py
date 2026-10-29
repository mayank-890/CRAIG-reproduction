"""Metrics evaluation and timing utilities."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import torch


@dataclass
class EvaluationMetrics:
    loss: float
    accuracy: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeMetrics:
    data_loading_time: float = 0.0
    gradient_computation_time: float = 0.0
    similarity_computation_time: float = 0.0
    selection_time: float = 0.0
    training_time: float = 0.0
    total_time: float = 0.0
    peak_memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-1 classification accuracy.
    
    Args:
        outputs: Raw logits of shape [N, num_classes] or [N].
        targets: Ground truth class labels of shape [N].
        
    Returns:
        Accuracy between 0.0 and 1.0.
    """
    with torch.no_grad():
        if outputs.ndim > 1:
            preds = outputs.argmax(dim=-1)
        else:
            preds = (outputs > 0.0).long()
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return float(correct / total) if total > 0 else 0.0
