"""Model evaluation loop."""

from typing import Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .metrics import EvaluationMetrics, calculate_accuracy


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> EvaluationMetrics:
    """Evaluate model on a dataset.
    
    Args:
        model: PyTorch module.
        dataloader: Evaluation DataLoader.
        criterion: Loss function (defaults to nn.CrossEntropyLoss()).
        device: Device to run evaluation on.
        
    Returns:
        EvaluationMetrics instance with average loss and accuracy.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch["input"], batch["target"]
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            
            preds = outputs.argmax(dim=-1) if outputs.ndim > 1 else (outputs > 0).long()
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size
            
    avg_loss = float(total_loss / total_samples) if total_samples > 0 else 0.0
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0
    
    return EvaluationMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        num_samples=total_samples,
    )
