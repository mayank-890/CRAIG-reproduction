"""Model training engine supporting standard and weighted coreset optimization."""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .evaluator import evaluate_model
from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_weights: bool = False,
) -> Tuple[float, float]:
    """Train model for a single epoch.
    
    Args:
        model: PyTorch module to train.
        dataloader: Training DataLoader. Batches may be (inputs, targets) or (inputs, targets, weights).
        optimizer: Torch optimizer.
        criterion: Loss function. If use_weights is True, should support per-sample reduction (reduction='none').
        device: Torch device.
        use_weights: Whether to weight individual sample losses.
        
    Returns:
        Tuple of (average_epoch_loss, epoch_accuracy).
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        weights = None
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                inputs, targets, weights = batch[0], batch[1], batch[2]
            else:
                inputs, targets = batch[0], batch[1]
        else:
            inputs = batch["input"]
            targets = batch["target"]
            weights = batch.get("weight", None)
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        if weights is not None:
            weights = weights.to(device)
            
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if use_weights and weights is not None:
            # Per-sample loss calculation
            if hasattr(criterion, "reduction"):
                prev_reduction = criterion.reduction
                criterion.reduction = "none"
                per_sample_loss = criterion(outputs, targets)
                criterion.reduction = prev_reduction
            else:
                per_sample_criterion = nn.CrossEntropyLoss(reduction="none")
                per_sample_loss = per_sample_criterion(outputs, targets)
                
            # Normalize by sum of batch weights so gradient scale matches standard expectation
            weights_sum = weights.sum()
            if weights_sum > 0:
                loss = (per_sample_loss * weights).sum() / weights_sum
            else:
                loss = per_sample_loss.mean()
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=-1) if outputs.ndim > 1 else (outputs > 0).long()
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size
        
    avg_loss = float(total_loss / total_samples) if total_samples > 0 else 0.0
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    use_weights: bool = False,
    log_interval: int = 5,
) -> Dict[str, Any]:
    """Complete multi-epoch training pipeline with wall-clock timing.
    
    Args:
        model: PyTorch module.
        train_loader: DataLoader for training.
        val_loader: Optional DataLoader for validation/test tracking.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Number of epochs.
        device: Device to train on.
        use_weights: Whether to apply sample coreset weights.
        log_interval: Epochs between logs.
        
    Returns:
        Dictionary containing training history and elapsed wall-clock training time.
    """
    model.to(device)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    start_time = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_weights=use_weights,
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, criterion=criterion, device=device)
            history["val_loss"].append(val_metrics.loss)
            history["val_acc"].append(val_metrics.accuracy)
        else:
            val_metrics = None
            
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            val_info = f" | Val Acc: {val_metrics.accuracy:.4f}" if val_metrics else ""
            logger.info("Epoch %3d/%3d - Loss: %.4f - Acc: %.4f%s", epoch, epochs, train_loss, train_acc, val_info)
            
    training_time = time.perf_counter() - start_time
    
    final_val_metrics = (
        evaluate_model(model, val_loader, criterion=criterion, device=device)
        if val_loader is not None
        else None
    )
    
    return {
        "history": history,
        "training_time": training_time,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
        "final_val_metrics": final_val_metrics,
    }
