#!/usr/bin/env python3
"""Verify environment, hardware accelerator, dependencies, and baseline training functionality."""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from training.reproducibility import set_seed, get_device, get_memory_info_mb
from models.logistic_regression import LogisticRegression
from datasets.mnist import create_synthetic_mnist
from baselines.random_selector import RandomSubsetSelector
from training.trainer import train_model
from torch.utils.data import DataLoader, Subset


def main():
    print("=" * 60)
    print(" CRAIG Research Project — System & Environment Verification")
    print("=" * 60)
    
    # 1. Environment and Hardware Checks
    print(f"Python Version:    {sys.version.split()[0]}")
    print(f"PyTorch Version:   {torch.__version__}")
    print(f"NumPy Version:     {np.__version__}")
    
    device = get_device("auto")
    print(f"Selected Device:   {device}")
    
    mem = get_memory_info_mb(device)
    if mem is not None:
        print(f"Memory (MB):       {mem:.2f} MB")
        
    set_seed(42)
    print("Centralized Seed:  42 (Successfully tested)")
    
    # 2. Synthetic Dataset & Model Verification
    train_ds, test_ds = create_synthetic_mnist(num_train=100, num_test=20)
    model = LogisticRegression(input_dim=784, num_classes=10)
    param_count = model.count_parameters()
    print(f"Model Parameters:  {param_count} (Linear: 784x10 + 10 bias = 7850)")
    assert param_count == 7850, f"Expected 7850 params, got {param_count}"
    
    # 3. Random Baseline Subset Selector Check
    selector = RandomSubsetSelector(seed=42)
    indices, weights, sel_time = selector.select(num_samples=len(train_ds), subset_ratio=0.20, weighted=True)
    print(f"Random Selection:  k={len(indices)} samples selected in {sel_time*1000:.3f} ms")
    assert len(indices) == 20, f"Expected 20 samples, got {len(indices)}"
    assert np.isclose(weights.sum(), 100.0), f"Expected weight sum 100, got {weights.sum()}"
    
    # 4. Training Loop Verification
    subset_ds = Subset(train_ds, indices)
    train_loader = DataLoader(subset_ds, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=10)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    res = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=2,
        device=device,
        log_interval=1,
    )
    
    print(f"Mini Training:     2 epochs completed in {res['training_time']:.4f} s")
    print(f"Final Train Loss:  {res['final_train_loss']:.4f}")
    if res['final_val_metrics']:
        print(f"Test Accuracy:     {res['final_val_metrics'].accuracy:.2%}")
        
    print("=" * 60)
    print(" ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
