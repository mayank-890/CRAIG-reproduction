"""Logistic regression model for flattened image classification."""

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Multinomial Logistic Regression (single linear layer with bias).
    
    Transforms flattened inputs [batch_size, input_dim] to logits [batch_size, num_classes].
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small standard normal and zero biases."""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [B, input_dim] or [B, 1, 28, 28].
            
        Returns:
            Logits of shape [B, num_classes].
        """
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
