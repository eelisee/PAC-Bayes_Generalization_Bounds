"""
Model architectures module.
===========================

This module defines model architectures for binary classification experiments.

Original Implementation:
- Custom logistic regression with parameter tracking
- Small MLP architectures designed for PAC-Bayes experiments
- Parameter extraction utilities for trajectory recording

Adapted from:
- Standard PyTorch module patterns
"""

import torch
import torch.nn as nn
from typing import List


class LogisticRegression(nn.Module):
    """
    Logistic Regression model for binary classification.
    
    Simple linear model: y = sigmoid(w^T x + b)
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize logistic regression model.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension (e.g., 784 for flattened MNIST)
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sigmoid(self.linear(x))
    
    def get_parameters_flat(self) -> torch.Tensor:
        """Get all parameters as a flat vector."""
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def set_parameters_flat(self, flat_params: torch.Tensor):
        """Set parameters from a flat vector."""
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data = flat_params[offset:offset+numel].view(param.shape)
            offset += numel


class SmallMLP(nn.Module):
    """
    Small Multi-Layer Perceptron for binary classification.
    
    Architecture: input -> hidden1 -> hidden2 -> ... -> output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = "relu"
    ):
        """
        Initialize small MLP.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        hidden_dims : List[int]
            List of hidden layer dimensions
        activation : str
            Activation function: 'relu', 'tanh', or 'sigmoid'
        """
        super(SmallMLP, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def get_parameters_flat(self) -> torch.Tensor:
        """Get all parameters as a flat vector."""
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def set_parameters_flat(self, flat_params: torch.Tensor):
        """Set parameters from a flat vector."""
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data = flat_params[offset:offset+numel].view(param.shape)
            offset += numel


def create_model(
    model_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Parameters:
    -----------
    model_type : str
        Type of model: 'logistic' or 'mlp'
    input_dim : int
        Input feature dimension
    **kwargs : dict
        Additional arguments for model construction
        
    Returns:
    --------
    model : nn.Module
    """
    if model_type == "logistic":
        return LogisticRegression(input_dim)
    elif model_type == "mlp":
        hidden_dims = kwargs.get("hidden_dims", [128, 64])
        activation = kwargs.get("activation", "relu")
        return SmallMLP(input_dim, hidden_dims, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())
