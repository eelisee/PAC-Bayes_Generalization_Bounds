"""
Training module with trajectory recording.
=========================================

This module implements training loops with parameter trajectory recording
for PAC-Bayes bound computation.

Original Implementation:
- Trajectory recording mechanism during training
- Integration with Hessian estimation
- Epoch-wise and mini-batch-wise snapshot capabilities
- Support for multiple random seeds

Adapted from:
- Standard PyTorch training loop structure
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import numpy as np
from tqdm import tqdm


class TrajectoryTrainer:
    """
    Trainer class that records parameter trajectory during optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        record_every: int = 1,
        verbose: bool = True
    ):
        """
        Initialize trajectory trainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train
        optimizer : optim.Optimizer
            Optimizer (e.g., SGD, Adam)
        criterion : nn.Module
            Loss function (e.g., BCELoss)
        device : str
            Device to use ('cpu' or 'cuda')
        record_every : int
            Record trajectory every N epochs
        verbose : bool
            Print progress
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.record_every = record_every
        self.verbose = verbose
        
        # Trajectory storage
        self.trajectory: List[torch.Tensor] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * len(batch_X)
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += len(batch_y)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> tuple:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * len(batch_X)
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += len(batch_y)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100
    ) -> Dict:
        """
        Train model and record trajectory.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : Optional[DataLoader]
            Validation data loader
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        history : Dict
            Training history with trajectory
        """
        if self.verbose:
            pbar = tqdm(range(epochs), desc="Training")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
            
            # Record trajectory
            if (epoch + 1) % self.record_every == 0:
                params = self.model.get_parameters_flat().cpu().clone()
                self.trajectory.append(params)
            
            # Update progress
            if self.verbose:
                if val_loader is not None:
                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'train_acc': f'{train_acc:.4f}',
                        'val_acc': f'{val_acc:.4f}'
                    })
                else:
                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'train_acc': f'{train_acc:.4f}'
                    })
        
        # Final trajectory point
        if len(self.trajectory) == 0 or (epochs % self.record_every != 0):
            params = self.model.get_parameters_flat().cpu().clone()
            self.trajectory.append(params)
        
        history = {
            'trajectory': self.trajectory,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'final_params': self.model.get_parameters_flat().cpu().clone()
        }
        
        return history
    
    def compute_empirical_risk(
        self,
        data_loader: DataLoader
    ) -> float:
        """
        Compute empirical risk on dataset.
        
        Parameters:
        -----------
        data_loader : DataLoader
            Data loader for risk computation
            
        Returns:
        --------
        risk : float
            Average loss over dataset
        """
        self.model.eval()
        total_loss = 0.0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * len(batch_X)
                total += len(batch_y)
        
        return total_loss / total


def train_model_with_trajectory(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    lr: float = 0.01,
    optimizer_type: str = "sgd",
    device: str = "cpu",
    record_every: int = 1,
    seed: Optional[int] = None,
    verbose: bool = True
) -> tuple:
    """
    Convenience function to train a model and record trajectory.
    
    Parameters:
    -----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : Optional[DataLoader]
        Validation data loader
    epochs : int
        Number of epochs
    lr : float
        Learning rate
    optimizer_type : str
        Optimizer type: 'sgd' or 'adam'
    device : str
        Device to use
    record_every : int
        Record trajectory every N epochs
    seed : Optional[int]
        Random seed
    verbose : bool
        Print progress
        
    Returns:
    --------
    model, history : trained model and history dict
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Setup optimizer
    if optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Setup loss
    criterion = nn.BCELoss()
    
    # Train
    trainer = TrajectoryTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        record_every=record_every,
        verbose=verbose
    )
    
    history = trainer.train(train_loader, val_loader, epochs)
    
    return model, history
