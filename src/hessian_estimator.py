"""
Hessian estimation and trajectory averaging module.
==================================================

This module implements Hessian computation, trajectory averaging, and
posterior covariance construction for PAC-Bayes bounds.

Original Implementation:
- Trajectory-averaged Hessian computation
- Efficient Hessian-vector product computation
- Stochastic Hessian estimation for large models
- Posterior covariance construction with stability control

Adapted from:
- PyTorch autograd for Hessian computation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm


def compute_hessian_full(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute full Hessian matrix of empirical loss.
    
    WARNING: This is expensive for large models (O(d^2) memory).
    Use only for small models or debugging.
    
    Parameters:
    -----------
    model : nn.Module
        Model to compute Hessian for
    data_loader : DataLoader
        Data for loss computation
    criterion : nn.Module
        Loss function
    device : str
        Device to use
        
    Returns:
    --------
    hessian : torch.Tensor
        Full Hessian matrix (d x d)
    """
    model.eval()
    model.zero_grad()
    
    # Get parameter dimension
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    
    # Compute loss
    total_loss = 0.0
    for batch_X, batch_y in data_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss
    
    avg_loss = total_loss / len(data_loader)
    
    # Compute Hessian using autograd
    grads = torch.autograd.grad(avg_loss, params, create_graph=True)
    flat_grad = torch.cat([g.view(-1) for g in grads])
    
    hessian = torch.zeros(d, d, device=device)
    
    for i in range(d):
        grad2 = torch.autograd.grad(flat_grad[i], params, retain_graph=True)
        hessian[i] = torch.cat([g.view(-1) for g in grad2])
    
    return hessian


def compute_hessian_vector_product(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    vector: torch.Tensor,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute Hessian-vector product H @ v efficiently.
    
    Uses the trick: H @ v = grad(grad(loss) @ v)
    This avoids forming the full Hessian.
    
    Parameters:
    -----------
    model : nn.Module
        Model
    data_loader : DataLoader
        Data for loss
    criterion : nn.Module
        Loss function
    vector : torch.Tensor
        Vector to multiply (d,)
    device : str
        Device
        
    Returns:
    --------
    hv : torch.Tensor
        Hessian-vector product (d,)
    """
    model.eval()
    model.zero_grad()
    
    # Compute loss
    total_loss = 0.0
    for batch_X, batch_y in data_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss
    
    avg_loss = total_loss / len(data_loader)
    
    # First gradient
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(avg_loss, params, create_graph=True)
    flat_grad = torch.cat([g.view(-1) for g in grads])
    
    # Gradient-vector product
    gv = torch.sum(flat_grad * vector)
    
    # Second gradient (Hessian-vector product)
    hvs = torch.autograd.grad(gv, params)
    hv = torch.cat([g.view(-1) for g in hvs])
    
    return hv


def estimate_hessian_stochastic(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    num_samples: int = 10,
    device: str = "cpu",
    verbose: bool = False
) -> torch.Tensor:
    """
    Estimate Hessian diagonal using stochastic sampling.
    
    Uses Hutchinson's estimator with Hessian-vector products.
    Returns diagonal approximation.
    
    Parameters:
    -----------
    model : nn.Module
        Model
    data_loader : DataLoader
        Data
    criterion : nn.Module
        Loss
    num_samples : int
        Number of random samples for estimation
    device : str
        Device
    verbose : bool
        Show progress
        
    Returns:
    --------
    hess_diag : torch.Tensor
        Estimated Hessian diagonal (d,)
    """
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    
    hess_diag = torch.zeros(d, device=device)
    
    iterator = tqdm(range(num_samples), desc="Hessian estimation") if verbose else range(num_samples)
    
    for _ in iterator:
        # Sample random vector
        z = torch.randn(d, device=device)
        
        # Compute Hessian-vector product
        hz = compute_hessian_vector_product(model, data_loader, criterion, z, device)
        
        # Accumulate diagonal estimate
        hess_diag += z * hz
    
    hess_diag /= num_samples
    
    return hess_diag


class TrajectoryHessianEstimator:
    """
    Estimates trajectory-averaged Hessian and constructs posterior covariance.
    
    This is the core component for our trajectory-aware PAC-Bayes bounds.
    """
    
    def __init__(
        self,
        model: nn.Module,
        trajectory: List[torch.Tensor],
        data_loader: DataLoader,
        criterion: nn.Module,
        device: str = "cpu",
        lambda_reg: float = 1e-2,
        c_Q: float = 1.0,
        use_diagonal: bool = True,
        num_hess_samples: int = 50,
        verbose: bool = True
    ):
        """
        Initialize trajectory Hessian estimator.
        
        Parameters:
        -----------
        model : nn.Module
            Model (will be updated with trajectory parameters)
        trajectory : List[torch.Tensor]
            List of parameter vectors along trajectory
        data_loader : DataLoader
            Training data for Hessian computation
        criterion : nn.Module
            Loss function
        device : str
            Device
        lambda_reg : float
            Regularization parameter λ for stability
        c_Q : float
            Posterior scaling constant
        use_diagonal : bool
            If True, use diagonal Hessian approximation (faster)
        num_hess_samples : int
            Number of samples for stochastic Hessian estimation
        verbose : bool
            Print progress
        """
        self.model = model.to(device)
        self.trajectory = trajectory
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.lambda_reg = lambda_reg
        self.c_Q = c_Q
        self.use_diagonal = use_diagonal
        self.num_hess_samples = num_hess_samples
        self.verbose = verbose
        
        self.T = len(trajectory)
        self.d = trajectory[0].numel()
        
        # Compute trajectory-averaged Hessian
        self.H_bar = self._compute_trajectory_averaged_hessian()
        
        # Compute posterior covariance
        self.Sigma_Q = self._compute_posterior_covariance()
        
    def _compute_trajectory_averaged_hessian(self) -> torch.Tensor:
        """
        Compute trajectory-averaged Hessian: H_bar = (1/T) * sum_t H_t
        
        Returns:
        --------
        H_bar : torch.Tensor
            Trajectory-averaged Hessian (d,) for diagonal or (d,d) for full
        """
        if self.verbose:
            print(f"Computing trajectory-averaged Hessian (T={self.T}, d={self.d})...")
        
        if self.use_diagonal:
            H_bar = torch.zeros(self.d, device=self.device)
        else:
            H_bar = torch.zeros(self.d, self.d, device=self.device)
        
        iterator = tqdm(self.trajectory, desc="Trajectory Hessian") if self.verbose else self.trajectory
        
        for params_t in iterator:
            # Set model to trajectory point
            self.model.set_parameters_flat(params_t.to(self.device))
            
            # Compute Hessian at this point
            if self.use_diagonal:
                H_t = estimate_hessian_stochastic(
                    self.model,
                    self.data_loader,
                    self.criterion,
                    num_samples=self.num_hess_samples,
                    device=self.device,
                    verbose=False
                )
            else:
                H_t = compute_hessian_full(
                    self.model,
                    self.data_loader,
                    self.criterion,
                    device=self.device
                )
            
            # Accumulate
            H_bar += H_t
        
        # Average
        H_bar /= self.T
        
        return H_bar
    
    def _compute_posterior_covariance(self) -> torch.Tensor:
        """
        Compute posterior covariance: Sigma_Q = c_Q^2 * (H_bar + lambda*I)^{-1}
        
        Returns:
        --------
        Sigma_Q : torch.Tensor
            Posterior covariance (d,) for diagonal or (d,d) for full
        """
        if self.verbose:
            print("Computing posterior covariance...")
        
        if self.use_diagonal:
            # Diagonal approximation
            Sigma_Q = self.c_Q**2 / (self.H_bar + self.lambda_reg)
        else:
            # Full matrix
            H_reg = self.H_bar + self.lambda_reg * torch.eye(self.d, device=self.device)
            Sigma_Q = self.c_Q**2 * torch.linalg.inv(H_reg)
        
        return Sigma_Q
    
    def get_posterior_mean(self) -> torch.Tensor:
        """Get posterior mean (final trajectory point)."""
        return self.trajectory[-1]
    
    def get_posterior_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Gaussian posterior parameters.
        
        Returns:
        --------
        mu_Q : torch.Tensor
            Posterior mean (d,)
        Sigma_Q : torch.Tensor
            Posterior covariance (d,) or (d,d)
        """
        return self.get_posterior_mean(), self.Sigma_Q
