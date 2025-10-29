"""
Utility functions module.
=========================

This module provides utility functions for PAC-Bayes bound computation.

Original Implementation:
- Hutchinson trace estimator with error control
- Hutchinson log-determinant estimator (heuristic)
- Matrix function utilities for bound computation
- Stochastic error estimation

Adapted from:
- Standard Hutchinson estimator algorithm
"""

import torch
import numpy as np
from typing import Tuple, Optional


def hutchinson_trace_estimator(
    matrix: torch.Tensor,
    num_samples: int = 100,
    return_variance: bool = False
) -> float or Tuple[float, float]:
    """
    Estimate trace of a matrix using Hutchinson's estimator.
    
    tr(A) ≈ (1/m) * sum_i z_i^T A z_i, where z_i ~ N(0, I)
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Matrix to estimate trace of (d x d)
    num_samples : int
        Number of random samples m
    return_variance : bool
        If True, also return variance estimate
        
    Returns:
    --------
    trace_est : float
        Trace estimate
    variance : float (optional)
        Variance of estimate
    """
    d = matrix.shape[0]
    device = matrix.device
    
    estimates = []
    
    for _ in range(num_samples):
        # Sample random vector
        z = torch.randn(d, device=device)
        
        # Compute z^T A z
        Az = torch.mv(matrix, z)
        estimate = torch.dot(z, Az).item()
        estimates.append(estimate)
    
    trace_est = np.mean(estimates)
    
    if return_variance:
        variance = np.var(estimates)
        return trace_est, variance
    else:
        return trace_est


def hutchinson_log_det_estimator(
    matrix: torch.Tensor,
    num_samples: int = 100,
    return_variance: bool = False
) -> float or Tuple[float, float]:
    """
    Estimate tr(log(A)) using Hutchinson's estimator.
    
    CAUTION: This is a heuristic extension. For log(A), we compute:
    tr(log(A)) ≈ (1/m) * sum_i z_i^T log(A) z_i
    
    This requires matrix logarithm, which is expensive.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Positive definite matrix (d x d)
    num_samples : int
        Number of samples
    return_variance : bool
        Return variance estimate
        
    Returns:
    --------
    trace_log_est : float
        Estimate of tr(log(A))
    variance : float (optional)
        Variance
    """
    d = matrix.shape[0]
    device = matrix.device
    
    # Compute matrix logarithm (expensive!)
    # Move to CPU for scipy if needed
    if device.type == 'cuda':
        matrix_cpu = matrix.cpu()
        log_matrix = torch.tensor(
            scipy.linalg.logm(matrix_cpu.numpy()),
            dtype=torch.float32
        ).to(device)
    else:
        import scipy.linalg
        log_matrix = torch.tensor(
            scipy.linalg.logm(matrix.numpy()),
            dtype=torch.float32,
            device=device
        )
    
    estimates = []
    
    for _ in range(num_samples):
        z = torch.randn(d, device=device)
        log_Az = torch.mv(log_matrix, z)
        estimate = torch.dot(z, log_Az).item()
        estimates.append(estimate)
    
    trace_log_est = np.mean(estimates)
    
    if return_variance:
        variance = np.var(estimates)
        return trace_log_est, variance
    else:
        return trace_log_est


def compute_top_eigenvectors(
    matrix: torch.Tensor,
    r: int
) -> torch.Tensor:
    """
    Compute top-r eigenvectors of a matrix.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Symmetric matrix (d x d)
    r : int
        Number of top eigenvectors
        
    Returns:
    --------
    V_r : torch.Tensor
        Top-r eigenvectors (d x r)
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # Sort in descending order
    indices = torch.argsort(eigenvalues, descending=True)
    top_indices = indices[:r]
    
    V_r = eigenvectors[:, top_indices]
    
    return V_r


def projection_matrix(V_r: torch.Tensor) -> torch.Tensor:
    """
    Compute projection matrix Pi_r = V_r V_r^T
    
    Parameters:
    -----------
    V_r : torch.Tensor
        Orthonormal basis vectors (d x r)
        
    Returns:
    --------
    Pi_r : torch.Tensor
        Projection matrix (d x d)
    """
    return torch.mm(V_r, V_r.T)


def operator_norm(matrix: torch.Tensor) -> float:
    """
    Compute operator norm (largest singular value) of a matrix.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Matrix (d x d)
        
    Returns:
    --------
    norm : float
        Operator norm
    """
    singular_values = torch.linalg.svdvals(matrix)
    return singular_values[0].item()


def condition_number(matrix: torch.Tensor) -> float:
    """
    Compute condition number of a matrix.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Matrix (d x d)
        
    Returns:
    --------
    cond : float
        Condition number (ratio of largest to smallest singular value)
    """
    singular_values = torch.linalg.svdvals(matrix)
    return (singular_values[0] / singular_values[-1]).item()


def safe_log_det(matrix: torch.Tensor, epsilon: float = 1e-10) -> float:
    """
    Safely compute log determinant.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Positive definite matrix (d x d)
    epsilon : float
        Small value to add to diagonal for numerical stability
        
    Returns:
    --------
    log_det : float
        Log determinant
    """
    # Add epsilon to diagonal for stability
    d = matrix.shape[0]
    matrix_stable = matrix + epsilon * torch.eye(d, device=matrix.device)
    
    # Compute log determinant via Cholesky
    try:
        L = torch.linalg.cholesky(matrix_stable)
        log_det = 2 * torch.sum(torch.log(torch.diag(L))).item()
    except RuntimeError:
        # Fall back to eigenvalue method
        eigenvalues = torch.linalg.eigvalsh(matrix_stable)
        log_det = torch.sum(torch.log(eigenvalues)).item()
    
    return log_det


def quadratic_form(
    vector: torch.Tensor,
    matrix: torch.Tensor
) -> float:
    """
    Compute v^T M v efficiently.
    
    Parameters:
    -----------
    vector : torch.Tensor
        Vector (d,)
    matrix : torch.Tensor
        Matrix (d x d)
        
    Returns:
    --------
    quad_form : float
        v^T M v
    """
    Mv = torch.mv(matrix, vector)
    return torch.dot(vector, Mv).item()


def set_random_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
