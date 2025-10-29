"""
PAC-Bayes bounds computation module.
====================================

This module implements the trajectory-aware PAC-Bayes bound with
KL decomposition into top-subspace alignment, residual complexity,
and trajectory curvature terms.

Original Implementation:
- Gaussian KL decomposition with top-subspace projection
- Trajectory curvature term computation
- Stochastic error control integration
- Complete PAC-Bayes bound computation

Based on theory from:
- Paper "Trajectory-Aware PAC-Bayes Bounds with High-Dimensional Stochastic Error Control"
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from .utils import (
    compute_top_eigenvectors,
    projection_matrix,
    quadratic_form,
    safe_log_det,
    hutchinson_trace_estimator,
    hutchinson_log_det_estimator
)


class GaussianPACBayesBound:
    """
    Computes PAC-Bayes bounds for Gaussian posteriors with trajectory-aware curvature.
    """
    
    def __init__(
        self,
        mu_P: torch.Tensor,
        Sigma_P: torch.Tensor,
        mu_Q: torch.Tensor,
        Sigma_Q: torch.Tensor,
        r: int = 10,
        device: str = "cpu",
        use_diagonal: bool = True
    ):
        """
        Initialize Gaussian PAC-Bayes bound computer.
        
        Parameters:
        -----------
        mu_P : torch.Tensor
            Prior mean (d,)
        Sigma_P : torch.Tensor
            Prior covariance (d,) or (d, d)
        mu_Q : torch.Tensor
            Posterior mean (d,)
        Sigma_Q : torch.Tensor
            Posterior covariance (d,) or (d, d)
        r : int
            Dimension for top-subspace projection
        device : str
            Device
        use_diagonal : bool
            Whether using diagonal approximation
        """
        self.mu_P = mu_P.to(device)
        self.Sigma_P = Sigma_P.to(device)
        self.mu_Q = mu_Q.to(device)
        self.Sigma_Q = Sigma_Q.to(device)
        self.r = r
        self.device = device
        self.use_diagonal = use_diagonal
        self.d = mu_P.shape[0]
        
    def compute_gaussian_kl(self) -> float:
        """
        Compute exact Gaussian KL divergence KL(Q||P).
        
        KL(Q||P) = 1/2 * [tr(Sigma_P^{-1} Sigma_Q) + (mu_P - mu_Q)^T Sigma_P^{-1} (mu_P - mu_Q) 
                          - d + log(det(Sigma_P)/det(Sigma_Q))]
        
        Returns:
        --------
        kl : float
            KL divergence
        """
        if self.use_diagonal:
            # Diagonal approximation
            trace_term = torch.sum(self.Sigma_Q / self.Sigma_P).item()
            
            diff = self.mu_P - self.mu_Q
            mahalanobis_term = torch.sum((diff ** 2) / self.Sigma_P).item()
            
            log_det_term = torch.sum(torch.log(self.Sigma_P) - torch.log(self.Sigma_Q)).item()
        else:
            # Full matrix
            Sigma_P_inv = torch.linalg.inv(self.Sigma_P)
            
            trace_term = torch.trace(torch.mm(Sigma_P_inv, self.Sigma_Q)).item()
            
            diff = self.mu_P - self.mu_Q
            mahalanobis_term = quadratic_form(diff, Sigma_P_inv)
            
            log_det_P = safe_log_det(self.Sigma_P)
            log_det_Q = safe_log_det(self.Sigma_Q)
            log_det_term = log_det_P - log_det_Q
        
        kl = 0.5 * (trace_term + mahalanobis_term - self.d + log_det_term)
        
        return kl
    
    def compute_kl_decomposition(
        self,
        num_hutchinson_samples: int = 100
    ) -> Dict[str, float]:
        """
        Decompose KL divergence into interpretable components.
        
        Components:
        - top_subspace_alignment: ||Pi_r (mu_Q - mu_P)||^2_{Sigma_P^{-1}}
        - residual_complexity: tr(Sigma_P^{-1} Sigma_Q) - tr(Pi_r Sigma_P^{-1} Sigma_Q Pi_r)
        - dimension_term: -d
        - log_det_term: log(det(Sigma_P)/det(Sigma_Q))
        
        Parameters:
        -----------
        num_hutchinson_samples : int
            Number of samples for stochastic estimation
            
        Returns:
        --------
        decomposition : Dict[str, float]
            Dictionary with KL components
        """
        decomposition = {}
        
        if self.use_diagonal:
            # For diagonal, we skip subspace decomposition
            # Just compute standard terms
            diff = self.mu_P - self.mu_Q
            decomposition['mean_alignment'] = torch.sum((diff ** 2) / self.Sigma_P).item()
            decomposition['trace_term'] = torch.sum(self.Sigma_Q / self.Sigma_P).item()
            decomposition['dimension_term'] = -self.d
            decomposition['log_det_term'] = torch.sum(
                torch.log(self.Sigma_P) - torch.log(self.Sigma_Q)
            ).item()
            decomposition['total_kl'] = 0.5 * sum(decomposition.values())
            
        else:
            # Full matrix decomposition
            
            # 1. Compute top-r eigenvectors of Sigma_P
            V_r = compute_top_eigenvectors(self.Sigma_P, self.r)
            Pi_r = projection_matrix(V_r)
            Pi_r_complement = torch.eye(self.d, device=self.device) - Pi_r
            
            # 2. Top-subspace alignment
            diff = self.mu_Q - self.mu_P
            Pi_r_diff = torch.mv(Pi_r, diff)
            Sigma_P_inv = torch.linalg.inv(self.Sigma_P)
            top_alignment = quadratic_form(Pi_r_diff, Sigma_P_inv)
            decomposition['top_subspace_alignment'] = top_alignment
            
            # 3. Complement alignment
            Pi_r_c_diff = torch.mv(Pi_r_complement, diff)
            complement_alignment = quadratic_form(Pi_r_c_diff, Sigma_P_inv)
            decomposition['complement_alignment'] = complement_alignment
            
            # 4. Trace decomposition
            Sigma_P_inv_Sigma_Q = torch.mm(Sigma_P_inv, self.Sigma_Q)
            full_trace = torch.trace(Sigma_P_inv_Sigma_Q).item()
            
            # Top-subspace trace
            top_trace = torch.trace(
                torch.mm(torch.mm(Pi_r, Sigma_P_inv_Sigma_Q), Pi_r)
            ).item()
            
            # Residual trace
            residual_trace = full_trace - top_trace
            
            decomposition['trace_full'] = full_trace
            decomposition['trace_top_subspace'] = top_trace
            decomposition['trace_residual'] = residual_trace
            
            # 5. Log-det term
            log_det_P = safe_log_det(self.Sigma_P)
            log_det_Q = safe_log_det(self.Sigma_Q)
            decomposition['log_det_term'] = log_det_P - log_det_Q
            
            # 6. Dimension term
            decomposition['dimension_term'] = -self.d
            
            # 7. Total KL
            decomposition['total_kl'] = 0.5 * (
                top_alignment + complement_alignment + full_trace 
                - self.d + log_det_P - log_det_Q
            )
        
        return decomposition
    
    def compute_pac_bayes_bound(
        self,
        empirical_risk: float,
        n_samples: int,
        delta: float = 0.05,
        epsilon_stochastic: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute PAC-Bayes bound.
        
        Bound: E_{w~Q}[R(w)] <= R_S(Q) + sqrt((KL(Q||P) + log(1/delta) + epsilon) / (2n))
        
        Parameters:
        -----------
        empirical_risk : float
            Empirical risk R_S(Q) on training data
        n_samples : int
            Number of training samples
        delta : float
            Confidence parameter
        epsilon_stochastic : float
            Stochastic estimation error
            
        Returns:
        --------
        bound_info : Dict[str, float]
            Dictionary with bound components
        """
        kl = self.compute_gaussian_kl()
        
        numerator = kl + np.log(1.0 / delta) + epsilon_stochastic
        bound_term = np.sqrt(numerator / (2 * n_samples))
        
        bound = empirical_risk + bound_term
        
        return {
            'empirical_risk': empirical_risk,
            'kl_divergence': kl,
            'stochastic_error': epsilon_stochastic,
            'confidence_term': np.log(1.0 / delta),
            'bound_term': bound_term,
            'pac_bayes_bound': bound,
            'n_samples': n_samples,
            'delta': delta
        }


class TrajectoryCurvatureTerm:
    """
    Computes trajectory curvature contribution to KL divergence.
    
    This implements the stochastic log-determinant term:
    FD_T = (1/T) * sum_t (1/m) * sum_i z_{t,i}^T log(I + lambda^{-1} H_t) z_{t,i}
    """
    
    def __init__(
        self,
        hessians: list,
        lambda_reg: float = 1e-2,
        num_samples: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize trajectory curvature term computer.
        
        Parameters:
        -----------
        hessians : list
            List of Hessian matrices (or diagonals) along trajectory
        lambda_reg : float
            Regularization parameter λ
        num_samples : int
            Number of Hutchinson samples m
        device : str
            Device
        """
        self.hessians = hessians
        self.lambda_reg = lambda_reg
        self.num_samples = num_samples
        self.device = device
        self.T = len(hessians)
        
    def compute_curvature_term(self) -> Tuple[float, float]:
        """
        Compute trajectory curvature term with variance.
        
        Returns:
        --------
        curvature_term : float
            Estimated curvature contribution
        variance : float
            Variance of estimate
        """
        estimates = []
        
        for H_t in self.hessians:
            H_t = H_t.to(self.device)
            
            # Compute I + lambda^{-1} H_t
            if H_t.ndim == 1:
                # Diagonal
                matrix = 1.0 + H_t / self.lambda_reg
                # For diagonal, log is element-wise
                log_matrix = torch.log(matrix)
                # Trace is just sum of diagonal
                trace_log = torch.sum(log_matrix).item()
            else:
                # Full matrix
                d = H_t.shape[0]
                matrix = torch.eye(d, device=self.device) + H_t / self.lambda_reg
                
                # Use Hutchinson for log-determinant
                trace_log, _ = hutchinson_log_det_estimator(
                    matrix,
                    num_samples=self.num_samples,
                    return_variance=False
                )
            
            estimates.append(trace_log)
        
        curvature_term = np.mean(estimates)
        variance = np.var(estimates)
        
        return curvature_term, variance


def compute_baseline_pac_bayes_bound(
    empirical_risk: float,
    n_samples: int,
    d: int,
    delta: float = 0.05,
    prior_variance: float = 1.0,
    posterior_variance: float = 0.01
) -> Dict[str, float]:
    """
    Compute baseline PAC-Bayes bound with isotropic Gaussian.
    
    Prior: N(0, prior_variance * I)
    Posterior: N(mu_Q, posterior_variance * I)
    
    Parameters:
    -----------
    empirical_risk : float
        Empirical risk
    n_samples : int
        Number of samples
    d : int
        Dimension
    delta : float
        Confidence
    prior_variance : float
        Prior variance
    posterior_variance : float
        Posterior variance
        
    Returns:
    --------
    bound_info : Dict[str, float]
    """
    # Simplified KL for isotropic Gaussians
    # KL = d/2 * [posterior_var/prior_var - 1 - log(posterior_var/prior_var)]
    kl = 0.5 * d * (
        posterior_variance / prior_variance 
        - 1 
        - np.log(posterior_variance / prior_variance)
    )
    
    numerator = kl + np.log(1.0 / delta)
    bound_term = np.sqrt(numerator / (2 * n_samples))
    bound = empirical_risk + bound_term
    
    return {
        'empirical_risk': empirical_risk,
        'kl_divergence': kl,
        'bound_term': bound_term,
        'pac_bayes_bound': bound,
        'method': 'baseline_isotropic'
    }
