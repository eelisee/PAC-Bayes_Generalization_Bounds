"""
PAC-Bayes Generalization Bounds Implementation
===============================================

This package implements trajectory-aware PAC-Bayes bounds with high-dimensional
stochastic error control for deep learning models.

Modules:
--------
- data_loader: Dataset loading and preprocessing utilities
- models: Model architectures (logistic regression, small MLPs)
- training: Training loop with trajectory recording
- hessian_estimator: Hessian computation and trajectory averaging
- pac_bayes_bounds: PAC-Bayes bound computation with KL decomposition
- utils: Utility functions (Hutchinson estimator, etc.)

Original Implementation:
-----------------------
This implementation is original work for the Advanced Machine Learning course
at ENSAE Paris. Key novel contributions:
- Trajectory-averaged Hessian estimation
- Gaussian KL decomposition into top-subspace, residual, and curvature terms
- Stochastic trace and log-determinant estimation with error control
- Integration of these components into a unified PAC-Bayes framework

Adapted Components:
------------------
- Standard PyTorch training loops (adapted with trajectory recording)
- Standard MNIST data loading from torchvision
- Standard Hutchinson trace estimator (adapted for log-determinant)
"""

__version__ = "0.1.0"
__author__ = "Team Name"
__email__ = "team@example.com"

from . import data_loader
from . import models
from . import training
from . import hessian_estimator
from . import pac_bayes_bounds
from . import utils

__all__ = [
    "data_loader",
    "models",
    "training",
    "hessian_estimator",
    "pac_bayes_bounds",
    "utils",
]
