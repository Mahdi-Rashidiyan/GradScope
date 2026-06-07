"""
GradScope — Training Dynamics Diagnostics for PyTorch
"""

from .tracker import GradTracker
from .metrics import (
    compute_grad_norm,
    compute_grad_cosine_similarity,
    compute_linear_cka,
    compute_grad_norm_imbalance,
)
from .viz import plot_dashboard, plot_grad_norms_per_layer

__version__ = "0.1.0"
__author__ = "Mahdi"

__all__ = [
    "GradTracker",
    "compute_grad_norm",
    "compute_grad_cosine_similarity",
    "compute_linear_cka",
    "compute_grad_norm_imbalance",
    "plot_dashboard",
    "plot_grad_norms_per_layer",
]
