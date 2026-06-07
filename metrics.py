import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


def compute_grad_norm(model: nn.Module, norm_type: float = 2.0) -> Dict[str, float]:
    """
    Compute per-layer and global gradient norms.

    Args:
        model: PyTorch model after loss.backward()
        norm_type: Type of norm (default L2)

    Returns:
        Dict mapping layer names to their gradient norms,
        plus 'global' for the total norm across all layers.
    """
    norms = {}
    total_norm_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_norm = param.grad.data.norm(norm_type).item()
            norms[name] = layer_norm
            total_norm_sq += layer_norm ** norm_type

    norms["global"] = total_norm_sq ** (1.0 / norm_type)
    return norms


def compute_grad_cosine_similarity(
    grads_t: Dict[str, torch.Tensor],
    grads_t1: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute cosine similarity between gradients at two consecutive steps.
    Low values (near -1 or 0) indicate gradient direction instability.

    Args:
        grads_t:  Gradients at step t   {layer_name: grad_tensor}
        grads_t1: Gradients at step t+1 {layer_name: grad_tensor}

    Returns:
        Dict mapping layer names to cosine similarity values in [-1, 1].
    """
    similarities = {}

    for name in grads_t:
        if name not in grads_t1:
            continue
        g1 = grads_t[name].flatten().float()
        g2 = grads_t1[name].flatten().float()

        denom = (g1.norm() * g2.norm()).clamp(min=1e-8)
        sim = (g1 @ g2) / denom
        similarities[name] = sim.item()

    if similarities:
        similarities["mean"] = float(np.mean(list(similarities.values())))

    return similarities


def compute_linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA) between two
    representation matrices. Used to track how layer representations
    drift across training steps or curricula.

    CKA = 1.0 means representations are identical (up to linear transform).
    CKA ~ 0.0 means representations are unrelated.

    Args:
        X: Activation matrix [n_samples, dim_x]
        Y: Activation matrix [n_samples, dim_y]

    Returns:
        CKA similarity score in [0, 1].
    """
    X = X.float()
    Y = Y.float()

    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    gram_X = X @ X.T
    gram_Y = Y @ Y.T

    # HSIC
    hsic_xy = _hsic(gram_X, gram_Y)
    hsic_xx = _hsic(gram_X, gram_X)
    hsic_yy = _hsic(gram_Y, gram_Y)

    denom = (hsic_xx * hsic_yy) ** 0.5
    if denom < 1e-8:
        return 0.0

    return float(hsic_xy / denom)


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion (unbiased estimator)."""
    n = K.shape[0]
    K = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
    L = L - L.mean(dim=0, keepdim=True) - L.mean(dim=1, keepdim=True) + L.mean()
    return (K * L).sum() / ((n - 1) ** 2)


def compute_grad_norm_imbalance(grad_norms: Dict[str, float]) -> float:
    """
    Measure gradient norm imbalance across layers.
    High imbalance = some layers updating much faster than others,
    a key diagnostic from curriculum learning research.

    Returns:
        Ratio of max layer norm to min layer norm (excluding 'global').
        Values >> 1 indicate imbalance.
    """
    layer_norms = [v for k, v in grad_norms.items() if k != "global" and v > 1e-10]
    if len(layer_norms) < 2:
        return 1.0
    return max(layer_norms) / min(layer_norms)