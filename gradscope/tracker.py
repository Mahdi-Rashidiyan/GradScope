"""
gradscope/tracker.py
GradTracker: attaches to any PyTorch model and records gradient
diagnostics at every training step with zero boilerplate.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import warnings

from .metrics import (
    compute_grad_norm,
    compute_grad_cosine_similarity,
    compute_grad_norm_imbalance,
    compute_linear_cka,
)


class GradTracker:
    """
    One-line gradient diagnostics for any PyTorch model.

    Usage:
        tracker = GradTracker(model)
        tracker.attach()

        for batch in dataloader:
            loss = criterion(model(x), y)
            loss.backward()
            tracker.step()          # call after backward, before optimizer
            optimizer.step()
            optimizer.zero_grad()

        tracker.plot()              # visualize everything
        report = tracker.summary()  # get dict of all recorded metrics
    """

    def __init__(
        self,
        model: nn.Module,
        track_cosine: bool = True,
        track_cka: bool = False,
        cka_layers: Optional[List[str]] = None,
        warn_imbalance_threshold: float = 100.0,
        warn_vanishing_threshold: float = 1e-5,
        warn_exploding_threshold: float = 1e3,
    ):
        """
        Args:
            model: Your PyTorch model.
            track_cosine: Whether to track gradient cosine similarity between steps.
            track_cka: Whether to track linear CKA on layer activations.
            cka_layers: Layer names to track CKA on. If None, auto-selects last 3 linear layers.
            warn_imbalance_threshold: Warn if max/min norm ratio exceeds this.
            warn_vanishing_threshold: Warn if global grad norm drops below this.
            warn_exploding_threshold: Warn if global grad norm exceeds this.
        """
        self.model = model
        self.track_cosine = track_cosine
        self.track_cka = track_cka
        self.cka_layers = cka_layers
        self.warn_imbalance_threshold = warn_imbalance_threshold
        self.warn_vanishing_threshold = warn_vanishing_threshold
        self.warn_exploding_threshold = warn_exploding_threshold

        # Storage
        self.history: Dict[str, List] = defaultdict(list)
        self._prev_grads: Optional[Dict[str, torch.Tensor]] = None
        self._activation_buffer: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._step_count = 0
        self._attached = False

    def attach(self) -> "GradTracker":
        """
        Register forward hooks for CKA tracking (if enabled).
        Safe to call multiple times — won't duplicate hooks.
        """
        if self._attached:
            return self

        if self.track_cka:
            target_layers = self._resolve_cka_layers()
            for name, module in self.model.named_modules():
                if name in target_layers:
                    hook = module.register_forward_hook(self._make_activation_hook(name))
                    self._hooks.append(hook)

        self._attached = True
        return self

    def step(self) -> Dict:
        """
        Call this after loss.backward() and before optimizer.step().
        Records all enabled metrics for the current step.

        Returns:
            Dict of metrics recorded at this step.
        """
        step_metrics = {"step": self._step_count}

        # --- Gradient Norms ---
        grad_norms = compute_grad_norm(self.model)
        step_metrics["grad_norm_global"] = grad_norms["global"]
        step_metrics["grad_norms"] = grad_norms

        imbalance = compute_grad_norm_imbalance(grad_norms)
        step_metrics["grad_norm_imbalance"] = imbalance

        self.history["grad_norm_global"].append(grad_norms["global"])
        self.history["grad_norm_imbalance"].append(imbalance)

        # --- Warnings ---
        self._check_warnings(grad_norms["global"], imbalance)

        # --- Cosine Similarity ---
        if self.track_cosine and self._prev_grads is not None:
            current_grads = self._snapshot_grads()
            cos_sim = compute_grad_cosine_similarity(self._prev_grads, current_grads)
            step_metrics["grad_cosine_mean"] = cos_sim.get("mean", float("nan"))
            self.history["grad_cosine_mean"].append(cos_sim.get("mean", float("nan")))

        # Snapshot grads for next step's cosine computation
        if self.track_cosine:
            self._prev_grads = self._snapshot_grads()

        # --- CKA ---
        if self.track_cka and len(self._activation_buffer) >= 2:
            layer_names = list(self._activation_buffer.keys())
            for i in range(len(layer_names) - 1):
                name_a, name_b = layer_names[i], layer_names[i + 1]
                X = self._activation_buffer[name_a]
                Y = self._activation_buffer[name_b]
                if X.shape[0] == Y.shape[0]:
                    cka_key = f"cka_{name_a}_vs_{name_b}"
                    cka_val = compute_linear_cka(X, Y)
                    step_metrics[cka_key] = cka_val
                    self.history[cka_key].append(cka_val)

        self._step_count += 1
        return step_metrics

    def summary(self) -> Dict:
        """
        Return a summary of all tracked metrics across training.
        Includes mean, min, max for each metric.
        """
        import numpy as np
        report = {"total_steps": self._step_count, "metrics": {}}

        for key, values in self.history.items():
            arr = [v for v in values if v == v]  # filter NaN
            if not arr:
                continue
            report["metrics"][key] = {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "final": float(arr[-1]),
            }

        # Diagnosis
        report["diagnosis"] = self._diagnose()
        return report

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Render a training dynamics dashboard.
        Calls viz.py under the hood.
        """
        from .viz import plot_dashboard
        plot_dashboard(self.history, save_path=save_path)

    def reset(self) -> None:
        """Clear all recorded history and start fresh."""
        self.history = defaultdict(list)
        self._prev_grads = None
        self._activation_buffer = {}
        self._step_count = 0

    def detach(self) -> None:
        """Remove all registered hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attached = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _snapshot_grads(self) -> Dict[str, torch.Tensor]:
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.data.clone()
        return grads

    def _make_activation_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                act = output.detach()
                # Flatten spatial dims, keep batch
                self._activation_buffer[name] = act.view(act.shape[0], -1)
        return hook

    def _resolve_cka_layers(self) -> List[str]:
        if self.cka_layers:
            return self.cka_layers
        # Auto-select last 3 linear/conv layers
        linear_layers = [
            name for name, m in self.model.named_modules()
            if isinstance(m, (nn.Linear, nn.Conv2d))
        ]
        return linear_layers[-3:] if len(linear_layers) >= 3 else linear_layers

    def _check_warnings(self, global_norm: float, imbalance: float) -> None:
        if global_norm < self.warn_vanishing_threshold:
            warnings.warn(
                f"[GradScope] ⚠️  Vanishing gradients detected at step {self._step_count}. "
                f"Global norm = {global_norm:.2e}",
                RuntimeWarning,
                stacklevel=3,
            )
        if global_norm > self.warn_exploding_threshold:
            warnings.warn(
                f"[GradScope] 💥 Exploding gradients at step {self._step_count}. "
                f"Global norm = {global_norm:.2e}",
                RuntimeWarning,
                stacklevel=3,
            )
        if imbalance > self.warn_imbalance_threshold:
            warnings.warn(
                f"[GradScope] 📊 Gradient norm imbalance = {imbalance:.1f}x at step {self._step_count}. "
                f"Some layers are updating much faster than others.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _diagnose(self) -> Dict[str, str]:
        diagnosis = {}

        norms = self.history.get("grad_norm_global", [])
        if norms:
            if norms[-1] < self.warn_vanishing_threshold:
                diagnosis["gradient_flow"] = "VANISHING — consider higher LR or skip connections"
            elif norms[-1] > self.warn_exploding_threshold:
                diagnosis["gradient_flow"] = "EXPLODING — consider gradient clipping"
            else:
                diagnosis["gradient_flow"] = "HEALTHY"

        imbalances = self.history.get("grad_norm_imbalance", [])
        if imbalances:
            avg_imbalance = sum(imbalances) / len(imbalances)
            if avg_imbalance > self.warn_imbalance_threshold:
                diagnosis["layer_balance"] = f"IMBALANCED ({avg_imbalance:.1f}x) — curriculum may be locking capacity"
            else:
                diagnosis["layer_balance"] = "BALANCED"

        cosines = self.history.get("grad_cosine_mean", [])
        if cosines:
            recent = [c for c in cosines[-20:] if c == c]
            if recent:
                avg_cos = sum(recent) / len(recent)
                if avg_cos < 0.1:
                    diagnosis["gradient_direction"] = "UNSTABLE — high directional variance between steps"
                elif avg_cos > 0.9:
                    diagnosis["gradient_direction"] = "STABLE (possibly over-smooth — check for underfitting)"
                else:
                    diagnosis["gradient_direction"] = "NORMAL"

        return diagnosis
