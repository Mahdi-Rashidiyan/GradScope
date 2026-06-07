
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


def plot_dashboard(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    style: str = "dark",
) -> None:
    """
    Render the full GradScope training dynamics dashboard.

    Panels:
      1. Global gradient norm over steps
      2. Gradient norm imbalance over steps
      3. Gradient cosine similarity over steps (if tracked)
      4. CKA similarity over steps (if tracked)

    Args:
        history: tracker.history dict
        save_path: If provided, saves figure to this path instead of showing
        figsize: Figure dimensions
        style: 'dark' or 'light'
    """
    _apply_style(style)

    has_cosine = "grad_cosine_mean" in history and len(history["grad_cosine_mean"]) > 0
    cka_keys = [k for k in history if k.startswith("cka_")]
    has_cka = len(cka_keys) > 0

    n_panels = 2 + int(has_cosine) + int(has_cka)
    fig = plt.figure(figsize=figsize)
    fig.suptitle("GradScope — Training Dynamics Report", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[0, 1]))
    if has_cosine:
        axes.append(fig.add_subplot(gs[1, 0]))
    if has_cka:
        axes.append(fig.add_subplot(gs[1, 1]))

    # Panel 1 — Global Gradient Norm
    ax = axes[0]
    norms = history.get("grad_norm_global", [])
    if norms:
        steps = list(range(len(norms)))
        ax.plot(steps, norms, color="#4FC3F7", linewidth=1.5, label="Global norm")
        _smooth_overlay(ax, norms, steps, color="#0288D1")
        ax.set_title("Global Gradient Norm", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("L2 Norm")
        ax.set_yscale("log")
        _add_threshold_line(ax, 1e-5, "Vanishing", "#EF5350")
        _add_threshold_line(ax, 1e3, "Exploding", "#FF7043")
        ax.legend(fontsize=8)

    # Panel 2 — Norm Imbalance
    ax = axes[1]
    imbalance = history.get("grad_norm_imbalance", [])
    if imbalance:
        steps = list(range(len(imbalance)))
        colors = ["#EF5350" if v > 100 else "#66BB6A" for v in imbalance]
        ax.bar(steps, imbalance, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=100, color="#EF5350", linestyle="--", linewidth=1, label="Imbalance threshold")
        ax.set_title("Gradient Norm Imbalance (max/min ratio)", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Ratio")
        ax.legend(fontsize=8)

    # Panel 3 — Cosine Similarity
    if has_cosine:
        ax = axes[2]
        cosines = history["grad_cosine_mean"]
        steps = list(range(len(cosines)))
        cosines_clean = [c if c == c else 0.0 for c in cosines]  # replace NaN
        ax.plot(steps, cosines_clean, color="#CE93D8", linewidth=1.5)
        _smooth_overlay(ax, cosines_clean, steps, color="#8E24AA")
        ax.axhline(y=0, color="#888", linestyle="--", linewidth=0.8)
        ax.fill_between(steps, cosines_clean, 0,
                        where=[c > 0 for c in cosines_clean],
                        alpha=0.15, color="#CE93D8", label="Positive alignment")
        ax.fill_between(steps, cosines_clean, 0,
                        where=[c < 0 for c in cosines_clean],
                        alpha=0.15, color="#EF5350", label="Negative alignment")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Gradient Cosine Similarity (step-to-step)", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cosine Similarity")
        ax.legend(fontsize=8)

    # Panel 4 — CKA
    if has_cka:
        ax = axes[-1]
        cmap = plt.cm.get_cmap("plasma", len(cka_keys))
        for i, key in enumerate(cka_keys):
            vals = history[key]
            label = key.replace("cka_", "").replace("_vs_", " ↔ ")
            ax.plot(range(len(vals)), vals, color=cmap(i), linewidth=1.5, label=label)
        ax.set_ylim(0, 1.05)
        ax.set_title("Linear CKA — Layer Similarity", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("CKA Score")
        ax.legend(fontsize=7, loc="lower right")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GradScope] Dashboard saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_grad_norms_per_layer(
    history: Dict[str, List],
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot per-layer gradient norm evolution for the top-k most variable layers.
    Useful for identifying which layers are responsible for imbalance.
    """
    _apply_style("dark")

    # Extract per-layer data from history
    layer_keys = [k for k in history if k not in
                  ("grad_norm_global", "grad_norm_imbalance", "grad_cosine_mean")
                  and not k.startswith("cka_")]

    if not layer_keys:
        print("[GradScope] No per-layer norm history found. "
              "Access tracker.history['grad_norms'] for raw step data.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab20", min(top_k, len(layer_keys)))

    for i, key in enumerate(layer_keys[:top_k]):
        vals = history[key]
        ax.plot(range(len(vals)), vals, color=cmap(i), linewidth=1.2,
                label=_shorten_layer_name(key), alpha=0.85)

    ax.set_yscale("log")
    ax.set_title("Per-Layer Gradient Norms", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("L2 Norm (log scale)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _smooth_overlay(ax, values, steps, color, window=20):
    if len(values) < window:
        return
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    ax.plot(steps[window - 1:], smoothed, color=color, linewidth=2.5,
            label=f"Smoothed (w={window})", alpha=0.9)


def _add_threshold_line(ax, y, label, color):
    ax.axhline(y=y, color=color, linestyle="--", linewidth=0.9, alpha=0.7, label=label)


def _apply_style(style: str):
    if style == "dark":
        plt.style.use("dark_background")
        plt.rcParams.update({
            "axes.facecolor": "#1a1a2e",
            "figure.facecolor": "#0f0f1a",
            "axes.edgecolor": "#444",
            "grid.color": "#2a2a3e",
            "axes.grid": True,
        })
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def _shorten_layer_name(name: str, max_len: int = 30) -> str:
    if len(name) <= max_len:
        return name
    parts = name.split(".")
    return "..." + ".".join(parts[-2:]) if len(parts) >= 2 else name[-max_len:]
