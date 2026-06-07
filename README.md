# GradScope 🔬

**Training dynamics diagnostics for PyTorch. One line of code.**

[![PyPI version](https://badge.fury.io/py/gradscope.svg)](https://badge.fury.io/py/gradscope)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

Most training bugs are invisible. Your loss goes down. Your accuracy looks fine. But your model is stuck in a bad basin of attraction — locked there by gradient norm imbalance, directional instability, or a curriculum that's destroying generalization.

GradScope makes these dynamics visible in real time.

Built from the diagnostic tools behind a NeurIPS 2026 paper on curriculum learning — showing that slow easy-to-hard curricula can *harm* generalization by locking low-capacity models into harmful basins.

---

## Install

```bash
pip install gradscope
```

---

## Quickstart

```python
from gradscope import GradTracker

model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
tracker = GradTracker(model).attach()

for epoch in range(epochs):
    for x, y in dataloader:
        loss = criterion(model(x), y)
        loss.backward()

        tracker.step()          # ← this is all you add

        optimizer.step()
        optimizer.zero_grad()

tracker.plot()                  # dashboard
print(tracker.summary())        # full report + diagnosis
```

That's it. No configuration required.

---

## What It Tracks

### 1. Gradient Norm (Global + Per-Layer)
Detects vanishing and exploding gradients automatically.
Warns you at runtime before your training silently fails.

```
[GradScope] ⚠️  Vanishing gradients detected at step 142. Global norm = 3.2e-07
[GradScope] 💥 Exploding gradients at step 89. Global norm = 1.4e+04
```

### 2. Gradient Norm Imbalance
The ratio of the largest to smallest per-layer gradient norm.
A high imbalance (>100x) means some layers are updating much faster than others — a key signal that your curriculum or learning rate is misaligned with your model's capacity.

### 3. Gradient Cosine Similarity (step-to-step)
Measures how consistent gradient *directions* are between consecutive steps.
Low cosine similarity = your optimizer is thrashing. High = stable but possibly over-smooth.

### 4. Linear CKA (optional)
Centered Kernel Alignment between layer activations. Tracks how representations drift across training — useful for diagnosing representational collapse or curriculum-induced capacity lock-in.

---

## Dashboard

```python
tracker.plot()
```

Renders a 4-panel training dynamics dashboard:

```
┌─────────────────────────┬──────────────────────────────┐
│  Global Gradient Norm   │  Gradient Norm Imbalance      │
│  (log scale, with       │  (per-step bar chart,         │
│   vanishing/exploding   │   red = danger zone)          │
│   thresholds)           │                               │
├─────────────────────────┼──────────────────────────────┤
│  Cosine Similarity      │  Linear CKA                   │
│  (step-to-step          │  (layer-pair similarity       │
│   directional           │   across training)            │
│   stability)            │                               │
└─────────────────────────┴──────────────────────────────┘
```

Save to file:
```python
tracker.plot(save_path="training_report.png")
```

---

## Automatic Diagnosis

```python
report = tracker.summary()
print(report["diagnosis"])

# Example output:
# {
#   "gradient_flow": "HEALTHY",
#   "layer_balance": "IMBALANCED (214.3x) — curriculum may be locking capacity",
#   "gradient_direction": "UNSTABLE — high directional variance between steps"
# }
```

---

## Advanced Usage

### Enable CKA tracking
```python
tracker = GradTracker(
    model,
    track_cka=True,
    cka_layers=["encoder.layer3", "encoder.layer4", "classifier"]
).attach()
```

### Custom warning thresholds
```python
tracker = GradTracker(
    model,
    warn_vanishing_threshold=1e-6,
    warn_exploding_threshold=500.0,
    warn_imbalance_threshold=50.0,
)
```

### Per-layer norm plot
```python
from gradscope import plot_grad_norms_per_layer
plot_grad_norms_per_layer(tracker.history, top_k=10)
```

### Use metrics standalone
```python
from gradscope import compute_grad_norm, compute_linear_cka

norms = compute_grad_norm(model)
print(norms["global"])          # global norm
print(norms["encoder.fc1"])     # per-layer norm

cka = compute_linear_cka(activations_A, activations_B)
```

---

## Curriculum Learning Diagnostics

GradScope was built specifically to diagnose training failures caused by curriculum ordering. If you are training with easy-to-hard or hard-to-easy curricula, high gradient norm imbalance early in training is a reliable predictor of poor generalization — the model gets locked into a narrow basin before seeing the full data distribution.

```python
# Detect curriculum-induced capacity lock-in
tracker = GradTracker(
    model,
    warn_imbalance_threshold=50.0,   # tighter threshold for curriculum settings
    track_cosine=True,
    track_cka=True,
).attach()
```

---

## Roadmap

- [ ] Wandb and TensorBoard integration
- [ ] Per-layer cosine similarity tracking
- [ ] Gradient noise scale tracking (OpenAI's critical batch size)
- [ ] Automatic PDF training report
- [ ] HuggingFace Trainer callback

---

## Citation

If GradScope helps your research, please cite:

```bibtex
@inproceedings{mahdi2026curriculum,
  title     = {Slow Curricula Harm Generalization: Gradient Dynamics Evidence},
  author    = {[Your Name]},
  booktitle = {NeurIPS},
  year      = {2026}
}
```

---

## License

MIT. Use it, fork it, build on it.

---

*Built from research. Designed for practitioners.*
