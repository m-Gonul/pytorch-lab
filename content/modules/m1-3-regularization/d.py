import torch
from torch import nn


def fit_reg(X, y, lam2, lam1=0.0, steps=3000, lr=0.05):
    """MSE + lam2‖w‖² + lam1‖w‖₁ (bias hariç), cezalar kayba elle eklenir."""
    ...


def fit_weight_decay(X, y, lam2, steps=3000, lr=0.05):
    """Aynı ridge modeli, ama ceza yerine SGD weight_decay ile."""
    ...


torch.manual_seed(0)
X = torch.randn(100, 3)
y = X @ torch.tensor([1.0, -2.0, 0.0]) + 0.5
print(fit_reg(X, y, 0.1).weight, fit_weight_decay(X, y, 0.1).weight)
