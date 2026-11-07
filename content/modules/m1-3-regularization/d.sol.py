import torch
from torch import nn


def fit_reg(X, y, lam2, lam1=0.0, steps=3000, lr=0.05):
    """MSE + lam2‖w‖² + lam1‖w‖₁ (bias hariç), cezalar kayba elle eklenir."""
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    t = y.reshape(-1, 1)
    for _ in range(steps):
        w = model.weight
        loss = ((model(X) - t) ** 2).mean() + lam2 * (w ** 2).sum() + lam1 * w.abs().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def fit_weight_decay(X, y, lam2, steps=3000, lr=0.05):
    """Aynı ridge modeli, ama ceza yerine SGD weight_decay ile."""
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.SGD([
        {"params": [model.weight], "weight_decay": 2 * lam2},   # ∇(λ‖w‖²) = 2λw
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=lr)
    t = y.reshape(-1, 1)
    for _ in range(steps):
        loss = ((model(X) - t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


torch.manual_seed(0)
X = torch.randn(100, 3)
y = X @ torch.tensor([1.0, -2.0, 0.0]) + 0.5
print(fit_reg(X, y, 0.1).weight, fit_weight_decay(X, y, 0.1).weight)
