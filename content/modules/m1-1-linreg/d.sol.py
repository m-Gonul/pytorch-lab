import torch
from torch import nn


def one_step_grads():
    """x=[1,2,3], y=[2,4,5], w=b=0 için (dL/dw, dL/db) — autograd ile."""
    x = torch.tensor([1., 2., 3.])
    y = torch.tensor([2., 4., 5.])
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    loss = ((w * x + b - y) ** 2).mean()
    loss.backward()
    return w.grad, b.grad


def fit_torch(X, y, lr=0.1, steps=500):
    """X: (n, d), y: (n,) → eğitilmiş nn.Linear(d, 1)."""
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    y = y.reshape(-1, 1)                 # (n,) → (n, 1): broadcasting tuzağından kaçın
    for _ in range(steps):
        loss = loss_fn(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


print(one_step_grads())   # (tensor(-16.6667), tensor(-7.3333))
