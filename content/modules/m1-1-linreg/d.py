import torch
from torch import nn


def one_step_grads():
    """x=[1,2,3], y=[2,4,5], w=b=0 için (dL/dw, dL/db) — autograd ile."""
    x = torch.tensor([1., 2., 3.])
    y = torch.tensor([2., 4., 5.])
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    ...


def fit_torch(X, y, lr=0.1, steps=500):
    """X: (n, d), y: (n,) → eğitilmiş nn.Linear(d, 1)."""
    model = nn.Linear(X.shape[1], 1)
    ...
    return model


print(one_step_grads())   # (tensor(-16.6667), tensor(-7.3333))
