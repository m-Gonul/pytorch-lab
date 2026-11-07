import torch
from torch.func import grad, vmap


def loss_one(w, b, x):
    return torch.tanh(x @ w + b)


def per_sample_grads(X, w, b):
    """→ (n, d): her satır ∂tanh(xᵢ·w + b)/∂w. Döngü yok."""
    return vmap(grad(loss_one), in_dims=(None, None, 0))(w, b, X)


X = torch.randn(4, 3)
w = torch.randn(3)
print(per_sample_grads(X, w, 0.1))
