import torch
from torch.func import grad, jacrev, vmap


def f_scalar(p):            # p: (2,) → skaler
    return (p[0] + p[1]) * p[1]


def g_vec(p):               # p: (2,) → (2,)
    return torch.stack([p[0] * p[1], p[0] + p[1] ** 2])


def batch_grad(P):
    """P: (N, 2) → (N, 2): her noktada ∇f. Döngü yok."""
    ...


def batch_jacobian(P):
    """P: (N, 2) → (N, 2, 2). Döngü yok."""
    ...


P = torch.tensor([[2.0, 3.0], [0.0, 1.0]])
print(batch_grad(P))        # [[3, 8], [1, 2]]
print(batch_jacobian(P))
