import torch
from torch.func import vjp, jvp


def jacobian_rev(f, x):
    """→ (m, n), vjp + vmap. Döngü yok."""
    ...


def jacobian_fwd(f, x):
    """→ (m, n), jvp + vmap. Döngü yok."""
    ...


A = torch.randn(3, 4)
f = lambda x: torch.tanh(A @ x)
x = torch.randn(4)
print(jacobian_rev(f, x).shape, jacobian_fwd(f, x).shape)   # (3, 4) (3, 4)
