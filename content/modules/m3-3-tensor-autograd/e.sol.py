import torch
from torch.func import vjp, jvp


def jacobian_rev(f, x):
    """→ (m, n), vjp + vmap. Döngü yok."""
    y, vjp_fn = vjp(f, x)
    I = torch.eye(y.numel(), dtype=y.dtype).view(-1, *y.shape)
    return torch.vmap(lambda v: vjp_fn(v)[0])(I).reshape(y.numel(), x.numel())


def jacobian_fwd(f, x):
    """→ (m, n), jvp + vmap. Döngü yok."""
    I = torch.eye(x.numel(), dtype=x.dtype).view(-1, *x.shape)
    cols = torch.vmap(lambda v: jvp(f, (x,), (v,))[1])(I)     # (n, m)
    return cols.reshape(x.numel(), -1).T


A = torch.randn(3, 4)
f = lambda x: torch.tanh(A @ x)
x = torch.randn(4)
print(jacobian_rev(f, x).shape, jacobian_fwd(f, x).shape)   # (3, 4) (3, 4)
