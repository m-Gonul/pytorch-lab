import math
import torch


def em_restarts_diag(X, mus0, iters, reg=1e-6):
    """mus0: (R, K, d) → (lls: (R,), best: (pis, mus, vars)). Yalnız iterasyon döngüsü."""
    ...


g = torch.Generator().manual_seed(0)
X = torch.cat([torch.randn(100, 2, generator=g) + torch.tensor(c) for c in ([0.0, 0], [5, 0], [0, 5])])
mus0 = X[torch.randint(0, 300, (16, 3), generator=g)]
lls, best = em_restarts_diag(X, mus0, 30)
print(lls)
