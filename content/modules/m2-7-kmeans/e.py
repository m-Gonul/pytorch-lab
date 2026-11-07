import torch
import torch.nn.functional as F


def kmeans_restarts(X, C0s, iters):
    """C0s: (R, k, d) → (best_C: (k, d), inertias: (R,)). Yalnız iterasyon döngüsü."""
    ...


g = torch.Generator().manual_seed(0)
X = torch.cat([torch.randn(50, 2, generator=g) + torch.tensor(c) for c in ([0.0, 0], [6, 0], [0, 6], [6, 6])])
C0s = X[torch.randint(0, 200, (32, 4), generator=g)]
best, J = kmeans_restarts(X, C0s, 20)
print(J.min(), J.max())
