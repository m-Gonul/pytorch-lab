import torch
import torch.nn.functional as F


def kmeans_restarts(X, C0s, iters):
    """C0s: (R, k, d) → (best_C: (k, d), inertias: (R,)). Yalnız iterasyon döngüsü."""
    C = C0s.clone()
    k = C.shape[1]
    Xb = X.expand(C.shape[0], *X.shape)                    # (R, n, d) view
    for _ in range(iters):
        A = F.one_hot(torch.cdist(Xb, C).argmin(2), k).to(X.dtype)      # (R, n, k)
        cnt = A.sum(1)                                                  # (R, k)
        sums = torch.einsum("rnk,nd->rkd", A, X)
        C = torch.where(cnt[..., None] > 0, sums / cnt.clamp_min(1)[..., None], C)
    D = torch.cdist(Xb, C) ** 2
    J = D.min(2).values.sum(1)
    return C[J.argmin()], J


g = torch.Generator().manual_seed(0)
X = torch.cat([torch.randn(50, 2, generator=g) + torch.tensor(c) for c in ([0.0, 0], [6, 0], [0, 6], [6, 6])])
C0s = X[torch.randint(0, 200, (32, 4), generator=g)]
best, J = kmeans_restarts(X, C0s, 20)
print(J.min(), J.max())
