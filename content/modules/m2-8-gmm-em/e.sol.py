import math
import torch


def em_restarts_diag(X, mus0, iters, reg=1e-6):
    """mus0: (R, K, d) → (lls: (R,), best: (pis, mus, vars)). Yalnız iterasyon döngüsü."""
    Rn, K, d = mus0.shape
    n = X.shape[0]
    mus = mus0.clone()
    pis = torch.full((Rn, K), 1.0 / K, dtype=X.dtype)
    vars_ = X.var(0, unbiased=False).expand(Rn, K, d).clone()

    def logp(pis, mus, vars_):
        diff = X[None, :, None, :] - mus[:, None]                                 # (R, n, K, d)
        return pis.log()[:, None] - 0.5 * (torch.log(2 * math.pi * vars_)[:, None] + diff ** 2 / vars_[:, None]).sum(-1)

    for _ in range(iters):
        L = logp(pis, mus, vars_)                                                  # (R, n, K)
        Rr = (L - torch.logsumexp(L, 2, keepdim=True)).exp()
        Nk = Rr.sum(1)                                                             # (R, K)
        pis = Nk / n
        mus = torch.einsum("rnk,nd->rkd", Rr, X) / Nk[..., None]
        vars_ = torch.einsum("rnk,rnkd->rkd", Rr, (X[None, :, None] - mus[:, None]) ** 2) / Nk[..., None] + reg
    lls = torch.logsumexp(logp(pis, mus, vars_), 2).sum(1)
    b = int(lls.argmax())
    return lls, (pis[b], mus[b], vars_[b])


g = torch.Generator().manual_seed(0)
X = torch.cat([torch.randn(100, 2, generator=g) + torch.tensor(c) for c in ([0.0, 0], [5, 0], [0, 5])])
mus0 = X[torch.randint(0, 300, (16, 3), generator=g)]
lls, best = em_restarts_diag(X, mus0, 30)
print(lls)
