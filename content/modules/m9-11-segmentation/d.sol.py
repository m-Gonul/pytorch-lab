import math
import torch


def _logprob(Z, mu, cov, w):
    d = Z.shape[1]
    L = torch.linalg.cholesky(cov)
    diff = (Z[None] - mu[:, None]).unsqueeze(-1)
    y = torch.linalg.solve_triangular(L[:, None], diff, upper=False).squeeze(-1)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    lp = -0.5 * (y ** 2).sum(-1) - 0.5 * logdet[:, None] - 0.5 * d * math.log(2 * math.pi) + torch.log(w)[:, None]
    return lp.T


def gmm_em(Z, mu0, iters=50, reg=1e-6):
    N, d = Z.shape
    K = len(mu0)
    I = torch.eye(d, dtype=Z.dtype)
    mu = mu0.clone()
    w = torch.full((K,), 1.0 / K, dtype=Z.dtype)
    cov = (torch.cov(Z.T) + reg * I).expand(K, d, d).clone()
    for _ in range(iters):
        lp = _logprob(Z, mu, cov, w)
        r = torch.exp(lp - torch.logsumexp(lp, 1, keepdim=True))
        Nk = r.sum(0)
        w = Nk / N
        mu = (r.T @ Z) / Nk[:, None]
        D = Z[None] - mu[:, None]
        cov = torch.einsum("kn,kni,knj->kij", r.T, D, D) / Nk[:, None, None] + reg * I
    ll = torch.logsumexp(_logprob(Z, mu, cov, w), 1).mean()
    return mu, cov, w, ll


Z = torch.randn(200, 2, dtype=torch.float64)
print(gmm_em(Z, Z[:3])[3])
