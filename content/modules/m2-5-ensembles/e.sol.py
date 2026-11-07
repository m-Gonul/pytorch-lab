import torch


def bagged_stumps(X, y, B, generator):
    """→ (j, t, sol, sağ), her biri (B,). Döngü yok."""
    n, d = X.shape
    idx = torch.randint(n, (B, n), generator=generator)
    W = torch.zeros(B, n, dtype=X.dtype).scatter_add_(1, idx, torch.ones(B, n, dtype=X.dtype))
    Xs, order = torch.sort(X, dim=0, stable=True)                     # (n, d)
    Ws = W[:, order]                                              # (B, n, d)
    Ss = (W * y)[:, order]
    WL, SL = Ws.cumsum(1)[:, :-1], Ss.cumsum(1)[:, :-1]          # (B, n-1, d)
    WR, SR = Ws.sum(1, keepdim=True) - WL, Ss.sum(1, keepdim=True) - SL
    valid = (WL > 0) & (WR > 0) & (Xs[:-1] != Xs[1:])
    gain = torch.where(valid, SL ** 2 / WL.clamp_min(1e-12) + SR ** 2 / WR.clamp_min(1e-12), -torch.inf)
    k = gain.transpose(1, 2).reshape(B, -1).argmax(1)             # (B,)
    j, i = k // (n - 1), k % (n - 1)
    b = torch.arange(B)
    t = (Xs[i, j] + Xs[i + 1, j]) / 2
    return j, t, SL[b, i, j] / WL[b, i, j], SR[b, i, j] / WR[b, i, j]


def bagged_predict(model, X):
    """→ (n,) B stump'ın ortalaması. Döngü yok."""
    j, t, l, r = model
    xj = X[:, j]                                                   # (n, B)
    return torch.where(xj <= t, l, r).mean(1)


g = torch.Generator().manual_seed(0)
X = torch.rand(200, 3, generator=g)
y = torch.sin(6 * X[:, 0]) + 0.1 * torch.randn(200, generator=g)
model = bagged_stumps(X, y, 100, g)
print(((bagged_predict(model, X) - y) ** 2).mean())
