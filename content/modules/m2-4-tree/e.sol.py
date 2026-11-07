import torch
import torch.nn.functional as F


def best_split_torch(X, y, C, w=None):
    """Örnek ağırlıklı Gini. → (j, t, skor) ya da None. Döngü yok."""
    n, d = X.shape
    w = torch.ones(n, dtype=X.dtype) if w is None else w.to(X.dtype)
    Xs, order = torch.sort(X, dim=0, stable=True)                     # (n, d)
    Y = F.one_hot(y[order], C).to(X.dtype) * w[order][..., None]     # (n, d, C)
    left = Y.cumsum(0)[:-1]
    right = Y.sum(0) - left
    wl, wr = left.sum(2), right.sum(2)                               # (n-1, d)
    W = w.sum()
    gl = 1 - ((left / wl[..., None].clamp_min(1e-12)) ** 2).sum(2)
    gr = 1 - ((right / wr[..., None].clamp_min(1e-12)) ** 2).sum(2)
    score = wl / W * gl + wr / W * gr
    score = torch.where(Xs[:-1] == Xs[1:], torch.inf, score)
    if not torch.isfinite(score).any():
        return None
    s = score.T.reshape(-1)
    k = int(torch.argmin(s))
    j, i = divmod(k, n - 1)
    return j, float((Xs[i, j] + Xs[i + 1, j]) / 2), float(s[k])


X = torch.tensor([[1.0], [2], [3], [4], [5], [6]])
y = torch.tensor([0, 0, 1, 1, 1, 0])
print(best_split_torch(X, y, 2))
print(best_split_torch(X, y, 2, torch.tensor([1.0, 1, 1, 1, 1, 10])))
