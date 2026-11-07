import torch


def ridge_path_torch(X, y, lams):
    """→ (W: (L, d), b: (L,)). Tek SVD, döngü yok."""
    xm, ym = X.mean(0), y.mean()
    U, s, Vh = torch.linalg.svd(X - xm, full_matrices=False)
    D = s / (s ** 2 + lams[:, None])                 # (L, k)
    W = (D * (U.T @ (y - ym))) @ Vh                  # (L, d)
    return W, ym - W @ xm


def cv_ridge(X, y, lams, k=5):
    """→ (mse: (L,), best_lam). Fold döngüsü serbest, λ döngüsü yok. i. fold: idx[i::k]."""
    n = X.shape[0]
    idx = torch.arange(n)
    total = torch.zeros_like(lams)
    for i in range(k):
        val = idx[i::k]
        tr = idx[torch.arange(n) % k != i]
        W, b = ridge_path_torch(X[tr], y[tr], lams)
        P = X[val] @ W.T + b                         # (n_val, L)
        total += ((P - y[val, None]) ** 2).mean(0)
    mse = total / k
    return mse, lams[mse.argmin()]


torch.manual_seed(0)
X = torch.randn(200, 30, dtype=torch.float64)
y = X[:, :5] @ torch.ones(5, dtype=torch.float64) + torch.randn(200, dtype=torch.float64)
lams = torch.logspace(-3, 3, 50, dtype=torch.float64)
mse, best = cv_ridge(X, y, lams)
print(best)
