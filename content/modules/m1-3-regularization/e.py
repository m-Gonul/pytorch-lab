import torch


def ridge_path_torch(X, y, lams):
    """→ (W: (L, d), b: (L,)). Tek SVD, döngü yok."""
    ...


def cv_ridge(X, y, lams, k=5):
    """→ (mse: (L,), best_lam). Fold döngüsü serbest, λ döngüsü yok. i. fold: idx[i::k]."""
    ...


torch.manual_seed(0)
X = torch.randn(200, 30, dtype=torch.float64)
y = X[:, :5] @ torch.ones(5, dtype=torch.float64) + torch.randn(200, dtype=torch.float64)
lams = torch.logspace(-3, 3, 50, dtype=torch.float64)
mse, best = cv_ridge(X, y, lams)
print(best)
