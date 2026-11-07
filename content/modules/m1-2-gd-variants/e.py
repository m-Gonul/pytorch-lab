import torch


def sweep(X, y, lrs, betas, steps):
    """P = len(lrs) model; tam batch momentum GD. → son MSE kayıpları (P,). Yalnız adım döngüsü."""
    P, d = lrs.shape[0], X.shape[1]
    W = torch.zeros(P, d, dtype=X.dtype)
    V = torch.zeros(P, d, dtype=X.dtype)
    ...


torch.manual_seed(0)
X = torch.randn(300, 2) * torch.tensor([1.0, 8.0])
y = X @ torch.tensor([2.0, 0.3])
lr_grid, beta_grid = torch.meshgrid(torch.logspace(-4, -2, 6), torch.linspace(0, 0.95, 8), indexing="ij")
losses = sweep(X, y, lr_grid.reshape(-1), beta_grid.reshape(-1), 100)
print(losses.view(6, 8))
