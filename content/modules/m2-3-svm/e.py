import torch


def rbf_kernel(A, B, gamma):
    ...


def fit_kernel_svm(X, y, gamma, lam=1e-3, steps=300, lr=0.5):
    """y ∈ {−1, +1}. → (alpha: (n,), b). Yalnız adım döngüsü."""
    ...


def predict_kernel(Xtr, alpha, b, gamma, X):
    ...


g = torch.Generator().manual_seed(0)
r = torch.cat([torch.full((100,), 1.0), torch.full((100,), 3.0)]) + 0.2 * torch.randn(200, generator=g)
t = torch.rand(200, generator=g) * 6.2832
X = torch.stack([r * t.cos(), r * t.sin()], 1)
y = torch.cat([torch.ones(100), -torch.ones(100)])
alpha, b = fit_kernel_svm(X, y, gamma=0.5)
print((predict_kernel(X, alpha, b, 0.5, X) == y).float().mean())
