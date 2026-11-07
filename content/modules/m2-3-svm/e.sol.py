import torch


def rbf_kernel(A, B, gamma):
    D = (A ** 2).sum(1, keepdim=True) + (B ** 2).sum(1) - 2 * A @ B.T
    return torch.exp(-gamma * D.clamp_min(0))


def fit_kernel_svm(X, y, gamma, lam=1e-3, steps=300, lr=0.5):
    """y ∈ {−1, +1}. → (alpha: (n,), b). Yalnız adım döngüsü."""
    n = X.shape[0]
    K = rbf_kernel(X, X, gamma)
    alpha = torch.zeros(n, dtype=X.dtype)
    b = torch.zeros((), dtype=X.dtype)
    for _ in range(steps):
        f = K @ alpha + b
        a = (y * f < 1).to(X.dtype)            # aktif örnekler
        g_alpha = K.T @ (-y * a) / n + 2 * lam * K @ alpha
        g_b = (-y * a).sum() / n
        alpha = alpha - lr * g_alpha
        b = b - lr * g_b
    return alpha, b


def predict_kernel(Xtr, alpha, b, gamma, X):
    return torch.sign(rbf_kernel(X, Xtr, gamma) @ alpha + b)


g = torch.Generator().manual_seed(0)
r = torch.cat([torch.full((100,), 1.0), torch.full((100,), 3.0)]) + 0.2 * torch.randn(200, generator=g)
t = torch.rand(200, generator=g) * 6.2832
X = torch.stack([r * t.cos(), r * t.sin()], 1)
y = torch.cat([torch.ones(100), -torch.ones(100)])
alpha, b = fit_kernel_svm(X, y, gamma=0.5)
print((predict_kernel(X, alpha, b, 0.5, X) == y).float().mean())
