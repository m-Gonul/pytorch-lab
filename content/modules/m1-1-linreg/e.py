import torch


def sweep(X, y, lrs, steps=200):
    """L = len(lrs) modeli birlikte eğit. → (W: (L, d), b: (L,), son kayıplar: (L,))
    Yalnız adım döngüsü; lr'ler üzerinde döngü yok."""
    L, d = lrs.shape[0], X.shape[1]
    W = torch.zeros(L, d, requires_grad=True)
    b = torch.zeros(L, requires_grad=True)
    ...


torch.manual_seed(0)
X = torch.randn(200, 4)
y = X @ torch.tensor([1.0, -2.0, 0.5, 0.0]) + 1.0
lrs = torch.logspace(-3, 0, 16)
W, b, losses = sweep(X, y, lrs)
print(losses)
