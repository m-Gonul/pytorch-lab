import torch


def train_ovr(X, y, K, lr=0.5, steps=300):
    """→ (W: (d, K), b: (K,)). Yalnız adım döngüsü; sınıflar üzerinde döngü yok."""
    n, d = X.shape
    Y = torch.nn.functional.one_hot(y, K).float()   # (n, K)
    W = torch.zeros(d, K)
    b = torch.zeros(K)
    ...


def predict_ovr(X, W, b):
    ...


torch.manual_seed(0)
centers = torch.tensor([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0]])
y = torch.arange(3).repeat_interleave(100)
X = centers[y] + torch.randn(300, 2)
W, b = train_ovr(X, y, 3)
print("doğruluk:", (predict_ovr(X, W, b) == y).float().mean().item())
