import torch


def train_ovr(X, y, K, lr=0.5, steps=300):
    """→ (W: (d, K), b: (K,)). Yalnız adım döngüsü; sınıflar üzerinde döngü yok."""
    n, d = X.shape
    Y = torch.nn.functional.one_hot(y, K).float()   # (n, K)
    W = torch.zeros(d, K)
    b = torch.zeros(K)
    for _ in range(steps):
        R = torch.sigmoid(X @ W + b) - Y              # (n, K): her sütun bir ikili problem
        W -= lr * X.T @ R / n
        b -= lr * R.mean(0)
    return W, b


def predict_ovr(X, W, b):
    return (X @ W + b).argmax(1)


torch.manual_seed(0)
centers = torch.tensor([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0]])
y = torch.arange(3).repeat_interleave(100)
X = centers[y] + torch.randn(300, 2)
W, b = train_ovr(X, y, 3)
print("doğruluk:", (predict_ovr(X, W, b) == y).float().mean().item())
