import torch


def pca_torch(X, k):
    """→ (components (k, d), variances (k,)); eigh ile."""
    ...


def top_component_gd(X, steps=500, lr=0.05):
    """Rayleigh bölümünü autograd ile maksimize et → birim v."""
    ...


torch.manual_seed(0)
X = torch.randn(300, 3) @ torch.tensor([[3.0, 1.0, 0], [0, 1.0, 0], [0, 0, 0.2]])
print(pca_torch(X, 1)[0], top_component_gd(X))
