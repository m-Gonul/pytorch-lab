import torch


def sign_fix(V):
    idx = V.abs().argmax(-1, keepdim=True)
    return V * V.gather(-1, idx).sign()


def pca_torch(X, k):
    """→ (components (k, d), variances (k,)); eigh ile."""
    Xc = X - X.mean(0)
    w, V = torch.linalg.eigh(Xc.T @ Xc / (X.shape[0] - 1))
    return sign_fix(V.flip(1)[:, :k].T), w.flip(0)[:k]


def top_component_gd(X, steps=500, lr=0.05):
    """Rayleigh bölümünü autograd ile maksimize et → birim v."""
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / (X.shape[0] - 1)
    v = torch.ones(X.shape[1], dtype=X.dtype, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        loss = -(v @ C @ v) / (v @ v)
        opt.zero_grad()
        loss.backward()
        opt.step()
    v = v.detach() / v.detach().norm()
    return sign_fix(v[None])[0]


torch.manual_seed(0)
X = torch.randn(300, 3) @ torch.tensor([[3.0, 1.0, 0], [0, 1.0, 0], [0, 0, 0.2]])
print(pca_torch(X, 1)[0], top_component_gd(X))
