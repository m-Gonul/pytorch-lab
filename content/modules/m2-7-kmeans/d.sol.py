import torch


def kmeans_torch(X, C0, iters):
    C = C0.clone()
    k = C.shape[0]
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        sums = torch.zeros_like(C).index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=k).to(X.dtype)
        C = torch.where(cnt[:, None] > 0, sums / cnt.clamp_min(1)[:, None], C)
    D = torch.cdist(X, C) ** 2
    lab = D.argmin(1)
    return C, lab, D.gather(1, lab[:, None]).sum()


def lloyd_step_via_grad(X, C):
    """Atamaları sabitle, inertia'nın gradyanından Newton adımı."""
    C = C.clone().requires_grad_(True)
    lab = torch.cdist(X, C.detach()).argmin(1)
    J = ((X - C[lab]) ** 2).sum()
    g, = torch.autograd.grad(J, C)
    cnt = torch.bincount(lab, minlength=C.shape[0]).to(X.dtype)
    step = g / (2 * cnt.clamp_min(1)[:, None])
    return (C - step).detach()


X = torch.tensor([[0.0, 0], [1, 0], [4, 4], [5, 4]])
print(kmeans_torch(X, X[:2].clone(), 2))
print(lloyd_step_via_grad(X, torch.tensor([[0.0, 0], [1, 0]])))   # [[0, 0], [3.33, 2.67]]
