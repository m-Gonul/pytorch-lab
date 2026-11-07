import torch


def policy_actions(W, X):
    return torch.einsum("pkd,nd->pnk", W, X).argmax(-1)


def fit_reward_model(X, A, R, K, lam=1.0):
    oh = torch.nn.functional.one_hot(A, K).to(X.dtype)
    d = X.shape[1]
    G = torch.einsum("nk,ni,nj->kij", oh, X, X) + lam * torch.eye(d, dtype=X.dtype)
    b = torch.einsum("nk,ni,n->ki", oh, X, R)
    return torch.linalg.solve(G, b)


def estimates(W, X, A, R, p, Theta):
    pa = policy_actions(W, X)
    rhat = X @ Theta.T
    match = (pa == A[None]).to(X.dtype)
    w = match / p[None]
    dm = rhat.gather(1, pa.T).T
    resid = R - rhat.gather(1, A[:, None]).squeeze(1)
    return {"DM": dm.mean(1), "IPS": (w * R[None]).mean(1), "SNIPS": (w * R[None]).sum(1) / w.sum(1).clamp(min=1e-12),
            "DR": (dm + w * resid[None]).mean(1)}


g = torch.Generator().manual_seed(0)
X = torch.randn(1000, 4, generator=g, dtype=torch.float64)
W = torch.randn(10, 3, 4, generator=g, dtype=torch.float64)
print(policy_actions(W, X).shape)
