import torch


def _H(P, R, gamma, V):
    return (R + gamma * torch.einsum("...ast,...t->...sa", P, V)).max(-1).values


def batched_vi(P, R, gamma, iters):
    V = torch.zeros(R.shape[:-1], dtype=R.dtype)
    diffs = []
    for _ in range(iters):
        Vn = _H(P, R, gamma, V)
        diffs.append((Vn - V).abs().amax(-1))
        V = Vn
    return V, torch.stack(diffs)


def contraction_ratio(P, R, gamma, n=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = R.shape[0]
    V = torch.randn(n, S, generator=g, dtype=R.dtype) * 10
    U = torch.randn(n, S, generator=g, dtype=R.dtype) * 10
    return (_H(P, R, gamma, V) - _H(P, R, gamma, U)).abs().amax(-1) / (V - U).abs().amax(-1)


def soft_v_star(P, R, gamma, iters):
    V = torch.zeros(R.shape[0], dtype=R.dtype)
    for _ in range(iters):
        V = _H(P, R, gamma, V)
    return V


P = torch.rand(4, 2, 5, 5, dtype=torch.float64); P /= P.sum(-1, keepdim=True)
print(batched_vi(P, torch.rand(4, 5, 2, dtype=torch.float64), 0.9, 50)[0].shape)
