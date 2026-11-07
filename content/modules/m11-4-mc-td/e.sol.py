import torch


def batched_td(n_runs, n_steps, alpha, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.full((n_runs, n + 2), 0.5, dtype=torch.float64)
    V[:, 0] = V[:, -1] = 0
    true = torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)
    s = torch.full((n_runs,), (n + 1) // 2, dtype=torch.long)
    ar = torch.arange(n_runs)
    rms = torch.empty(n_steps, dtype=torch.float64)
    for t in range(n_steps):
        s2 = s + torch.where(torch.rand(n_runs, generator=g) < 0.5, 1, -1)
        r = (s2 == n + 1).double()
        V[ar, s] += alpha * (r + V[ar, s2] - V[ar, s])
        s = torch.where((s2 == 0) | (s2 == n + 1), torch.full_like(s2, (n + 1) // 2), s2)
        rms[t] = ((V[:, 1:n + 1] - true) ** 2).mean(1).sqrt().mean()
    return V, rms


V, rms = batched_td(100, 500, 0.1)
print(V.shape, rms[-1])
