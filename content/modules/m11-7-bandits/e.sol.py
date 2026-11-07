import math
import torch


def vec_bandit(probs, T, n_runs, method, eps=0.1, seed=0):
    torch.manual_seed(seed)
    p = torch.tensor(probs, dtype=torch.float64)
    K, best = p.numel(), p.max()
    N = torch.zeros(n_runs, K, dtype=torch.float64)
    S = torch.zeros_like(N)
    ar = torch.arange(n_runs)
    reg = torch.empty(T, dtype=torch.float64)
    tot = torch.zeros((), dtype=torch.float64)
    for t in range(1, T + 1):
        if method == "ucb":
            a = torch.full((n_runs,), t - 1) if t <= K else (S / N + torch.sqrt(2 * math.log(t) / N)).argmax(1)
        elif method == "ts":
            a = torch.distributions.Beta(1 + S, 1 + N - S).sample().argmax(1)
        else:
            est = torch.where(N > 0, S / N.clamp(min=1), 0.0) + 1e-9 * torch.rand(n_runs, K, dtype=torch.float64)
            a = torch.where(torch.rand(n_runs) < eps, torch.randint(0, K, (n_runs,)), est.argmax(1))
        r = (torch.rand(n_runs, dtype=torch.float64) < p[a]).double()
        N[ar, a] += 1
        S[ar, a] += r
        tot = tot + (best - p[a]).mean()
        reg[t - 1] = tot
    return reg


def lai_robbins(probs):
    b = max(probs)
    kl = lambda x, y: x * math.log(x / y) + (1 - x) * math.log((1 - x) / (1 - y))
    return sum((b - q) / kl(q, b) for q in probs if q < b)


print(vec_bandit([0.9, 0.8, 0.5], 500, 1000, "ucb")[-1])
