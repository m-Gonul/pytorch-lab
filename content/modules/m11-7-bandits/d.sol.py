import torch


def normal_posterior(sum_r, n, mu0, var0, noise_var):
    prec = 1.0 / var0 + n / noise_var
    return (mu0 / var0 + sum_r / noise_var) / prec, 1.0 / prec


def thompson(probs, T, n_runs, k=1, seed=0):
    torch.manual_seed(seed)
    p = torch.tensor(probs, dtype=torch.float64)
    K, best = p.numel(), p.max()
    S = torch.zeros(n_runs, K, dtype=torch.float64)
    F = torch.zeros_like(S)
    ar = torch.arange(n_runs)
    reg = torch.empty(T, dtype=torch.float64)
    tot = torch.zeros((), dtype=torch.float64)
    for t in range(T):
        a = torch.distributions.Beta(1 + S, 1 + F).sample((k,)).mean(0).argmax(1)
        r = (torch.rand(n_runs, dtype=torch.float64) < p[a]).double()
        S[ar, a] += r
        F[ar, a] += 1 - r
        tot = tot + (best - p[a]).mean()
        reg[t] = tot
    return reg


def gaussian_thompson(means, T, n_runs, noise_sd=1.0, seed=0):
    torch.manual_seed(seed)
    m = torch.tensor(means, dtype=torch.float64)
    K, best = m.numel(), m.max()
    Sr = torch.zeros(n_runs, K, dtype=torch.float64)
    N = torch.zeros_like(Sr)
    ar = torch.arange(n_runs)
    reg = torch.empty(T, dtype=torch.float64)
    tot = torch.zeros((), dtype=torch.float64)
    for t in range(T):
        mu, var = normal_posterior(Sr, N, 0.0, 1.0, noise_sd ** 2)
        a = (mu + var.sqrt() * torch.randn_like(mu)).argmax(1)
        r = m[a] + noise_sd * torch.randn(n_runs, dtype=torch.float64)
        Sr[ar, a] += r
        N[ar, a] += 1
        tot = tot + (best - m[a]).mean()
        reg[t] = tot
    return reg


print(thompson([0.9, 0.8, 0.5], 200, 100)[-1])
