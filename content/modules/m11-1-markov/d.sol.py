import torch


def simulate(T, s0, steps, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(n, steps + 1, dtype=torch.long)
    out[:, 0] = s0
    for t in range(steps):
        out[:, t + 1] = torch.multinomial(T[:, out[:, t]].T, 1, generator=g)[:, 0]
    return out


def count_mle(traj, M):
    a, b = traj[:, :-1].reshape(-1), traj[:, 1:].reshape(-1)
    N = torch.bincount(b * M + a, minlength=M * M).view(M, M).double()
    return N / N.sum(0, keepdim=True).clamp_min(1)


def grad_mle(traj, M, steps=500, lr=0.5):
    a, b = traj[:, :-1].reshape(-1), traj[:, 1:].reshape(-1)
    th = torch.zeros(M, M, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        ll = torch.log_softmax(th, 0)[b, a].mean()
        opt.zero_grad()
        (-ll).backward()
        opt.step()
    return torch.softmax(th.detach(), 0)


T = torch.tensor([[0.9, 0.5], [0.1, 0.5]], dtype=torch.float64)
tr = simulate(T, 0, 50, 100)
print(count_mle(tr, 2))
