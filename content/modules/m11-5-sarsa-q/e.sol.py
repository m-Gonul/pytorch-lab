import torch
H, W, START, GOAL = 4, 12, 36, 47
DR = torch.tensor([-1, 0, 1, 0])
DC = torch.tensor([0, 1, 0, -1])


def step(s, a):
    r = (s // W + DR[a]).clamp(0, H - 1)
    c = (s % W + DC[a]).clamp(0, W - 1)
    cliff = (r == 3) & (c >= 1) & (c <= 10)
    s2 = torch.where(cliff, torch.full_like(r, START), r * W + c)
    rew = torch.where(cliff, -100.0, -1.0).double()
    return s2, rew, s2 == GOAL


def _choose(Qs, eps, g):
    n = Qs.shape[0]
    greedy = (Qs + 1e-9 * torch.rand(Qs.shape, generator=g, dtype=Qs.dtype)).argmax(1)
    rnd = torch.randint(0, 4, (n,), generator=g)
    return torch.where(torch.rand(n, generator=g) < eps, rnd, greedy)


def batched_control(n_envs, n_steps, alpha, eps, method="q", seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.zeros(n_envs, H * W, 4, dtype=torch.float64)
    ar = torch.arange(n_envs)
    s = torch.full((n_envs,), START)
    a = _choose(Q[ar, s], eps, g)
    rew = torch.empty(n_steps, dtype=torch.float64)
    for t in range(n_steps):
        s2, r, done = step(s, a)
        a2 = _choose(Q[ar, s2], eps, g)
        q2 = Q[ar, s2]
        if method == "sarsa":
            nxt = q2[ar, a2]
        elif method == "expected":
            p = torch.full_like(q2, eps / 4)
            p[ar, q2.argmax(1)] += 1 - eps
            nxt = (p * q2).sum(1)
        else:
            nxt = q2.max(1).values
        nxt = torch.where(done, 0.0, nxt)
        Q[ar, s, a] += alpha * (r + nxt - Q[ar, s, a])
        rew[t] = r.mean()
        s = torch.where(done, torch.full_like(s2, START), s2)
        a = torch.where(done, _choose(Q[ar, s], eps, g), a2)
    return Q, rew


def greedy_returns(Q, max_steps=100):
    n = Q.shape[0]
    ar = torch.arange(n)
    s = torch.full((n,), START)
    G = torch.zeros(n, dtype=torch.float64)
    alive = torch.ones(n, dtype=torch.bool)
    for _ in range(max_steps):
        s2, r, done = step(s, Q[ar, s].argmax(1))
        G += torch.where(alive, r, 0.0)
        alive &= ~done
        s = s2
    return G


Q, rew = batched_control(100, 2000, 0.5, 0.1)
print(greedy_returns(Q)[:10])
