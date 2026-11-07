import torch

A_, B_, Q_, R_, GAM = 1.1, 0.5, 1.0, 0.5, 0.95


def riccati(iters=2000):
    P = 0.0
    for _ in range(iters):
        P = Q_ + GAM * A_ * A_ * P - (GAM * A_ * B_ * P) ** 2 / (R_ + GAM * B_ * B_ * P)
    return P, -GAM * A_ * B_ * P / (R_ + GAM * B_ * B_ * P)


def collect(k, n_envs, T, sigma, g):
    s = torch.randn(n_envs, generator=g, dtype=torch.float64)
    S, U, Rw, S2 = [], [], [], []
    for _ in range(T):
        u = k * s + sigma * torch.randn(n_envs, generator=g, dtype=torch.float64)
        s2 = A_ * s + B_ * u + 0.1 * torch.randn(n_envs, generator=g, dtype=torch.float64)
        S.append(s)
        U.append(u)
        Rw.append(-(Q_ * s ** 2 + R_ * u ** 2))
        S2.append(s2)
        s = s2.clamp(-5, 5)
    return torch.cat(S), torch.cat(U), torch.cat(Rw), torch.cat(S2)


def _phi(s, u):
    return torch.stack([s * s, s * u, u * u, torch.ones_like(s)], -1)


def lstdq(S, U, R, S2, k):
    phi = _phi(S, U)
    phi2 = _phi(S2, k * S2)
    return torch.linalg.solve(phi.T @ (phi - GAM * phi2), phi.T @ R)


def dpg(n_iter=40, k0=-0.5, alpha=None, n_envs=256, T=20, sigma=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    k, hist = k0, [k0]
    for _ in range(n_iter):
        S, U, Rw, S2 = collect(k, n_envs, T, sigma, g)
        w = lstdq(S, U, Rw, S2, k)
        if alpha is None:
            k = float(-w[1] / (2 * w[2]))
        else:
            k = float(k + alpha * (S * S).mean() * (w[1] + 2 * w[2] * k))
        hist.append(k)
    return k, hist


print(riccati()[1], dpg()[0])
