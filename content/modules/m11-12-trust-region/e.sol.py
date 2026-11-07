import torch


def gae_scan(r, v, d, v_last, gamma, lam):
    adv = torch.zeros_like(r)
    last = torch.zeros_like(v_last)
    for t in range(r.shape[0] - 1, -1, -1):
        nv = v_last if t == r.shape[0] - 1 else v[t + 1]
        last = r[t] + gamma * (1 - d[t]) * nv - v[t] + gamma * lam * (1 - d[t]) * last
        adv[t] = last
    return adv


def gae_matrix(r, v, d, v_last, gamma, lam):
    T, N = r.shape
    nv = torch.cat([v[1:], v_last[None]], 0)
    delta = r + gamma * (1 - d) * nv - v
    dex = torch.cat([torch.zeros(1, N, dtype=d.dtype), d.cumsum(0)[:-1]], 0)
    t = torch.arange(T)
    k = (t[None, :] - t[:, None]).to(r.dtype)
    same = dex.T[:, :, None] == dex.T[:, None, :]
    M = torch.where((k >= 0)[None] & same, (gamma * lam) ** k.clamp(min=0), torch.zeros((), dtype=r.dtype))
    return torch.einsum("ntk,kn->tn", M, delta)


def walk_batch(N, T, g, n=19):
    s = torch.full((N,), (n + 1) // 2)
    S, R, D = [s.clone()], [], []
    done = torch.zeros(N, dtype=torch.bool)
    for _ in range(T):
        s2 = torch.where(done, s, s + torch.where(torch.rand(N, generator=g) < 0.5, 1, -1))
        newly = (~done) & ((s2 == 0) | (s2 == n + 1))
        R.append(torch.where(newly, torch.where(s2 == n + 1, 1.0, -1.0), 0.0).double())
        done = done | newly
        D.append(done.double())
        s = s2
        S.append(s.clone())
    return torch.stack(S), torch.stack(R), torch.stack(D)


g = torch.Generator().manual_seed(0)
S, R, D = walk_batch(4, 50, g)
print(S.shape, R.sum(0))
