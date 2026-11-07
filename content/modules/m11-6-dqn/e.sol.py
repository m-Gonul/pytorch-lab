import math
import torch

G_, MC, MP, L, FM, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02


def cartpole_step(x, a):
    pos, vel, th, om = x.unbind(-1)
    f = torch.where(a == 1, FM, -FM).to(x.dtype)
    ct, st = torch.cos(th), torch.sin(th)
    tm, pml = MC + MP, MP * L
    temp = (f + pml * om ** 2 * st) / tm
    thacc = (G_ * st - ct * temp) / (L * (4.0 / 3.0 - MP * ct ** 2 / tm))
    xacc = temp - pml * thacc * ct / tm
    x2 = torch.stack([pos + TAU * vel, vel + TAU * xacc, th + TAU * om, om + TAU * thacc], -1)
    done = (x2[..., 0].abs() > 2.4) | (x2[..., 2].abs() > 12 * 2 * math.pi / 360)
    return x2, done


def rollout_lengths(W, x0, max_len=500):
    x = x0.clone()
    alive = torch.ones(x.shape[:2], dtype=torch.bool)
    lens = torch.zeros(x.shape[:2], dtype=x.dtype)
    for _ in range(max_len):
        a = ((x * W[:, None, :]).sum(-1) > 0).long()
        x, d = cartpole_step(x, a)
        lens += alive.to(x.dtype)
        alive &= ~d
        if not alive.any():
            break
    return lens.mean(1)


def random_search(n, n_envs=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.rand(n, 4, generator=g, dtype=torch.float64) * 2 - 1
    x0 = (torch.rand(n, n_envs, 4, generator=g, dtype=torch.float64) - 0.5) * 0.1
    return W, rollout_lengths(W, x0)


W, Lr = random_search(500)
print(Lr.max())
