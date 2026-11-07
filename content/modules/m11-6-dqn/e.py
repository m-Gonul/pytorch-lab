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
    ...


def random_search(n, n_envs=10, seed=0):
    ...


W, Lr = random_search(500)
print(Lr.max())
