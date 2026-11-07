import math
import torch
import torch.nn as nn

G_, MC, MP, L, FM, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02


def cartpole_step(x, a):
    pos, vel, th, om = x.unbind(1)
    f = torch.where(a == 1, FM, -FM).to(x.dtype)
    ct, st = torch.cos(th), torch.sin(th)
    tm, pml = MC + MP, MP * L
    temp = (f + pml * om ** 2 * st) / tm
    thacc = (G_ * st - ct * temp) / (L * (4.0 / 3.0 - MP * ct ** 2 / tm))
    xacc = temp - pml * thacc * ct / tm
    x2 = torch.stack([pos + TAU * vel, vel + TAU * xacc, th + TAU * om, om + TAU * thacc], 1)
    done = (x2[:, 0].abs() > 2.4) | (x2[:, 2].abs() > 12 * 2 * math.pi / 360)
    return x2, done


def evaluate(net, n=20, max_len=500, seed=123):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, 4, generator=g) - 0.5) * 0.1
    alive = torch.ones(n, dtype=torch.bool)
    lens = torch.zeros(n)
    for _ in range(max_len):
        with torch.no_grad():
            x, d = cartpole_step(x, net(x)[0].argmax(1))
        lens += alive.float()
        alive &= ~d
    return lens


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        ...


def nstep_returns(r, d, v_last, gamma):
    ...


def train_a2c(updates=1500, n_envs=32, n_steps=8, gamma=0.99, lr=2e-3, ent=0.01, seed=0):
    ...


net, lens = train_a2c(100)
print(len(lens), evaluate(net).mean())
