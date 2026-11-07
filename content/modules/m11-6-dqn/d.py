import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def make_net():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


def evaluate(q, n=20, max_len=500, seed=123):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, 4, generator=g) - 0.5) * 0.1
    alive = torch.ones(n, dtype=torch.bool)
    lens = torch.zeros(n)
    for _ in range(max_len):
        with torch.no_grad():
            x, d = cartpole_step(x, q(x).argmax(1))
        lens += alive.float()
        alive &= ~d
    return lens


class Replay:
    def __init__(self, cap, dim):
        ...

    def push(self, s, a, r, s2, d):
        ...

    def sample(self, b, g):
        ...

    def __len__(self):
        ...


def td_target(q_target, q_online, r, s2, d, gamma, double):
    ...


def soft_update(target, online, tau):
    ...


def train_dqn(steps=30000, seed=0, double=True):
    ...


q, lens = train_dqn(3000)
print(len(lens), evaluate(q).mean())
