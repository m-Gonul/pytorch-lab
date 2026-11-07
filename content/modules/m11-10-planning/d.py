import math
import torch
import torch.nn as nn

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


def random_data(n_eps, g):
    ...


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x, a):
        ...


def fit(model, X, A, X2, epochs=100, seed=0):
    ...


def mpc_action(model, x, g, n_seq=500, H=20):
    ...


def run_mpc(model, g, max_len=200):
    ...


g = torch.Generator().manual_seed(0)
X, A, X2 = random_data(50, g)
m = Model(); fit(m, X, A, X2, 20)
print(run_mpc(m, g, 50))
