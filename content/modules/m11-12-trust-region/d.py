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


class PolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
        self.v = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


def rollout(net, state, g, n_envs=16, T=128):
    """state = [x, t_ep, lens] (yerinde güncellenir). Döner: X, A, LP, V, R, D, v_last."""
    x, t_ep, lens = state
    X, A, LP, V, R, D = [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(T):
            lo, v = net(x)
            dist = torch.distributions.Categorical(logits=lo)
            a = dist.sample()
            x2, d = cartpole_step(x, a)
            t_ep += 1
            X.append(x); A.append(a); LP.append(dist.log_prob(a)); V.append(v); R.append(torch.ones(n_envs)); D.append(d.float())
            reset = d | (t_ep >= 500)
            if reset.any():
                lens += t_ep[reset].tolist()
                t_ep[reset] = 0
                x2 = torch.where(reset[:, None], (torch.rand(n_envs, 4, generator=g) - 0.5) * 0.1, x2)
            x = x2
        v_last = net(x)[1]
    state[0] = x
    return torch.stack(X), torch.stack(A), torch.stack(LP), torch.stack(V), torch.stack(R), torch.stack(D), v_last


def gae(r, v, d, v_last, gamma, lam):
    ...


def ppo_loss(logp_new, logp_old, adv, eps):
    ...


def train_ppo(iters=60, eps=0.2, epochs=4, mb=256, lr=3e-4, seed=0):
    ...


def fvp(pi, X, v, damping=0.1):
    ...


def cg(Avp, b, iters=10):
    ...


def trpo_step(pi, X, A, adv, old_logp, max_kl=0.01):
    ...


def train_trpo(iters=60, max_kl=0.01, seed=0):
    ...


net, lens = train_ppo(5)
print(evaluate(net).mean())
