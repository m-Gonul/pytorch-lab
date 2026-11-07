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
        self.s = torch.zeros(cap, dim)
        self.a = torch.zeros(cap, dtype=torch.long)
        self.r = torch.zeros(cap)
        self.s2 = torch.zeros(cap, dim)
        self.d = torch.zeros(cap)
        self.cap, self.i, self.n = cap, 0, 0

    def push(self, s, a, r, s2, d):
        k = s.shape[0]
        idx = (self.i + torch.arange(k)) % self.cap
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx] = s, a, r, s2, d.float()
        self.i = (self.i + k) % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, b, g):
        idx = torch.randint(0, self.n, (b,), generator=g)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.n


def td_target(q_target, q_online, r, s2, d, gamma, double):
    with torch.no_grad():
        if double:
            v = q_target(s2).gather(1, q_online(s2).argmax(1, keepdim=True)).squeeze(1)
        else:
            v = q_target(s2).max(1).values
        return r + gamma * (1 - d) * v


def soft_update(target, online, tau):
    with torch.no_grad():
        for pt, p in zip(target.parameters(), online.parameters()):
            pt.mul_(1 - tau).add_(tau * p)


def train_dqn(steps=30000, seed=0, double=True):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    q, qt = make_net(), make_net()
    soft_update(qt, q, 1.0)
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf = Replay(50000, 4)
    x = (torch.rand(1, 4, generator=g) - 0.5) * 0.1
    t_ep, lens = 0, []
    for t in range(steps):
        eps = max(0.05, 1 - t / (0.3 * steps))
        if torch.rand(1, generator=g).item() < eps:
            a = torch.randint(0, 2, (1,), generator=g)
        else:
            with torch.no_grad():
                a = q(x).argmax(1)
        x2, d = cartpole_step(x, a)
        t_ep += 1
        buf.push(x, a, torch.ones(1), x2, d)
        if d.item() or t_ep >= 500:
            lens.append(t_ep)
            t_ep = 0
            x = (torch.rand(1, 4, generator=g) - 0.5) * 0.1
        else:
            x = x2
        if len(buf) >= 1000:
            s, a_, r, s2, dd = buf.sample(64, g)
            y = td_target(qt, q, r, s2, dd, 0.99, double)
            loss = F.smooth_l1_loss(q(s).gather(1, a_[:, None]).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if t % 500 == 0:
                soft_update(qt, q, 1.0)
    return q, lens


q, lens = train_dqn(3000)
print(len(lens), evaluate(q).mean())
