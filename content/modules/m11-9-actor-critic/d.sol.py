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
        self.body = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, 2)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def nstep_returns(r, d, v_last, gamma):
    out = torch.empty_like(r)
    R = v_last
    for t in range(r.shape[0] - 1, -1, -1):
        R = r[t] + gamma * (1 - d[t]) * R
        out[t] = R
    return out


def train_a2c(updates=1500, n_envs=32, n_steps=8, gamma=0.99, lr=2e-3, ent=0.01, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    x = (torch.rand(n_envs, 4, generator=g) - 0.5) * 0.1
    t_ep = torch.zeros(n_envs)
    lens = []
    for _ in range(updates):
        logps, vals, ents, rs, ds = [], [], [], [], []
        for _ in range(n_steps):
            logits, v = net(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            x2, d = cartpole_step(x, a)
            t_ep += 1
            logps.append(dist.log_prob(a))
            vals.append(v)
            ents.append(dist.entropy())
            rs.append(torch.ones(n_envs))
            ds.append(d.float())
            reset = d | (t_ep >= 500)
            if reset.any():
                lens += t_ep[reset].tolist()
                t_ep[reset] = 0
                x2 = torch.where(reset[:, None], (torch.rand(n_envs, 4, generator=g) - 0.5) * 0.1, x2)
            x = x2
        with torch.no_grad():
            v_last = net(x)[1]
        ret = nstep_returns(torch.stack(rs), torch.stack(ds), v_last, gamma)
        adv = ret - torch.stack(vals)
        loss = -(torch.stack(logps) * adv.detach()).mean() + 0.5 * adv.pow(2).mean() - ent * torch.stack(ents).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
    return net, lens


net, lens = train_a2c(100)
print(len(lens), evaluate(net).mean())
