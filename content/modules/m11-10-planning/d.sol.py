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
    X, A, X2 = [], [], []
    for _ in range(n_eps):
        x = (torch.rand(1, 4, generator=g) - 0.5) * 0.1
        for _ in range(200):
            a = torch.randint(0, 2, (1,), generator=g)
            x2, d = cartpole_step(x, a)
            X.append(x)
            A.append(a)
            X2.append(x2)
            x = x2
            if d.item():
                break
    return torch.cat(X), torch.cat(A), torch.cat(X2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))
        self.register_buffer("mu", torch.zeros(5))
        self.register_buffer("sd", torch.ones(5))
        self.register_buffer("dsd", torch.ones(4))

    def forward(self, x, a):
        inp = (torch.cat([x, a.float()[..., None]], -1) - self.mu) / self.sd
        return x + self.net(inp) * self.dsd


def fit(model, X, A, X2, epochs=100, seed=0):
    torch.manual_seed(seed)
    inp = torch.cat([X, A.float()[:, None]], 1)
    model.mu.copy_(inp.mean(0))
    model.sd.copy_(inp.std(0) + 1e-6)
    model.dsd.copy_((X2 - X).std(0) + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            loss = (((model(X[idx], A[idx]) - X2[idx]) / model.dsd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.detach())


def mpc_action(model, x, g, n_seq=500, H=20):
    seq = torch.randint(0, 2, (n_seq, H), generator=g)
    xs = x.expand(n_seq, 4)
    cost = torch.zeros(n_seq)
    alive = torch.ones(n_seq)
    with torch.no_grad():
        for h in range(H):
            xs = model(xs, seq[:, h])
            bad = (xs[:, 0].abs() > 2.4) | (xs[:, 2].abs() > 12 * 2 * math.pi / 360)
            alive = alive * (~bad).float()
            cost += (1 - alive) * 10 + xs[:, 2] ** 2 + 0.01 * xs[:, 0] ** 2
    return seq[cost.argmin(), 0:1]


def run_mpc(model, g, max_len=200):
    x = (torch.rand(1, 4, generator=g) - 0.5) * 0.1
    for t in range(max_len):
        x, d = cartpole_step(x, mpc_action(model, x, g))
        if d.item():
            return t + 1
    return max_len


g = torch.Generator().manual_seed(0)
X, A, X2 = random_data(50, g)
m = Model(); fit(m, X, A, X2, 20)
print(run_mpc(m, g, 50))
