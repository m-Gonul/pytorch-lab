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
    adv = torch.zeros_like(r)
    last = torch.zeros_like(v_last)
    for t in range(r.shape[0] - 1, -1, -1):
        nv = v_last if t == r.shape[0] - 1 else v[t + 1]
        delta = r[t] + gamma * (1 - d[t]) * nv - v[t]
        last = delta + gamma * lam * (1 - d[t]) * last
        adv[t] = last
    return adv, adv + v


def ppo_loss(logp_new, logp_old, adv, eps):
    ratio = torch.exp(logp_new - logp_old)
    return -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()


def _batch(net, state, g):
    X, A, LP, V, R, D, v_last = rollout(net, state, g)
    adv, ret = gae(R, V, D, v_last, 0.99, 0.95)
    adv = adv.reshape(-1)
    return X.reshape(-1, 4), A.reshape(-1), LP.reshape(-1), (adv - adv.mean()) / (adv.std() + 1e-8), ret.reshape(-1)


def train_ppo(iters=60, eps=0.2, epochs=4, mb=256, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = PolicyValue()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lens = []
    state = [(torch.rand(16, 4, generator=g) - 0.5) * 0.1, torch.zeros(16), lens]
    for _ in range(iters):
        X, A, LP, adv, ret = _batch(net, state, g)
        for _ in range(epochs):
            perm = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], mb):
                j = perm[i:i + mb]
                lo, v = net(X[j])
                lp = torch.log_softmax(lo, 1).gather(1, A[j, None]).squeeze(1)
                loss = ppo_loss(lp, LP[j], adv[j], eps) + 0.5 * (v - ret[j]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
    return net, lens


def _flat(xs):
    return torch.cat([x.reshape(-1) for x in xs])


def fvp(pi, X, v, damping=0.1):
    lo = pi(X)
    p_old = torch.softmax(lo, 1).detach()
    kl = (p_old * (torch.log(p_old) - torch.log_softmax(lo, 1))).sum(1).mean()
    g = _flat(torch.autograd.grad(kl, list(pi.parameters()), create_graph=True))
    return _flat(torch.autograd.grad(g @ v, list(pi.parameters()))) + damping * v


def cg(Avp, b, iters=10):
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr2 = r @ r
        if rr2 < 1e-10:
            break
        p = r + rr2 / rr * p
        rr = rr2
    return x


def _surr(pi, X, A, adv, old_logp):
    lp = torch.log_softmax(pi(X), 1).gather(1, A[:, None]).squeeze(1)
    return (torch.exp(lp - old_logp) * adv).mean()


def trpo_step(pi, X, A, adv, old_logp, max_kl=0.01):
    params = list(pi.parameters())
    p_old = torch.softmax(pi(X), 1).detach()
    L0 = _surr(pi, X, A, adv, old_logp)
    g = _flat(torch.autograd.grad(L0, params))
    s = cg(lambda v: fvp(pi, X, v), g)
    beta = torch.sqrt(2 * max_kl / (s @ fvp(pi, X, s)))
    old = _flat([p.data for p in params])
    L0 = float(L0.detach())

    def setp(f):
        i = 0
        for p in params:
            n = p.numel()
            p.data.copy_(f[i:i + n].view_as(p))
            i += n

    for k in range(10):
        setp(old + 0.5 ** k * beta * s)
        with torch.no_grad():
            kl = (p_old * (torch.log(p_old) - torch.log_softmax(pi(X), 1))).sum(1).mean()
            if kl <= 1.5 * max_kl and _surr(pi, X, A, adv, old_logp) > L0:
                return True
    setp(old)
    return False


def train_trpo(iters=60, max_kl=0.01, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = PolicyValue()
    vopt = torch.optim.Adam(net.v.parameters(), lr=1e-3)
    lens = []
    state = [(torch.rand(16, 4, generator=g) - 0.5) * 0.1, torch.zeros(16), lens]
    for _ in range(iters):
        X, A, LP, adv, ret = _batch(net, state, g)
        trpo_step(net.pi, X, A, adv, LP, max_kl)
        for _ in range(4):
            perm = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], 256):
                j = perm[i:i + 256]
                l = (net.v(X[j]).squeeze(-1) - ret[j]).pow(2).mean()
                vopt.zero_grad()
                l.backward()
                vopt.step()
    return net, lens


net, lens = train_ppo(5)
print(evaluate(net).mean())
