import torch
import torch.nn.functional as F
from torch import nn


class Chain:
    """B paralel zincir ortamı. Sağ uç +1 ve bitiş, sol uç 0 ve bitiş."""
    def __init__(self, B=1, L=8, max_steps=30, start=3):
        self.B, self.L, self.T, self.start = B, L, max_steps, start

    def reset(self):
        self.s = torch.full((self.B,), self.start)
        self.t = 0
        self.alive = torch.ones(self.B, dtype=torch.bool)
        return self.obs()

    def obs(self):
        return F.one_hot(self.s, self.L).float()

    def step(self, a):
        self.s = torch.where(self.alive, (self.s + 2 * a - 1).clamp(0, self.L - 1), self.s)
        self.t += 1
        r = (self.s == self.L - 1).float() * self.alive
        done = (self.s == 0) | (self.s == self.L - 1) | (self.t >= self.T)
        self.alive = self.alive & ~done
        return self.obs(), r, done


def make_policy(L=8, hidden=32):
    return nn.Sequential(nn.Linear(L, hidden), nn.Tanh(), nn.Linear(hidden, 2))


# ---- senin kodun ----

def discounted_returns_vec(r, gamma):
    """(B, T) → (B, T). Döngü yok."""
    T = r.shape[1]
    i = torch.arange(T)
    expo = (i[:, None] - i[None, :]).to(r.dtype)
    Gam = torch.where(expo >= 0, gamma ** expo.clamp_min(0), torch.zeros_like(expo))
    return r @ Gam


def rollout(policy, env, T):
    """→ (logp, r, mask), hepsi (B, T). Tek döngü: zaman."""
    obs = env.reset()
    logps, rs, masks = [], [], []
    for _ in range(T):
        masks.append(env.alive.clone())
        d = torch.distributions.Categorical(logits=policy(obs))
        a = d.sample()
        logps.append(d.log_prob(a))
        obs, r, _ = env.step(a)
        rs.append(r)
    return torch.stack(logps, 1), torch.stack(rs, 1), torch.stack(masks, 1).float()


def train_batched(policy, env, iters=60, gamma=0.95, lr=0.01):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(iters):
        logp, r, mask = rollout(policy, env, env.T)
        G = discounted_returns_vec(r, gamma)
        b = (G * mask).sum(0) / mask.sum(0).clamp_min(1)
        loss = -(logp * (G - b) * mask).sum() / env.B
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


torch.manual_seed(0)
pi = make_policy()
train_batched(pi, Chain(B=64))
print(pi(Chain().reset()).softmax(-1))
