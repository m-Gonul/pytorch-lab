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


# ---- senin kodun ----

class Policy(nn.Module):
    def __init__(self, L=8, hidden=32):
        super().__init__()
        ...

    def forward(self, obs):
        ...


def discounted_returns(rewards, gamma):
    ...


def run_episode(policy, env):
    ...


def train_reinforce(policy, env, episodes=300, gamma=0.95, lr=0.01):
    ...


torch.manual_seed(0)
pi = Policy()
train_reinforce(pi, Chain())
print(pi(Chain().reset()).softmax(-1))   # sağa gitme olasılığı yüksek olmalı
