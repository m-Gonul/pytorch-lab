import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import profile, ProfilerActivity


class LoopAttention(nn.Module):
    """Yavaş başvuru: head başına ayrı Linear'lar, head'ler üzerinde döngü."""
    def __init__(self, d=128, h=8):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q = nn.ModuleList(nn.Linear(d, self.dk) for _ in range(h))
        self.k = nn.ModuleList(nn.Linear(d, self.dk) for _ in range(h))
        self.v = nn.ModuleList(nn.Linear(d, self.dk) for _ in range(h))
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool).triu(1)
        outs = []
        for i in range(self.h):
            q, k, v = self.q[i](x), self.k[i](x), self.v[i](x)
            att = (q @ k.transpose(-2, -1)) / self.dk ** 0.5
            att = att.masked_fill(mask, float("-inf")).softmax(-1)
            outs.append(att @ v)
        return self.proj(torch.cat(outs, -1))


# ---- senin kodun ----

def top_ops(fn, k=5):
    ...


class FusedAttention(nn.Module):
    @classmethod
    def from_loop(cls, loop):
        ...

    def forward(self, x):
        ...


torch.manual_seed(0)
loop = LoopAttention()
x = torch.randn(16, 64, 128)
print(top_ops(lambda: loop(x)))
fused = FusedAttention.from_loop(loop)
print((fused(x) - loop(x)).abs().max())
