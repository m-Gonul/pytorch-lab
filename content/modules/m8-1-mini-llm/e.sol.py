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
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    ev = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    return [e.key for e in ev[:k]]


class FusedAttention(nn.Module):
    @classmethod
    def from_loop(cls, loop):
        self = cls()
        self.h, self.dk = loop.h, loop.dk
        d = loop.h * loop.dk
        self.qkv = nn.Linear(d, 3 * d)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat([m.weight for group in (loop.q, loop.k, loop.v) for m in group], 0))
            self.qkv.bias.copy_(torch.cat([m.bias for group in (loop.q, loop.k, loop.v) for m in group], 0))
        self.proj = loop.proj
        return self

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, d))


torch.manual_seed(0)
loop = LoopAttention()
x = torch.randn(16, 64, 128)
print(top_ops(lambda: loop(x)))
fused = FusedAttention.from_loop(loop)
print((fused(x) - loop(x)).abs().max())
