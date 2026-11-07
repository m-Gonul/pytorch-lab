import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        q, k, v = (t.view(B, T, self.h, -1).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, V=64, T=256, d=64, L=4, h=4):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln_f = nn.LayerNorm(d)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ self.wte.weight.T


# ---- senin kodun ----

def forward_step(model, idx_new, cache, pos):
    """→ (logits (B, t, V), yeni cache). Tek döngü: bloklar."""
    ...


@torch.no_grad()
def generate_kv(model, idx, n):
    """Greedy; prefill + adım başına tek token. Tek döngü: adımlar."""
    ...


torch.manual_seed(0)
m = TinyGPT().eval()
print(generate_kv(m, torch.tensor([[1, 2, 3]]), 10))
