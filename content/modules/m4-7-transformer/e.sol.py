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
    B, t = idx_new.shape
    x = model.wte(idx_new) + model.wpe(torch.arange(pos, pos + t))
    d = x.shape[-1]
    new_cache = []
    for blk, c in zip(model.blocks, cache):
        q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
        q = q.view(B, t, blk.h, -1).transpose(1, 2)
        k = k.view(B, t, blk.h, -1).transpose(1, 2)
        v = v.view(B, t, blk.h, -1).transpose(1, 2)
        if c is not None:
            k = torch.cat([c[0], k], dim=2)
            v = torch.cat([c[1], v], dim=2)
        S = k.shape[2]
        mask = torch.arange(S)[None, :] <= (pos + torch.arange(t))[:, None]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + blk.proj(o.transpose(1, 2).reshape(B, t, d))
        x = x + blk.mlp(blk.ln2(x))
        new_cache.append((k, v))
    return model.ln_f(x) @ model.wte.weight.T, new_cache


@torch.no_grad()
def generate_kv(model, idx, n):
    """Greedy; prefill + adım başına tek token. Tek döngü: adımlar."""
    logits, cache = forward_step(model, idx, [None] * len(model.blocks), 0)
    for i in range(n):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], 1)
        if i < n - 1:
            logits, cache = forward_step(model, nxt, cache, idx.shape[1] - 1)
    return idx


torch.manual_seed(0)
m = TinyGPT().eval()
print(generate_kv(m, torch.tensor([[1, 2, 3]]), 10))
