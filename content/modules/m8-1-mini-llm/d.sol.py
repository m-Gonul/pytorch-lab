import math
import torch
import torch.nn.functional as F
from torch import nn

VOCAB = "0123456789+=\n"


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        q, k, v = (t.view(B, T, self.h, -1).transpose(1, 2) for t in (q, k, v))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V=13, T=10, d=64, L=2, h=4):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.wte.weight.T


def make_dataset(n, seed):
    g = torch.Generator().manual_seed(seed)
    a, b = torch.randint(0, 100, (2, n), generator=g)
    s = a + b
    return torch.stack([a // 10, a % 10, torch.full((n,), 10), b // 10, b % 10, torch.full((n,), 11),
                        s % 10, (s // 10) % 10, s // 100, torch.full((n,), 12)], 1)


# ---- senin kodun ----

def lr_at(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def train(model, D, steps=1500, batch=128, max_lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    model.train()
    for step in range(steps):
        ix = torch.randint(0, len(D), (batch,), generator=g)
        x, y = D[ix, :-1], D[ix, 1:]
        for group in opt.param_groups:
            group["lr"] = lr_at(step, 100, steps, max_lr, max_lr / 10)
        loss = F.cross_entropy(model(x)[:, 5:8].flatten(0, 1), y[:, 5:8].flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return loss.item()


@torch.no_grad()
def solve(model, prompts):
    idx = prompts
    for _ in range(3):
        nxt = model(idx)[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], 1)
    return idx[:, -3:]


def accuracy(model, D):
    return (solve(model, D[:, :6]) == D[:, 6:9]).all(1).float().mean().item()


torch.manual_seed(0)
model = GPT()
print(train(model, make_dataset(20000, 0), steps=300))
print(accuracy(model, make_dataset(1000, 99)))
