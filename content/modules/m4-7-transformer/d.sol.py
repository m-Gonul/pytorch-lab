import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.h = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def attn(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, block_size, d, n_layer, n_head):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, std=0.02)
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss


torch.manual_seed(0)
m = GPT(vocab=20, block_size=16, d=32, n_layer=2, n_head=4)
logits, loss = m(torch.randint(0, 20, (2, 10)), torch.randint(0, 20, (2, 10)))
print(logits.shape, loss)   # (2, 10, 20), ≈ log 20 = 3.0
