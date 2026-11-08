import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, CLS, SEP, MASK = 0, 1, 2, 3
V = 64


def corpus(n, seed, T=16, K=5):
    g = torch.Generator().manual_seed(seed)
    topic = torch.randint(0, K, (n,), generator=g)
    content = 24 + topic[:, None] * 8 + torch.randint(0, 8, (n, T // 2), generator=g)
    noise = torch.rand(n, T // 2, generator=g) < 0.5
    content = torch.where(noise, 24 + torch.randint(0, K * 8, (n, T // 2), generator=g), content)
    func = 4 + (content * 7 + torch.arange(T // 2)) % 20
    seq = torch.stack([func, content], 2).flatten(1)
    return torch.cat([torch.full((n, 1), CLS), seq], 1), topic


def mlm_mask(x, g, V, p=0.15):
    sel = (torch.rand(x.shape, generator=g) < p) & (x > MASK)
    u = torch.rand(x.shape, generator=g)
    inp = x.clone()
    inp[sel & (u < 0.8)] = MASK
    rnd = sel & (u >= 0.8) & (u < 0.9)
    inp[rnd] = torch.randint(4, V, (int(rnd.sum()),), generator=g)
    return inp, torch.where(sel, x, torch.full_like(x, -100))


class TinyBERT(nn.Module):
    def __init__(self, V, d=64, L=2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mlm = nn.Linear(d, V)
        self.cls = nn.Linear(d, 5)

    def forward(self, x):
        return self.norm(self.enc(self.emb(x) + self.pos[:, :x.shape[1]]))


def pretrain(model, U, epochs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(U), generator=g)
        for i in range(0, len(U), 128):
            inp, lab = mlm_mask(U[perm[i:i + 128]], g, model.mlm.out_features)
            loss = F.cross_entropy(model.mlm(model(inp)).flatten(0, 1), lab.flatten(), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss)


def finetune(model, X, y, steps, lr, freeze=False):
    opt = torch.optim.AdamW(model.cls.parameters() if freeze else model.parameters(), lr)
    for _ in range(steps):
        model.train()
        loss = F.cross_entropy(model.cls(model(X)[:, 0]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
