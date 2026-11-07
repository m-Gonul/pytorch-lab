import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_as_linear(emb):
    lin = nn.Linear(emb.num_embeddings, emb.embedding_dim, bias=False)
    with torch.no_grad():
        lin.weight.copy_(emb.weight.T)
    return lin


class SGNS(nn.Module):
    def __init__(self, V, d, sparse=False):
        super().__init__()
        self.inp = nn.Embedding(V, d, sparse=sparse)
        self.out = nn.Embedding(V, d, sparse=sparse)
        nn.init.normal_(self.inp.weight, 0, 0.1)
        nn.init.zeros_(self.out.weight)

    def forward(self, c, o, neg):
        v = self.inp(c)
        pos = F.logsigmoid((v * self.out(o)).sum(-1))
        negs = F.logsigmoid(-torch.einsum("bd,bkd->bk", v, self.out(neg))).sum(-1)
        return -(pos + negs).mean()


class CBOW(nn.Module):
    def __init__(self, V, d, pad):
        super().__init__()
        self.inp = nn.Embedding(V + 1, d, padding_idx=pad)
        self.out = nn.Embedding(V, d)
        nn.init.normal_(self.inp.weight, 0, 0.1)
        with torch.no_grad():
            self.inp.weight[pad].zero_()
        nn.init.zeros_(self.out.weight)
        self.pad = pad

    def forward(self, ctx, o, neg):
        m = (ctx != self.pad).float()
        h = self.inp(ctx).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        pos = F.logsigmoid((h * self.out(o)).sum(-1))
        negs = F.logsigmoid(-torch.einsum("bd,bkd->bk", h, self.out(neg))).sum(-1)
        return -(pos + negs).mean()


def train(model, batches, epochs, lr, sparse=False):
    opt = torch.optim.SparseAdam(list(model.parameters()), lr) if sparse else torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        for args in batches():
            loss = model(*args)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
