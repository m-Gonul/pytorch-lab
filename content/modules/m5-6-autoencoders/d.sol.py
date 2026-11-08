import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def digits():
    X, y = load_digits(return_X_y=True)
    return torch.tensor(X, dtype=torch.float32) / 16, torch.tensor(y)


class AE(nn.Module):
    def __init__(self, sizes=(64, 32, 12), tied=False):
        super().__init__()
        self.sizes, self.tied = tuple(sizes), tied
        self.enc = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
        if tied:
            self.dec_b = nn.ParameterList([nn.Parameter(torch.zeros(a)) for a in sizes[:-1]][::-1])
        else:
            self.dec = nn.ModuleList([nn.Linear(b, a) for a, b in zip(sizes[:-1], sizes[1:])][::-1])

    def encode(self, x):
        n = len(self.enc)
        for i, layer in enumerate(self.enc):
            x = layer(x)
            if i < n - 1:
                x = F.relu(x)
        return x

    def decode(self, z):
        n = len(self.enc)
        for i in range(n):
            z = F.linear(z, self.enc[n - 1 - i].weight.t(), self.dec_b[i]) if self.tied else self.dec[i](z)
            z = F.relu(z) if i < n - 1 else torch.sigmoid(z)
        return z

    def forward(self, x):
        return self.decode(self.encode(x))


def train_ae(model, X, epochs=100, lr=3e-3, mask=0.0):
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            xb = X[perm[i:i + 64]]
            xin = xb * (torch.rand(xb.shape) >= mask).float() if mask > 0 else xb
            loss = F.mse_loss(model(xin), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def greedy_pretrain(sizes, X, epochs):
    full = AE(sizes)
    h = X
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        shallow_enc, shallow_dec = nn.Linear(a, b), nn.Linear(b, a)
        out_act = torch.sigmoid if i == 0 else F.relu
        opt = torch.optim.Adam(list(shallow_enc.parameters()) + list(shallow_dec.parameters()), 3e-3)
        for _ in range(epochs):
            perm = torch.randperm(len(h))
            for j in range(0, len(h), 64):
                hb = h[perm[j:j + 64]]
                code = shallow_enc(hb) if last else F.relu(shallow_enc(hb))
                loss = F.mse_loss(out_act(shallow_dec(code)), hb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        full.enc[i].load_state_dict(shallow_enc.state_dict())
        full.dec[len(sizes) - 2 - i].load_state_dict(shallow_dec.state_dict())
        with torch.no_grad():
            h = shallow_enc(h) if last else F.relu(shallow_enc(h))
    return full


def classify(encoder_ae, X, y, Xt, yt, freeze, steps=300):
    torch.manual_seed(1)
    base = AE(encoder_ae.sizes if encoder_ae is not None else (64, 32, 12))
    if encoder_ae is not None:
        for a, b in zip(base.enc, encoder_ae.enc):
            a.load_state_dict(b.state_dict())
    head = nn.Linear(base.sizes[-1], 10)
    params = list(head.parameters()) + ([] if freeze else list(base.enc.parameters()))
    opt = torch.optim.Adam(params, 3e-3)
    for _ in range(steps):
        loss = F.cross_entropy(head(F.relu(base.encode(X))), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (head(F.relu(base.encode(Xt))).argmax(1) == yt).float().mean().item()
