import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def digits16():
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8) / 16
    return F.interpolate(X, size=16, mode="bilinear", align_corners=False), torch.tensor(y)


class PatchEmbed(nn.Module):
    def __init__(self, P, C, D):
        super().__init__()
        self.P = P
        self.proj = nn.Conv2d(C, D, P, P)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

    def forward_unfold(self, x):
        cols = F.unfold(x, self.P, stride=self.P)
        return cols.transpose(1, 2) @ self.proj.weight.flatten(1).T + self.proj.bias


class ViT(nn.Module):
    def __init__(self, img=16, P=4, D=64, L=4, h=4, classes=10, pool="cls", pos=True):
        super().__init__()
        N = (img // P) ** 2
        self.pool = pool
        self.patch = PatchEmbed(P, 1, D)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.randn(1, N + 1, D) * 0.02) if pos else None
        layer = nn.TransformerEncoderLayer(D, h, 2 * D, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, classes)

    def forward(self, x):
        z = self.patch(x)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1)
        if self.pos is not None:
            z = z + self.pos
        z = self.norm(self.enc(z))
        return self.head(z[:, 0] if self.pool == "cls" else z[:, 1:].mean(1))


def train_eval(model, Xtr, ytr, Xte, yte, epochs):
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.05)
    for _ in range(epochs):
        model.train()
        p = torch.randperm(len(Xtr))
        for i in range(0, len(p), 32):
            b = p[i:i + 32]
            loss = F.cross_entropy(model(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()
