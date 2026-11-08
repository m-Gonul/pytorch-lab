import torch
import torch.nn as nn
import torch.nn.functional as F


def lesions(n, seed, S=32):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(S).float(), torch.arange(S).float(), indexing="ij")
    X = F.avg_pool2d(0.3 * torch.randn(n, 1, S, S, generator=g), 3, 1, 1) * 3
    Y = torch.zeros(n, S, S)
    for i in range(n):
        for _ in range(int(torch.randint(0, 4, (1,), generator=g))):
            r = 1.5 + 3 * float(torch.rand(1, generator=g))
            cx, cy = (torch.rand(2, generator=g) * (S - 8) + 4).tolist()
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            Y[i][m] = 1
            X[i, 0][m] += 0.8
    return X + 0.3 * torch.randn(X.shape, generator=g), Y


def _cbr(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())


class SegNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.e1 = nn.Sequential(_cbr(1, c), _cbr(c, c))
        self.e2 = nn.Sequential(_cbr(c, 2 * c), _cbr(2 * c, 2 * c))
        self.up = nn.ConvTranspose2d(2 * c, c, 2, 2)
        self.d1 = nn.Sequential(_cbr(2 * c, c), _cbr(c, c))
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        return self.out(self.d1(torch.cat([self.up(self.e2(F.max_pool2d(a, 2))), a], 1)))[:, 0]


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2))
    return (1 - (2 * inter + eps) / (p.sum((1, 2)) + y.sum((1, 2)) + eps)).mean()


def train_seg(loss, steps, seed):
    X, Y = lesions(800, 0)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = SegNet()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(steps):
        idx = torch.randint(0, 800, (32,), generator=g)
        lo, yb = m(X[idx]), Y[idx]
        if loss == "bce":
            L = F.binary_cross_entropy_with_logits(lo, yb)
        elif loss == "wbce":
            L = F.binary_cross_entropy_with_logits(lo, yb, pos_weight=torch.tensor(20.0))
        elif loss == "dice":
            L = dice_loss(lo, yb)
        else:
            L = F.binary_cross_entropy_with_logits(lo, yb) + dice_loss(lo, yb)
        opt.zero_grad()
        L.backward()
        opt.step()
    return m.eval()


def seg_metrics(P, Y):
    inter = (P * Y).sum()
    return float(2 * inter / (P.sum() + Y.sum())), float(inter / Y.sum()), float(inter / P.sum().clamp(min=1))
