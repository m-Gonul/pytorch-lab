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
    ...


def train_seg(loss, steps, seed):
    ...


def seg_metrics(P, Y):
    ...
