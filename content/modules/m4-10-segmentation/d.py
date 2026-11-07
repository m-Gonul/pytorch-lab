import torch
import torch.nn as nn
import torch.nn.functional as F


def seg_dataset(n, seed, size=64):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij")
    X = torch.zeros(n, 1, size, size)
    Y = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        for _ in range(int(torch.randint(2, 6, (1,), generator=g))):
            s = float(torch.randint(6, 24, (1,), generator=g))
            cx, cy = (torch.rand(2, generator=g) * size).tolist()
            c = int(torch.randint(0, 3, (1,), generator=g))
            if c == 0:
                m = ((xx - cx).abs() <= s / 2) & ((yy - cy).abs() <= s / 2)
            elif c == 1:
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (s / 2) ** 2
            else:
                m = ((xx - cx).abs() + (yy - cy).abs()) <= s / 2
            X[i, 0][m] = 0.5 + 0.5 * float(torch.rand(1, generator=g))
            Y[i][m] = c + 1
    X += 0.15 * torch.randn(X.shape, generator=g)
    return X, Y


def cbr(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())


class Encoder(nn.Module):
    def __init__(self, w=12):
        super().__init__()
        self.e1 = nn.Sequential(cbr(1, w), cbr(w, w))
        self.e2 = nn.Sequential(cbr(w, 2 * w), cbr(2 * w, 2 * w))
        self.e3 = nn.Sequential(cbr(2 * w, 4 * w), cbr(4 * w, 4 * w))
        self.e4 = nn.Sequential(cbr(4 * w, 8 * w), cbr(8 * w, 8 * w))

    def forward(self, x):
        f1 = self.e1(x)
        f2 = self.e2(F.max_pool2d(f1, 2))
        f3 = self.e3(F.max_pool2d(f2, 2))
        f4 = self.e4(F.max_pool2d(f3, 2))
        return f1, f2, f3, f4


def conv_input_grad(y, w, s, p, H):
    ...


def bilinear_init_(tconv):
    ...


class FCN(nn.Module):
    def __init__(self, C=4, w=12, skips=False, bilinear=True):
        super().__init__()
        ...

    def forward(self, x):
        ...


class UNet(nn.Module):
    def __init__(self, C=4, w=12):
        super().__init__()
        ...

    def forward(self, x):
        ...


def miou(pred, gt, C):
    ...


def edge_acc(pred, gt, r=2):
    ...
