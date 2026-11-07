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
    x = torch.zeros(y.shape[0], w.shape[1], H, H, requires_grad=True)
    out = F.conv2d(x, w, stride=s, padding=p)
    (g,) = torch.autograd.grad(out, x, y)
    return g


def bilinear_init_(tconv):
    k = tconv.kernel_size[0]
    f = (k + 1) // 2
    c = f - 1 if k % 2 == 1 else f - 0.5
    t = 1 - (torch.arange(k, dtype=torch.float32) - c).abs() / f
    C = tconv.in_channels
    with torch.no_grad():
        tconv.weight.zero_()
        tconv.weight[range(C), range(C)] = torch.outer(t, t)
        if tconv.bias is not None:
            tconv.bias.zero_()
    return tconv


class FCN(nn.Module):
    def __init__(self, C=4, w=12, skips=False, bilinear=True):
        super().__init__()
        self.enc = Encoder(w)
        self.skips = skips
        self.s4, self.s3, self.s2 = nn.Conv2d(8 * w, C, 1), nn.Conv2d(4 * w, C, 1), nn.Conv2d(2 * w, C, 1)
        self.up8 = nn.ConvTranspose2d(C, C, 16, 8, 4)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(C, C, 4, 2, 1) for _ in range(3)])
        if bilinear:
            for m in [self.up8, *self.ups]:
                bilinear_init_(m)

    def forward(self, x):
        f1, f2, f3, f4 = self.enc(x)
        if not self.skips:
            return self.up8(self.s4(f4))
        y = self.ups[0](self.s4(f4)) + self.s3(f3)
        y = self.ups[1](y) + self.s2(f2)
        return self.ups[2](y)


class UNet(nn.Module):
    def __init__(self, C=4, w=12):
        super().__init__()
        self.enc = Encoder(w)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2), nn.Sequential(cbr(8 * w, 4 * w), cbr(4 * w, 4 * w))
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), nn.Sequential(cbr(4 * w, 2 * w), cbr(2 * w, 2 * w))
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), nn.Sequential(cbr(2 * w, w), cbr(w, w))
        self.out = nn.Conv2d(w, C, 1)

    def forward(self, x):
        f1, f2, f3, f4 = self.enc(x)
        y = self.d3(torch.cat([self.u3(f4), f3], 1))
        y = self.d2(torch.cat([self.u2(y), f2], 1))
        y = self.d1(torch.cat([self.u1(y), f1], 1))
        return self.out(y)


def miou(pred, gt, C):
    cm = torch.bincount(gt.flatten() * C + pred.flatten(), minlength=C * C).view(C, C).double()
    iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())
    return float(iou.nanmean())


def edge_acc(pred, gt, r=2):
    g = gt[:, None].float()
    k = 2 * r + 1
    edge = (F.max_pool2d(g, k, 1, r) != -F.max_pool2d(-g, k, 1, r))[:, 0]
    return float((pred == gt)[edge].float().mean())
