import math
import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma, ksize):
    t = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return g / g.sum()


class UnsharpMask(torch.nn.Module):
    def __init__(self, ksize=9):
        super().__init__()
        self.ksize = ksize
        self.log_sigma = torch.nn.Parameter(torch.tensor(0.0))
        self.amount = torch.nn.Parameter(torch.tensor(0.5))

    def blur(self, x):
        g = gaussian_kernel1d(self.log_sigma.exp(), self.ksize)
        r = self.ksize // 2
        y = F.pad(x[None, None], (r, r, r, r), mode="reflect")
        y = F.conv2d(y, g.view(1, 1, 1, -1))
        y = F.conv2d(y, g.view(1, 1, -1, 1))
        return y[0, 0]

    def forward(self, x):
        return x + self.amount * (x - self.blur(x))


def fit_unsharp(x, target, steps=400, lr=0.05):
    m = UnsharpMask()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        loss = ((m(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.log_sigma.exp().item(), m.amount.item()


from skimage import data
x = torch.tensor(data.moon(), dtype=torch.float32)[:128, :128] / 255
print(UnsharpMask()(x).shape)
