import math
import torch


def soft_histogram(x, bins=64, sigma=None):
    centers = (torch.arange(bins, dtype=x.dtype) + 0.5) / bins
    sigma = sigma or 1.0 / bins
    w = torch.exp(-((x.reshape(-1, 1) - centers) ** 2) / (2 * sigma ** 2))
    h = w.sum(0)
    return h / h.sum()


class ToneCurve(torch.nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.n = n
        self.logits = torch.nn.Parameter(torch.zeros(n))

    def knots(self):
        return torch.cat([torch.zeros(1), torch.cumsum(torch.softmax(self.logits, 0), 0)])

    def forward(self, x):
        k = self.knots()
        pos = x.clamp(0, 1) * self.n
        i = pos.floor().clamp(max=self.n - 1).long()
        f = pos - i
        return k[i] * (1 - f) + k[i + 1] * f


def learn_equalization(x, steps=300, lr=0.05):
    curve = ToneCurve()
    opt = torch.optim.Adam(curve.parameters(), lr=lr)
    sub = x.reshape(-1)[::7]
    for _ in range(steps):
        h = soft_histogram(curve(sub))
        loss = (h * torch.log(h + 1e-12)).sum()      # −entropi
        opt.zero_grad()
        loss.backward()
        opt.step()
    return curve


from skimage import data
x = torch.tensor(data.moon(), dtype=torch.float32) / 255
curve = learn_equalization(x)
print(curve(torch.linspace(0, 1, 5)))
