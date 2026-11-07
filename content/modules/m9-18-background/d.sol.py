import numpy as np
import torch


class GMM:
    def __init__(self, shape, K=3, alpha=0.01, var0=225.0, T=0.7, w0=0.05):
        H, W = shape
        self.K, self.alpha, self.var0, self.T, self.w0 = K, alpha, var0, T, w0
        self.w = torch.zeros(K, H, W, dtype=torch.float64)
        self.mu = torch.zeros(K, H, W, dtype=torch.float64)
        self.var = torch.full((K, H, W), var0, dtype=torch.float64)
        self.started = False

    def apply(self, frame):
        x = torch.as_tensor(frame, dtype=torch.float64)
        if not self.started:
            self.w[0], self.mu[0] = 1.0, x
            self.started = True
            return torch.zeros_like(x, dtype=torch.bool)
        o = (self.w / self.var.sqrt()).argsort(0, descending=True)
        w, mu, var = self.w.gather(0, o), self.mu.gather(0, o), self.var.gather(0, o)
        match = ((x - mu).abs() < 2.5 * var.sqrt()) & (w > 0)
        first = match & (match.int().cumsum(0) == 1)
        anym = first.any(0)
        cw = w.cumsum(0)
        isbg = (cw - w) < self.T
        fg = ~(first & isbg).any(0)
        a = self.alpha
        w = (1 - a) * w + a * first.double()
        d = x - mu
        mu = torch.where(first, mu + a * d, mu)
        var = torch.where(first, var + a * (d * d - var), var)
        last = torch.zeros_like(first); last[-1] = ~anym
        w = torch.where(last, torch.full_like(w, self.w0), w)
        mu = torch.where(last, x.expand_as(mu), mu)
        var = torch.where(last, torch.full_like(var, self.var0), var)
        self.w, self.mu, self.var = w / w.sum(0, keepdim=True), mu, var
        return fg


g = GMM((4, 5))
f = torch.full((4, 5), 100.0, dtype=torch.float64)
print(g.apply(f), g.apply(f + 120))
