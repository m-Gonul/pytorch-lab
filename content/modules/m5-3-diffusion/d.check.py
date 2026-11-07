import math
import torch
from torch import nn
from labcheck import close, need, ensure


def ring(n):
    k = torch.randint(0, 8, (n,))
    ang = k * 2 * math.pi / 8
    return torch.stack([2 * ang.cos(), 2 * ang.sin()], 1) + 0.05 * torch.randn(n, 2)


class Oracle(nn.Module):
    def __init__(self, abar, mu, s):
        super().__init__(); self.abar, self.mu, self.s = abar, mu, s

    def forward(self, x, t):
        a = self.abar[t][:, None]
        return (1 - a).sqrt() * (x - a.sqrt() * self.mu) / (a * self.s ** 2 + 1 - a)


def check(ns, R):
    def sched():
        b, a = need(ns, "make_schedule")(10, 0.1, 0.5)
        close(b, torch.linspace(0.1, 0.5, 10)); close(a, torch.cumprod(1 - torch.linspace(0.1, 0.5, 10), 0))
    R.test("make_schedule", sched)

    def emb():
        e = need(ns, "TimeEmbedding", "sınıf")(8)(torch.tensor([0, 3]))
        f = torch.exp(-math.log(1000) * torch.arange(4) / 4)
        close(e, torch.cat([(torch.tensor([[0.0], [3.0]]) * f).sin(), (torch.tensor([[0.0], [3.0]]) * f).cos()], 1))
    R.test("TimeEmbedding", emb)

    def loss():
        torch.manual_seed(0)
        m = need(ns, "EpsNet", "sınıf")()
        _, abar = ns["make_schedule"]()
        x0 = torch.randn(6, 2); t = torch.tensor([0, 10, 99, 50, 3, 3]); e = torch.randn(6, 2)
        a = abar[t][:, None]
        want = ((m(a.sqrt() * x0 + (1 - a).sqrt() * e, t) - e) ** 2).mean()
        close(need(ns, "ddpm_loss")(m, x0, abar, t=t, eps=e), want)
    R.test("ddpm_loss formülü (t, eps verilince)", loss)

    def oracle():
        betas, abar = need(ns, "make_schedule")(300, 1e-4, 0.05)
        mu, s = torch.tensor([1.0, -2.0]), torch.tensor([0.3, 0.8])
        torch.manual_seed(0)
        x = need(ns, "ddpm_sample")(Oracle(abar, mu, s), betas, abar, 8000)
        ensure((x.mean(0) - mu).abs().max() < 0.05 and (x.std(0) - s).abs().max() < 0.05,
               f"ortalama {x.mean(0).tolist()}, std {x.std(0).tolist()}")
    R.test("ddpm_sample kâhinle Gauss veriyor (eğitimsiz)", oracle)

    def train():
        torch.manual_seed(0)
        betas, abar = need(ns, "make_schedule")()
        m = need(ns, "EpsNet", "sınıf")()
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for _ in range(3000):
            l = ns["ddpm_loss"](m, ring(512), abar)
            opt.zero_grad(); l.backward(); opt.step()
        x = ns["ddpm_sample"](m, betas, abar, 2000)
        ang = torch.arange(8) * 2 * math.pi / 8
        C = torch.stack([2 * ang.cos(), 2 * ang.sin()], 1)
        dmin, k = torch.cdist(x, C).min(1)
        good = dmin < 0.2
        modes = (torch.bincount(k[good], minlength=8) >= 40).sum().item()
        q = good.float().mean().item()
        ensure(modes == 8 and q > 0.8, f"{modes}/8 mod, yakın örnek oranı %{100 * q:.0f}")
        return f"{modes}/8 mod, bir moda 0.2'den yakın örnek oranı %{100 * q:.0f}"
    R.test("eğitim: 8 Gauss halkası", train)
