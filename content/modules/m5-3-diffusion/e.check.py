import math
import torch
from torch import nn
from labcheck import close, max_loops, need, ensure


def ring(n):
    k = torch.randint(0, 8, (n,))
    ang = k * 2 * math.pi / 8
    return torch.stack([2 * ang.cos(), 2 * ang.sin()], 1) + 0.05 * torch.randn(n, 2)


class VNet(nn.Module):
    def __init__(self, h=128, td=32):
        super().__init__()
        half = td // 2
        self.register_buffer("f", torch.exp(-math.log(1000) * torch.arange(half) / half))
        self.net = nn.Sequential(nn.Linear(2 + td, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        a = (t * 1000)[:, None] * self.f
        return self.net(torch.cat([x, a.sin(), a.cos()], 1))


def check(ns, R):
    src = ns["__source__"]
    R.test("fm_loss döngüsüz", lambda: max_loops(src, 0, func="fm_loss"))
    R.test("euler_sample: tek döngü", lambda: max_loops(src, 1, func="euler_sample"))

    def lf():
        g = torch.Generator().manual_seed(0)
        W = torch.randn(3, 3, generator=g)
        model = lambda x, t: x @ W + t[:, None]
        x1, z = torch.randn(5, 3, generator=g), torch.randn(5, 3, generator=g); t = torch.rand(5, generator=g)
        xt = (1 - t[:, None]) * z + t[:, None] * x1
        close(need(ns, "fm_loss")(model, x1, z, t), ((model(xt, t) - (x1 - z)) ** 2).mean())
    R.test("fm_loss formülü", lf)
    mu, s = 3.0, 0.5
    oracle = lambda x, t: mu + (t[:, None] * s ** 2 - (1 - t[:, None])) / ((1 - t[:, None]) ** 2 + t[:, None] ** 2 * s ** 2) * (x - t[:, None] * mu)

    def steps():
        torch.manual_seed(0)
        z = torch.randn(20000, 1, dtype=torch.float64)
        rows = []
        for n in (1, 2, 5, 20, 100):
            x = need(ns, "euler_sample")(oracle, z, n)
            rows.append([str(n), f"{x.mean().item():.3f}", f"{x.std().item():.3f}"])
        x = ns["euler_sample"](oracle, z, 100)
        ensure(abs(x.mean().item() - mu) < 0.02 and abs(x.std().item() - s) < 0.02, f"100 adımda std {x.std().item():.3f}")
        x1 = ns["euler_sample"](oracle, z, 1)
        ensure(x1.std().item() < 1e-6, "tek adım her örneği ortalamaya götürmeli (kâhinle)")
        R.table("Kâhin hızla Euler: hedef N(3, 0.5²)", ["adım", "ortalama", "std"], rows,
                note="Tek adım koşullu ortalamaya gider: varyans 0. Adım arttıkça doğru dağılıma yaklaşır.")
    R.test("kâhin hızla adım sayısı", steps)

    def train():
        torch.manual_seed(0)
        m = VNet()
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for _ in range(3000):
            x1 = ring(512)
            l = need(ns, "fm_loss")(m, x1, torch.randn_like(x1), torch.rand(512))
            opt.zero_grad(); l.backward(); opt.step()
        x = ns["euler_sample"](m, torch.randn(2000, 2), 50)
        ang = torch.arange(8) * 2 * math.pi / 8
        C = torch.stack([2 * ang.cos(), 2 * ang.sin()], 1)
        dmin, k = torch.cdist(x, C).min(1)
        good = dmin < 0.2
        modes = (torch.bincount(k[good], minlength=8) >= 40).sum().item()
        q = good.float().mean().item()
        ensure(modes == 8 and q > 0.7, f"{modes}/8 mod, yakın örnek oranı %{100 * q:.0f}")
        return f"50 Euler adımı: {modes}/8 mod, yakın örnek oranı %{100 * q:.0f} (DDPM'de 100 adım)"
    R.test("eğitim: 8 Gauss halkası", train)
