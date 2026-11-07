import math
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def ring(n):
    k = torch.randint(0, 8, (n,))
    ang = k * 2 * math.pi / 8
    return torch.stack([2 * ang.cos(), 2 * ang.sin()], 1) + 0.05 * torch.randn(n, 2)


def check(ns, R):
    def losses():
        torch.manual_seed(0)
        G, D = need(ns, "Generator", "sınıf")(), need(ns, "Discriminator", "sınıf")()
        real = ring(16); fake = G(torch.randn(16, 2))
        lr, lf = D(real), D(fake)
        ensure(tuple(lr.shape) in ((16, 1), (16,)), "ayırıcı (B, 1) logit döndürmeli")
        want_d = F.softplus(-lr).mean() + F.softplus(lf).mean()
        close(need(ns, "d_loss")(D, real, fake), want_d, rtol=1e-5)
        close(need(ns, "g_loss")(D, fake), F.softplus(-lf).mean(), rtol=1e-5)
    R.test("kayıp formülleri", losses)

    def detach():
        torch.manual_seed(0)
        G, D = ns["Generator"](), ns["Discriminator"]()
        need(ns, "d_loss")(D, ring(8), G(torch.randn(8, 2))).backward()
        ensure(all(p.grad is None or p.grad.abs().sum() == 0 for p in G.parameters()), "d_loss G'ye gradyan taşımamalı (fake.detach())")
        ensure(any(p.grad is not None for p in D.parameters()), "d_loss D'ye gradyan vermeli")
    R.test("d_loss sahte örnekleri detach ediyor", detach)

    def run(seed):
        torch.manual_seed(seed)
        G, D = need(ns, "Generator", "sınıf")(), need(ns, "Discriminator", "sınıf")()
        need(ns, "train_gan")(G, D, ring, steps=3000)
        with torch.no_grad():
            s = G(torch.randn(2000, 2))
        ang = torch.arange(8) * 2 * math.pi / 8
        C = torch.stack([2 * ang.cos(), 2 * ang.sin()], 1)
        dmin, k = torch.cdist(s, C).min(1)
        good = dmin < 0.2
        return (torch.bincount(k[good], minlength=8) >= 40).sum().item(), good.float().mean().item()

    def train():
        tried = []
        for seed in (1, 2):
            modes, q = run(seed)
            tried.append(f"tohum {seed}: {modes}/8 mod, %{100 * q:.0f}")
            if modes >= 7 and q > 0.6:
                return f"{modes}/8 mod, bir moda 0.2'den yakın örnek oranı %{100 * q:.0f}"
        raise AssertionError("mod çöküşü ya da dağınık örnekler — " + "; ".join(tried))
    R.test("8 Gauss halkası: modlar kapsanıyor", train)
