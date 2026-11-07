import math
import torch
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(1)

    def lossf():
        x, xh, mu, lv = (torch.randn(5, k, generator=g) for k in (4, 4, 2, 2))
        want = (((x - xh) ** 2).sum(-1) + 0.5 * 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1)).mean()
        close(need(ns, "elbo_loss")(x, xh, mu, lv, beta=0.5), want)
    R.test("elbo_loss formülü (β = 0.5)", lossf)

    def shapes():
        torch.manual_seed(0)
        m = need(ns, "VAE", "sınıf")(12, 3)
        x = torch.randn(7, 12)
        mu, lv = m.encode(x)
        ensure(tuple(mu.shape) == (7, 3) and tuple(lv.shape) == (7, 3), "encode → (B, d_lat) iki tensör")
        xh, mu2, _ = m(x)
        ensure(tuple(xh.shape) == (7, 12), "forward → xhat (B, d_in)")
        ensure(tuple(need(ns, "sample")(m, 4).shape) == (4, 12), "sample(model, 4) → (4, d_in)")
    R.test("şekiller", shapes)

    def reparam():
        torch.manual_seed(0)
        m = need(ns, "VAE", "sınıf")(12, 3)
        x = torch.randn(7, 12)
        torch.manual_seed(5); a, _, _ = m(x)
        torch.manual_seed(6); b, _, _ = m(x)
        ensure(not torch.allclose(a, b), "forward örneklemeli (rastgele) olmalı")
        a.sum().backward()
        enc_params = [p for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        ensure(len(enc_params) == len(list(m.parameters())), "tüm parametreler (kodlayıcı dahil) yeniden yapılandırmadan gradyan almalı")
    R.test("reparametrization: gradyan kodlayıcıya akıyor", reparam)

    def train():
        A = torch.randn(2, 12, generator=g)
        X = torch.tanh(torch.randn(1000, 2, generator=g) @ A) + 0.05 * torch.randn(1000, 12, generator=g)
        torch.manual_seed(0)
        m = need(ns, "VAE", "sınıf")(12, 2)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for step in range(1000):
            loss = ns["elbo_loss"](X, *m(X))
            if step == 0:
                first = loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            mu, lv = m.encode(X)
            rec = ((X - m.decode(mu)) ** 2).sum(-1).mean().item()
            kl = (0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1)).mean().item()
        var = X.var(0).sum().item()
        ensure(loss.item() < 0.6 * first, f"kayıp {first:.2f} → {loss.item():.2f}")
        ensure(rec < 0.3 * var, f"yeniden yapılandırma {rec:.3f}, veri varyansı {var:.3f}")
        ensure(kl > 0.5, f"KL {kl:.3f}: gizil uzay kullanılmıyor (posterior collapse)")
        return f"kayıp {first:.2f} → {loss.item():.2f}; rec {rec:.3f} / var {var:.3f}; KL {kl:.2f} nat"
    R.test("sentetik veride eğitim", train)
