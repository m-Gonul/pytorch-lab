import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(0)
    Fv = torch.randn(20, generator=g, dtype=torch.float64)
    y = (torch.rand(20, generator=g) < 0.5).double()

    def pr():
        f = need(ns, "pseudo_residuals")
        close(f(Fv, y, lambda a, b: (a - b) ** 2), -2 * (Fv - y), what="MSE: −2(F − y)")
        close(f(Fv, y, lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction="none")), y - torch.sigmoid(Fv), what="logistic: y − σ(F)")
        close(f(Fv, y, lambda a, b: (a - b).abs()), -torch.sign(Fv - y), what="L1: −sign(F − y)")
    R.test("pseudo_residuals: MSE, logistic, L1", pr)

    def circ():
        n = 300
        r = torch.cat([torch.rand(n, generator=g) * 1.0, 1.6 + torch.rand(n, generator=g) * 1.0])
        t = torch.rand(2 * n, generator=g) * 6.2832
        X = torch.stack([r * t.cos(), r * t.sin()], 1).double()
        yy = torch.cat([torch.ones(n), torch.zeros(n)]).double()
        m = need(ns, "gbm_classify")(X, yy, rounds=200, lr=0.3)
        acc = ((need(ns, "gbm_logits")(m, X) > 0).double() == yy).double().mean().item()
        ensure(acc >= 0.9, f"doğruluk {acc:.2f}")
        return f"iç içe çemberler, 200 stump: doğruluk {acc:.3f}"
    R.test("gbm_classify çemberleri ayırıyor", circ)
