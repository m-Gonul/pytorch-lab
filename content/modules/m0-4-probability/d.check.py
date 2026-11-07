import math
import torch
from labcheck import close, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]
    R.test("torch.distributions kullanılıyor", lambda: ensure(uses_any(src, ["Normal", "log_prob"]), "Normal(...).log_prob ile yaz"))

    def nll():
        x = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
        v = need(ns, "normal_nll")(x, torch.tensor(4.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
        close(v, 0.5 * math.log(2 * math.pi) + (4 + 0 + 4) / 2 / 3)
        v2 = need(ns, "normal_nll")(x, torch.tensor(3.0, dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64))
        close(v2, 0.5 * math.log(2 * math.pi) + math.log(2) + (1 + 1 + 9) / (2 * 4) / 3, what="μ = 3, σ = 2")
    R.test("normal_nll: elle formülle aynı (ortalama)", nll)

    def fit():
        g = torch.Generator().manual_seed(0)
        x = 3.0 + 1.7 * torch.randn(400, generator=g)
        mu, sigma = need(ns, "fit_normal")(x)
        close(mu, x.mean(), atol=2e-3, rtol=0, what="μ̂")
        close(sigma, x.var(unbiased=False).sqrt(), atol=2e-3, rtol=0, what="σ̂ (MLE)")
        ensure(float(sigma) > 0, "σ pozitif olmalı")
        return f"gradyanla μ̂ = {float(mu):.4f}, σ̂ = {float(sigma):.4f}; kapalı form {float(x.mean()):.4f}, {float(x.std(unbiased=False)):.4f}"
    R.test("fit_normal kapalı form MLE'ye ulaşıyor", fit)
