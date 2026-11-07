import torch
import torch.nn.functional as F
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    x = torch.tensor(data.camera(), dtype=torch.float32)[::4, ::4] / 255

    def gk():
        s = torch.tensor(1.5, requires_grad=True)
        g = need(ns, "gaussian_kernel1d")(s, 9)
        close(g.sum(), 1.0, atol=1e-6)
        ensure(g.argmax().item() == 4, "tepe ortada")
        g[4].backward()
        ensure(s.grad is not None and s.grad < 0, "σ büyüdükçe merkez ağırlık azalır: dg/dσ < 0")
    R.test("gaussian_kernel1d türevlenebilir", gk)

    def fit():
        m = need(ns, "UnsharpMask", "sınıf")()
        with torch.no_grad():
            m.log_sigma.fill_(torch.tensor(1.8).log()); m.amount.fill_(1.6)
            target = m(x)
        torch.manual_seed(0)
        s, a = need(ns, "fit_unsharp")(x, target)
        ensure(abs(s - 1.8) < 0.05 and abs(a - 1.6) < 0.05, f"bulunan σ = {s:.3f}, kazanç = {a:.3f}; gerçek 1.8, 1.6")
        return f"σ = {s:.3f} (1.8), kazanç = {a:.3f} (1.6)"
    R.test("fit_unsharp parametreleri geri buluyor", fit)
