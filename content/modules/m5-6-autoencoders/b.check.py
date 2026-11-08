import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        ensure(need(ns, "ae_params")([784, 100, 30]) == 163814 and ns["ae_params"]([784, 100, 30], True) == 82414, "Géron örneği")
        close(need(ns, "kl_sparsity")(0.05, 0.2), 0.05 * math.log(0.25) + 0.95 * math.log(0.95 / 0.8), what="KL")
        close(ns["kl_sparsity"](0.05, 0.05), 0.0, what="hedefte 0")
        h = 1e-7
        close(need(ns, "kl_sparsity_grad")(0.05, 0.2), (ns["kl_sparsity"](0.05, 0.2 + h) - ns["kl_sparsity"](0.05, 0.2)) / h, atol=1e-4, what="türev")
        ensure(need(ns, "greedy_plan")([784, 100, 30]) == [(784, 100), (100, 30)], "aşamalar")
        rng = random.Random(0)
        x = [1.0] * 10000
        m = need(ns, "masking_noise")(x, 0.5, rng)
        ensure(abs(sum(v == 0 for v in m) / 10000 - 0.5) < 0.02 and set(m) <= {0.0, 1.0}, "maskeleme")
        gn = need(ns, "gaussian_noise")(x, 0.3, rng)
        mu = sum(gn) / len(gn)
        close(math.sqrt(sum((v - mu) ** 2 for v in gn) / len(gn)), 0.3, atol=0.01, what="Gauss gürültüsü std")
        rows = []
        for rh in (0.01, 0.05, 0.1, 0.2, 0.5, 0.9):
            rows.append([rh, f"{ns['kl_sparsity'](0.05, rh):.3f}", f"{ns['kl_sparsity_grad'](0.05, rh):+.2f}", f"{abs(rh - 0.05):.2f}"])
        R_.table("Seyreklik cezası, hedef ρ = 0.05", ["ρ̂", "KL(ρ‖ρ̂)", "∂/∂ρ̂", "|ρ̂ − ρ| (L1 benzeri)"], rows,
                 note="KL cezası asimetriktir: ρ̂ 1'e yaklaşırken çok hızlı büyür, 0'a giderken de büyür — nöron tamamen ölmez. Hedefin altında (ρ̂ = 0.01) gradyan negatiftir: aktivasyonu yükseltmeye iter.")
    R_.test("sayımlar ve cezalar", t)
