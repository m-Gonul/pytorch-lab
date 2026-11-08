import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        x0, eps = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        rows = []
        for abar in (0.999, 0.5, 0.01):
            a, s = np.sqrt(abar), np.sqrt(1 - abar)
            xt = a * x0 + s * eps
            close(need(ns, "eps_to_x0")(xt, eps, abar), x0, atol=1e-9, what="ε → x₀")
            close(need(ns, "x0_to_eps")(xt, x0, abar), eps, atol=1e-9, what="x₀ → ε")
            v = a * eps - s * x0
            close(need(ns, "eps_to_v")(xt, eps, abar), v, atol=1e-9, what="ε → v")
            close(need(ns, "v_to_eps")(xt, v, abar), eps, atol=1e-9, what="v → ε")
            close(need(ns, "eps_to_score")(eps, abar), -eps / s, atol=1e-9, what="skor")
            err = 0.01 * rng.normal(size=(5, 3))
            rows.append([abar, f"{np.abs(ns['eps_to_x0'](xt, eps + err, abar) - x0).mean():.4f}", f"{np.abs((a * xt - s * (v + err)) - x0).mean():.4f}"])
        R_.table("ε'da ya da v'de 0.01'lik tahmin hatasının x̂₀'a etkisi", ["ᾱ", "ε-tahmini → |Δx̂₀|", "v-tahmini → |Δx̂₀|"], rows,
                 note="x̂₀ = (x_t − σε)/α: ᾱ → 0'da (saf gürültü) ε'daki küçük bir hata 1/α kadar büyütülür. v = αε − σx₀ ile x̂₀ = αx_t − σv ve hata büyütmesi en fazla 1'dir: v-parametrizasyonu yüksek gürültü ucunda kararlıdır (progresif damıtma, Stable Diffusion 2). Hepsi aynı bilgiyi taşır; skor = −ε/σ.")
    R_.test("parametrizasyonlar", t)
