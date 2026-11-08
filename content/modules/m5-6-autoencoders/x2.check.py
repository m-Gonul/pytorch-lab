import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("encoder_jacobian", "cae_penalty")])

    def t():
        rng = np.random.default_rng(0)
        W, b = rng.normal(size=(4, 6)), rng.normal(size=4)
        x = rng.normal(size=6)
        J = need(ns, "encoder_jacobian")(x, W, b)
        f = lambda v: 1 / (1 + np.exp(-(W @ v + b)))
        Jn = np.stack([(f(x + 1e-6 * e) - f(x)) / 1e-6 for e in np.eye(6)], 1)
        close(J, Jn, atol=1e-5, what="Jacobian")
        X = rng.normal(size=(30, 6))
        close(need(ns, "cae_penalty")(X, W, b), np.mean([np.sum(ns["encoder_jacobian"](xx, W, b) ** 2) for xx in X]), what="‖J‖²_F ortalaması")
        rows = []
        for s in (0.1, 1.0, 3.0, 10.0):
            rows.append([s, f"{ns['cae_penalty'](X, s * W, b):.4f}"])
        R_.table("Ağırlık ölçeği ve büzülme cezası", ["W ölçeği", "ort. ‖J‖²_F"], rows,
                 note="Ceza iki yolla küçülür: küçük ağırlıklar ya da doymuş nöronlar (h ≈ 0 veya 1 → h(1 − h) ≈ 0). Büzüşen AE (Rifai ve ark.) kodun girdideki küçük değişimlere duyarsız olmasını ister ama yeniden yapılandırma bunu her yönde yapmasını engeller: sonuç, veri manifoldu boyunca duyarlı, ona dik yönlerde duyarsız bir kod — DAE ile yakın akrabadır (küçük gürültü limitinde eşdeğer).")
    R_.test("büzüşen AE", t)
