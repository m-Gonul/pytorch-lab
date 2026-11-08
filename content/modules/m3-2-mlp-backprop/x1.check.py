import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        sizes = [4, 5, 3, 2]
        Ws = [rng.normal(size=(b, a)) for a, b in zip(sizes[:-1], sizes[1:])]
        bs = [rng.normal(size=(b, 1)) for b in sizes[1:]]
        x, y = rng.normal(size=(4, 1)), rng.random((2, 1))
        sig = lambda z: 1 / (1 + np.exp(-z))

        def C():
            a = x
            for W, b in zip(Ws, bs):
                a = sig(W @ a + b)
            return 0.5 * np.sum((a - y) ** 2)
        deltas, dWs, dbs = need(ns, "backprop")(Ws, bs, x, y)
        h = 1e-6
        c0 = C()
        for l in range(3):
            for idx in [(0, 0), (1, 1)]:
                Ws[l][idx] += h
                close((C() - c0) / h, dWs[l][idx], atol=1e-5, what=f"BP4 katman {l}")
                Ws[l][idx] -= h
            bs[l][0, 0] += h
            close((C() - c0) / h, dbs[l][0, 0], atol=1e-5, what=f"BP3 katman {l}")
            bs[l][0, 0] -= h
        rows = []
        for scale in (1.0, 0.1, 0.01):
            x2 = x * 0 + scale
            Ws0 = [W.copy() for W in Ws]
            Ws0[0] = np.zeros_like(Ws0[0])
            _, dW, db = ns["backprop"](Ws, bs, x2, y)
            rows.append([scale, f"{np.abs(dW[0]).mean():.2e}", f"{np.abs(db[0]).mean():.2e}", f"{np.abs(dW[0]).mean() / np.abs(db[0]).mean():.3f}"])
        ensure(float(rows[2][3]) < 0.02, "küçük girişte ağırlık gradyanı ≪ bias gradyanı")
        R_.table("İlk katman: giriş aktivasyonları küçüldükçe (BP4)", ["giriş değeri", "ort. |∂C/∂W|", "ort. |∂C/∂b|", "oran"], rows,
                 note="BP4: ∂C/∂w = a_giriş · δ. Girişi ≈ 0 olan bir nöronun (ya da düşük aktivasyonlu bir gizli nöronun) çıkış ağırlıkları çok yavaş öğrenir; bias'ın gradyanı yalnız δ'ya bağlıdır (BP3). Aynı şekilde BP1–BP2'de σ'(z) ≈ 0 (doymuş nöron) δ'yı ve dolayısıyla tüm gelen ağırlıkların öğrenmesini durdurur.")
    R_.test("BP1–BP4", t)
