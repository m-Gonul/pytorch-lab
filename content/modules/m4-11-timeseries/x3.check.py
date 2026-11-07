import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        n = 12000
        phi = np.where(np.arange(n)[:, None] < 6000, [0.6, 0.3], [-0.5, 0.3])
        y = np.zeros(n)
        e = rng.normal(size=n)
        for i in range(2, n):
            y[i] = phi[i, 0] * y[i - 1] + phi[i, 1] * y[i - 2] + e[i]
        pred, W = need(ns, "nlms")(y, 2, 0.05)
        close(W[4000:6000].mean(0), [0.6, 0.3], atol=0.1, what="ilk rejimde ağırlıklar (ortalama)")
        close(W[-2000:].mean(0), [-0.5, 0.3], atol=0.1, what="ikinci rejime uyum (ortalama)")
        X = np.stack([y[1:5999], y[:5998]], 1)
        fixed = np.linalg.lstsq(X, y[2:6000], rcond=None)[0]
        seg = slice(7000, n)
        idx = np.arange(n)[seg]
        fx = fixed[0] * y[idx - 1] + fixed[1] * y[idx - 2]
        oracle = phi[idx, 0] * y[idx - 1] + phi[idx, 1] * y[idx - 2]
        m = {"sabit AR(2) (ilk yarıya uydurulmuş)": np.mean((y[seg] - fx) ** 2), "NLMS μ = 0.05": np.mean((y[seg] - pred[seg]) ** 2), "gerçek katsayılar (alt sınır)": np.mean((y[seg] - oracle) ** 2)}
        ensure(m["NLMS μ = 0.05"] < 0.8 * m["sabit AR(2) (ilk yarıya uydurulmuş)"], "NLMS rejim değişimini izler")
        rows = [[k, f"{v:.3f}"] for k, v in m.items()]
        for mu in (0.005, 0.2, 0.8):
            p2, _ = ns["nlms"](y, 2, mu)
            rows.append([f"NLMS μ = {mu}", f"{np.mean((y[seg] - p2[seg]) ** 2):.3f}"])
        R_.table("AR(2) katsayıları t = 6000'de değişiyor; 7000 sonrası tek adım MSE", ["öngörücü", "MSE"], rows,
                 note="Sabit parametreli model (ARMA, eğitilmiş RNN) durağanlık varsayar; rejim değişince yanlış katsayılarla tahmin etmeye devam eder. LMS ailesi her adımda gradyan adımı atarak izler. μ küçükse yavaş uyum, büyükse gürültülü ağırlıklar (gecikme–gürültü ödünleşimi); NLMS adımı girdi gücüne normalize ederek kararlılığı μ ∈ (0, 2) aralığında garanti eder.")
    R_.test("uyarlamalı filtre", t)
