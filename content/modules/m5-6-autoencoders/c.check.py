import numpy as np
from sklearn.datasets import load_digits
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüler", lambda: (max_loops(src, 0, func="tied_sparse_ae"), max_loops(src, 1, func="train")))

    def t_grad():
        rng = np.random.default_rng(0)
        X = rng.random((7, 5))
        W, b, c = rng.normal(0, 0.5, (5, 3)), rng.normal(size=3), rng.normal(size=5)
        L, dW, db, dc = need(ns, "tied_sparse_ae")(X, W, b, c, 0.1, 0.7)
        h = 1e-6
        for P, G in ((W, dW), (b, db), (c, dc)):
            for idx in np.ndindex(P.shape):
                P[idx] += h
                Lp = ns["tied_sparse_ae"](X, W, b, c, 0.1, 0.7)[0]
                P[idx] -= h
                close((Lp - L) / h, G[idx], atol=2e-4, what=f"sayısal gradyan {idx}")
    R_.test("bağlı ağırlık + KL gradyanı", t_grad)

    def t_train():
        X = load_digits().data[:600] / 16
        rows = []
        res = {}
        for beta in (0.0, 0.5, 3.0):
            W, b, c = need(ns, "train")(X, 25, 0.05, beta, 0.5, 1500, 0)
            z = 1 / (1 + np.exp(-(X @ W + b)))
            mse = np.mean((z @ W.T + c - X) ** 2)
            res[beta] = (z.mean(), mse)
            rows.append([beta, f"{z.mean():.3f}", f"{(z > 0.5).mean():.3f}", f"{mse:.4f}"])
        ensure(res[3.0][0] < 0.12 and res[0.0][0] > res[3.0][0] + 0.1, "β ile ortalama aktivasyon hedefe iner")
        ensure(res[0.0][1] < np.var(X, 0).mean(), "AE ortalama tahminden iyi")
        R_.table("Rakamlar (600 × 64), 25 sigmoid kod nöronu, ρ = 0.05", ["β", "ortalama aktivasyon", "> 0.5 oranı", "yeniden yapılandırma MSE"], rows,
                 note="β = 0'da kod nöronları yarı yarıya açık (yoğun kod); β büyüdükçe her örnekte az sayıda nöron açılır ve ortalama aktivasyon ρ'ya iner. Bedel biraz daha yüksek yeniden yapılandırma hatası — karşılığında her nöron daha seçici (yerel kalem darbesi benzeri) bir özniteliğe karşılık gelir.")
    R_.test("seyrek eğitim", t_train)
