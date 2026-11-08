import numpy as np
from sklearn.datasets import load_digits
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="max_norm"))

    def t():
        W = np.array([[3.0, 0.1], [4.0, 0.2]])
        close(need(ns, "max_norm")(W, 1.0), [[0.6, 0.1], [0.8, 0.2]], what="yalnız normu > r olan sütunlar ölçeklenir")
        X, y = load_digits(return_X_y=True)
        X = X / 16
        rng = np.random.default_rng(0)
        p = rng.permutation(len(X))
        tr, te = p[:300], p[300:]
        rows = []
        res = {}
        for r in (None, 3.0, 1.5, 1.0):
            W1, b1, W2, b2 = need(ns, "train")(X[tr], y[tr], 256, 0.5, 60, r, 0)
            acc = lambda idx: (np.argmax(np.maximum(0, X[idx] @ W1 + b1) @ W2 + b2, 1) == y[idx]).mean()
            res[r] = (acc(tr), acc(te), max(np.linalg.norm(W1, axis=0).max(), np.linalg.norm(W2, axis=0).max()))
            rows.append(["yok" if r is None else r, f"{res[r][0]:.3f}", f"{res[r][1]:.3f}", f"{res[r][2]:.2f}"])
        ensure(res[3.0][2] <= 3.0 + 1e-9 and res[1.5][2] <= 1.5 + 1e-9, "kısıt her adımda uygulanır")
        ensure(res[None][2] > 3.0 and res[1.0][1] < res[None][1], "kısıtsız normlar büyür; çok sıkı kısıt eksik uyum")
        R_.table("Rakamlar, 300 eğitim örneği, 256 gizli nöron, η = 0.5, 60 epoch", ["r", "eğitim doğr.", "test doğr.", "maks. sütun normu (W₁, W₂)"], rows,
                 note="Max-norm bir ceza değil, bir izdüşümdür: kısıt sınırına ulaşmayan nöronlara hiç dokunmaz (r = 3'te W₁ hiç etkilenmez). Bu küçük problemde etkisi sınırlı, r = 1 ise eksik uyuma yol açar. Büyük η'da bile ağırlıklar sınırsız büyüyemez — Srivastava ve ark. dropout'u yüksek öğrenme oranı + momentum + max-norm (r ≈ 3–4) ile birlikte kullanmayı önerir.")
    R_.test("max-norm", t)
