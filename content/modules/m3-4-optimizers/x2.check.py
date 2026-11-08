import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        st = {}
        w = need(ns, "adagrad_step")(np.array([1.0, 1.0]), np.array([2.0, -0.5]), st, 0.1)
        close(w, [0.9, 1.1], what="AdaGrad ilk adım: işaret × η")
        w = ns["adagrad_step"](w, np.array([2.0, -0.5]), st, 0.1)
        close(w, [0.9 - 0.1 * 2 / np.sqrt(8), 1.1 + 0.1 * 0.5 / np.sqrt(0.5)], what="birikmiş G")
        st = {}
        w = need(ns, "adamax_step")(np.array([0.0]), np.array([-4.0]), st, 0.1)
        close(w, [0.1], what="AdaMax ilk adım")
        w = ns["adamax_step"](w, np.array([1.0]), st, 0.1)
        m = 0.9 * (-0.4) + 0.1 * 1
        close(w, [0.1 - 0.1 / (1 - 0.81) * m / (0.999 * 4.0)], atol=1e-7, what="u = max(β₂u, |g|)")
        st = {}
        w = need(ns, "nadam_step")(np.array([0.0]), np.array([-6.0]), st, 0.1)
        mh = 0.9 * (-0.6) / (1 - 0.81) + 0.1 * (-6) / 0.1
        close(w, [-0.1 * mh / 6.0], atol=1e-7, what="Nadam ilk adım")
        rng = np.random.default_rng(0)
        N, d = 4000, 50
        X = (rng.random((N, d)) < 0.5).astype(float)
        X[:, 0] = rng.random(N) < 0.01
        wt = np.zeros(d)
        wt[0], wt[1:] = 6.0, rng.normal(0, 0.3, d - 1)
        y = (rng.random(N) < 1 / (1 + np.exp(-(X @ wt - 0.5 * (X[:, 1:] @ wt[1:]).mean())))).astype(float)
        rows = []
        res = {}
        for name, step, lr in (("SGD", None, 0.1), ("AdaGrad", ns["adagrad_step"], 0.1), ("AdaMax", ns["adamax_step"], 0.02), ("Nadam", ns["nadam_step"], 0.02)):
            w, st = np.zeros(d), {}
            for ep in range(3):
                for i in range(0, N, 20):
                    xb, yb = X[i:i + 20], y[i:i + 20]
                    g = xb.T @ (1 / (1 + np.exp(-xb @ w)) - yb) / len(xb)
                    w = w - lr * g if step is None else step(w, g, st, lr)
            res[name] = w[0]
            rows.append([name, lr, f"{w[0]:.2f}", f"{np.corrcoef(w[1:], wt[1:])[0, 1]:.2f}"])
        ensure(res["AdaGrad"] > 2 * res["SGD"], "seyrek öznitelik: uyarlamalı yöntem nadir özniteliğin ağırlığını hızla öğrenir")
        R_.table("Lojistik regresyon, öznitelik 0 örneklerin yalnız %1'inde açık (gerçek ağırlık 6), 3 epoch", ["yöntem", "η", "w₀ (nadir öznitelik)", "corr(w, w*) diğerleri"], rows,
                 note="SGD'de nadir öznitelik nadiren gradyan alır ve her seferinde aynı küçük η ile ilerler. AdaGrad her koordinatı geçmiş gradyan büyüklüğüne böler: az gradyan görmüş koordinat büyük adım atar (seyrek veride — kelimeler, tıklamalar — tarihsel başarısı). Bedeli: G hiç küçülmediği için uzun eğitimde öğrenme oranı sıfıra iner; RMSProp/Adam EMA ile bunu düzeltir.")
    R_.test("AdaGrad, AdaMax, Nadam", t)
