import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        a = rng.normal(3, 5, (4, 10))
        y = need(ns, "ln")(a, np.ones(10), np.zeros(10))
        close((y.mean(-1), y.std(-1)), (np.zeros(4), np.ones(4)), atol=1e-4, what="LN: örnek başına")
        X = rng.normal(size=(200, 16, 8))
        H = 64
        rows = []
        res = {}
        for scale in (0.5, 1.0, 2.0):
            Wx = rng.normal(0, 1 / np.sqrt(8), (8, H))
            Wh = rng.normal(0, scale / np.sqrt(H), (H, H))
            for name, lnp in (("RNN", None), ("LN-RNN", (np.ones(H), np.zeros(H)))):
                Hs = need(ns, "rnn_run")(X, Wx, Wh, np.zeros(H), lnp)
                sat = (np.abs(Hs[-50:]) > 0.97).mean()
                res[(scale, name)] = (np.abs(Hs[-50:]).mean(), sat)
                rows.append([scale, name, f"{np.abs(Hs[-50:]).mean():.3f}", f"{sat:.3f}"])
        ensure(res[(2.0, "RNN")][1] > 3 * res[(2.0, "LN-RNN")][1], "LN doymayı engeller")
        ensure(abs(res[(0.5, "LN-RNN")][0] - res[(2.0, "LN-RNN")][0]) < 0.05, "LN-RNN ölçekten bağımsız")
        R_.table("200 adımlık RNN (64 birim): son 50 adımda gizli durum", ["W_h ölçeği", "model", "ort. |h|", "doymuş oran (|h| > 0.97)"], rows,
                 note="Düz RNN'de rekürent ağırlık ölçeği her şeyi belirler: küçükse sinyal söner, büyükse tanh doyar (türev ≈ 0, gradyan akmaz). LN her adımda ön-aktivasyonu sıfır ortalamalı, birim varyanslı yapar: dinamik ağırlık ölçeğinden bağımsızlaşır, tanh doğrusal bölgesinde kalır. Géron'un 'kararsız gradyanlar' çözümlerinden biri (diğeri: rekürent dropout, gradyan kırpma).")
    R_.test("LayerNorm RNN", t)
