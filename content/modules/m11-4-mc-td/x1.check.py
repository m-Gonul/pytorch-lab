import random
import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rng = random.Random(3)
        for _ in range(200):
            st, r = need(ns, "play")(rng)
            ensure(r in (-1, 0, 1) and all(12 <= s[1] <= 21 and 1 <= s[2] <= 10 for s in st), "durum aralıkları")
        V = need(ns, "mc_blackjack")(30000, 1)
        ensure(V.shape == (2, 10, 10) and np.abs(V).max() <= 1, "şekil ve aralık")
        ensure(V[0, 8:, :].mean() > 0.3 and V[0, :8, :].mean() < -0.2, "20–21'de durmak kazandırır, 12–19'da çekmek riskli")
        R_.table("Kullanılabilir as yok, 30 000 el (Sutton Şekil 5.1)", ["oyuncu toplamı", "krupiye A", "krupiye 6", "krupiye 10"],
                 [[t + 12, f"{V[0, t, 0]:+.2f}", f"{V[0, t, 5]:+.2f}", f"{V[0, t, 9]:+.2f}"] for t in (0, 4, 8, 9)])
    R_.test("blackjack MC", t)
