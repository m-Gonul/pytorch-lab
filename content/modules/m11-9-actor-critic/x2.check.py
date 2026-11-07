import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        res = {}
        for n in (50, 200, 1000, 5000):
            a = np.mean([need(ns, "bc")(n, 0.5, s) for s in range(5)])
            b = np.mean([need(ns, "pg")(n, 0.5, s) for s in range(5)])
            res[n] = (a, b)
            rows.append([n, f"{a:.3f}", f"{b:.3f}"])
        ensure(res[200][0] > res[200][1] and res[5000][1] > 0.85, "taklit daha hızlı; REINFORCE de sonunda öğrenir")
        R_.table("Doğrusal softmax politika, test doğruluğu (5 tohum)", ["etkileşim", "taklit (uzman etiketi)", "REINFORCE (0/1 ödül)"], rows,
                 note="Uzman etiketi her örnekte doğru eylemi söyler (log K bit); ödül yalnız seçilen eylemin doğru olup olmadığını söyler (≤ 1 bit) ve önce keşif ister. AlphaGo önce uzman oyunlarıyla taklit, sonra kendine karşı REINFORCE kullanır.")
    R_.test("taklit vs politika gradyanı", t)
