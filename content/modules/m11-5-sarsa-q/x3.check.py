import numpy as np
from labcheck import need, ensure


def basic():
    p = np.zeros((2, 10, 10), dtype=int)
    for t in range(12, 22):
        for d in range(1, 11):
            p[0, t - 12, d - 1] = 1 if (t == 12 and d not in (4, 5, 6)) or (13 <= t <= 16 and not 2 <= d <= 6) else 0
            p[1, t - 12, d - 1] = 1 if t <= 17 or (t == 18 and d in (1, 9, 10)) else 0
    return p


def check(ns, R_):
    def t():
        Q, pol = need(ns, "mc_es")(200000, 1)
        ensure(Q.shape == (2, 10, 10, 2) and pol.shape == (2, 10, 10), "şekiller")
        ref = basic()
        agree = (pol == ref).mean()
        ensure(agree > 0.88, f"temel stratejiyle uyum %{100 * agree:.0f}")
        ensure((pol[:, 8:, :] == 0).all(), "20–21'de dur")
        def show(m):
            return ["".join("Ç" if v else "·" for v in row) for row in m[::-1]]
        rows = [[21 - i, a, b] for i, (a, b) in enumerate(zip(show(pol[0]), show(pol[1])))]
        R_.table("Öğrenilen politika (Ç = çek, · = dur), sütunlar krupiye A,2…10", ["toplam", "kullanılabilir as yok", "kullanılabilir as"], rows,
                 note=f"Temel stratejiyle uyum: %{100 * agree:.1f}. Uyumsuzluklar Q değerlerinin neredeyse eşit olduğu hücrelerde (ör. 12 – krupiye 3, yumuşak 18).")
    R_.test("MC keşifçi başlangıçlar", t)
