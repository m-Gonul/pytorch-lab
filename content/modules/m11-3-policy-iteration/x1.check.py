import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for S in (5, 20, 80, 200):
            p = need(ns, "pi_iterations")(S, 4, 0.95, 5, 1)
            v = need(ns, "vi_iterations")(S, 4, 0.95, 5, 1)
            rows.append([S, f"{np.mean(p):.1f} (en çok {max(p)})", f"{np.mean(v):.0f}"])
        ensure(all(float(r[1].split()[0]) < 15 for r in rows), "PI birkaç iyileştirmede bitmeli")
        R_.table("Rastgele MDP, 4 eylem, γ = 0.95", ["S", "PI iyileştirme", "VI iterasyon (ε = 1e-8)"], rows,
                 note="Politika sayısı 4^S ile patlasa da PI pratikte durum sayısıyla çok yavaş büyüyen sayıda adım atar.")
    R_.test("ölçeklenme", t)
