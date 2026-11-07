import numpy as np
import torch
from labcheck import close, need, ensure


def check(ns, R_):
    def loc():
        T, r = need(ns, "location")(3, 3)
        close(T.sum(1), np.ones(21), atol=1e-12, what="satırlar olasılık")
        ensure(r[0] == 0 and 29 < r[20] <= 30, f"r(0) = 0, r(20) ≈ 30 bekleniyordu: {r[20]:.2f}")
        T0, _ = ns["location"](3, 3)
        close(T0[0, :4], np.array([np.exp(-3) * 3 ** k / np.prod(range(1, k + 1)) for k in range(4)]), atol=1e-12, what="araba yokken ertesi gün = iadeler")
    R_.test("location", loc)

    def jp():
        V, pi, it = need(ns, "jack_pi")()
        V, pi = np.asarray(V), np.asarray(pi)
        ensure(pi[20, 0] == 5 and pi[0, 20] < 0 and pi[0, 0] == 0, "kitaptaki π*'ın köşeleri: (20, 0) → 5, (0, 20) → negatif, (0, 0) → 0")
        ensure(it <= 6, f"{it} iyileştirme")
        ensure(400 < V[0, 0] < 440 and V[20, 20] > V[0, 0], f"V aralığı: {V.min():.0f}–{V.max():.0f}")
        rows = ["".join(f"{v:+d}" if v else " 0" for v in pi[n1]) for n1 in (20, 15, 10, 5, 0)]
        R_.table("π* (satır: 1. şubedeki araba; sütun: 2. şube 0…20)", ["n₁", "taşınan araba (+: 1 → 2)"], [[n, r] for n, r in zip((20, 15, 10, 5, 0), rows)],
                 note=f"{it} politika iyileştirmesi; V ∈ [{V.min():.0f}, {V.max():.0f}]. Sutton Şekil 4.2'deki bantlı yapı: 2. şubenin talebi yüksek, iadesi düşük olduğu için arabalar oraya taşınır.")
    R_.test("Jack'in araba kiralama problemi", jp)
