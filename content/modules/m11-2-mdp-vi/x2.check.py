import numpy as np
from labcheck import need, ensure


def retail():
    P = np.zeros((2, 4, 4))
    for a, sale in enumerate((0.3, 0.8)):
        P[a, 0, 1] = 1
        for k in range(1, 4):
            P[a, k, k - 1] = sale; P[a, k, k] = 1 - sale
    R = np.array([[-5, -5], [-1, -1], [-1, -1], [-2, -2]], float)
    return P, R, np.array([0, -4, -8, -12.])


def check(ns, R_):
    P, R, VT = retail()
    name = lambda p: "".join("Y" if a == 0 else "İ" for a in p[1:])

    def h():
        pols, k = need(ns, "stationary_policy_horizon")(P, R, VT, 1.0, 15)
        ensure(len(pols) == 15 and name(pols[0]) == "İİİ" and name(pols[1]) == "Yİİ", "h = 1 ve 2'deki politikalar")
        R_.table("Perakende: başlangıç politikası (S1 S2 S3; Y yüksek, İ indirim)", ["h", "politika"], [[i + 1, name(p)] for i, p in enumerate(pols) if i < 6 or i == 14],
                 note=f"h ≥ {k} için politika değişmiyor: uzun ufukta durağan politikaya yaklaşılır.")
    R_.test("ufukla politika", h)

    def g():
        out = need(ns, "policy_vs_gamma")(P, R, [0.1, 0.5, 0.8, 0.95, 0.99])
        ensure(len(out) == 5, "5 politika")
        R_.table("Sonsuz ufuk (terminal yok): γ'ya göre politika", ["γ", "politika (S1 S2 S3)"], [[gg, name(p)] for gg, p in zip([0.1, 0.5, 0.8, 0.95, 0.99], out)],
                 note="Küçük γ: yalnız anlık maliyete bakılır (tüm eylemler aynı maliyette → eşitlikte ilk eylem); büyük γ: stoksuz kalma cezası planlamaya girer.")
    R_.test("γ ile politika", g)
