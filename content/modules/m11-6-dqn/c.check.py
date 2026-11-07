import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız yineleme döngüsü", lambda: (max_loops(src, 0, func="baird_features"), max_loops(src, 1, func="expected_td"), max_loops(src, 1, func="expected_tdc")))
    w0 = [1, 1, 1, 1, 1, 1, 10, 1]

    def feats():
        X = need(ns, "baird_features")()
        close(X @ np.array(w0, float), np.array([3, 3, 3, 3, 3, 3, 12.0]), what="v̂ = Xw")
        ensure(np.linalg.matrix_rank(X) == 7, "öznitelikler bağımsız")
    R_.test("baird_features", feats)

    def div():
        mu = np.full(7, 1 / 7)
        _, h = need(ns, "expected_td")(w0, 0.01, 0.99, 1000, mu)
        w1, _ = ns["expected_td"](w0, 0.01, 0.99, 1, mu)
        X = ns["baird_features"]()
        v = X @ np.array(w0, float)
        close(w1, np.array(w0, float) + 0.01 * (mu * (0.99 * v[6] - v)) @ X, what="bir beklenen güncelleme")
        _, ho = ns["expected_td"](w0, 0.01, 0.99, 1000, np.eye(7)[6])
        wt, htc = need(ns, "expected_tdc")(w0, 0.005, 0.05, 0.99, 1000, mu)
        ensure(h[-1] > 10 * h[0], "düzgün dağılımda TD ıraksamalı")
        ensure(ho[-1] <= ho[0], "politika üstü dağılımda kararlı")
        ensure(htc[-1] < htc[0] and np.isfinite(htc).all(), "TDC kararlı")
        vt = X @ wt
        dl = 0.99 * vt[6] - vt
        v0 = X @ np.array(w0, float)
        r = (mu @ dl ** 2) / (mu @ (0.99 * v0[6] - v0) ** 2)
        pbe = np.abs(X.T @ (mu * dl)).max()
        ensure(r < 1e-3, f"TDC: ortalama kare TD hatası 1000 kat düşmeli (oran {r:.1e})")
        R_.table("‖w‖, γ = 0.99", ["yineleme", "yarı-gradyan TD (μ düzgün)", "yarı-gradyan TD (μ = politika üstü)", "TDC (μ düzgün)"],
                 [[k, f"{h[k]:.2f}", f"{ho[k]:.2f}", f"{htc[k]:.2f}"] for k in (0, 100, 300, 600, 1000)],
                 note=f"TDC: ortalama kare TD hatası {r:.1e} katına iner, ‖Xᵀμδ‖∞ = {pbe:.1e}; ama v̂ ≈ {vt[0]:.2f} ≠ 0: TD hatasını hızla küçültür, gerçek değere (0) ise çok yavaş kayar (Sutton Şekil 11.2 ve 11.6).")
    R_.test("ıraksama ve TDC", div)
