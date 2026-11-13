import numpy as np
from labcheck import close, need, ensure, no_imports


def ref_enet_cd(X, y, lam1, lam2, iters):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    cs = [sum(X[i][j] ** 2 for i in range(n)) / n for j in range(d)]
    for _ in range(iters):
        for j in range(d):
            rho = sum(X[i][j] * (y[i] - sum(X[i][k] * w[k] for k in range(d)) + X[i][j] * w[j]) for i in range(n)) / n
            s = rho - lam1 if rho > lam1 else (rho + lam1 if rho < -lam1 else 0.0)
            w[j] = s / (cs[j] + lam2)
    return w


rng = np.random.default_rng(0)
n, d = 40, 6
x1 = rng.normal(size=n)
x2 = x1 + 1e-8 * rng.normal(size=n)  # özdeş ikili (neredeyse)
X = np.column_stack([x1, x2, rng.normal(size=(n, 4))])
w_true = np.array([1.5, 1.5, -2.0, 0.0, 0.0, 0.0])
y = X @ w_true + 0.3 * rng.normal(size=n)
lmax = float(np.abs(X.T @ y / n).max())  # ≈ 2.27


def check(ns, R):
    src = ns["__source__"]
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    soft = need(ns, "soft")
    enet_cd = need(ns, "enet_cd")
    enet_path = need(ns, "enet_path")
    grouping_gap = need(ns, "grouping_gap")

    def soft_ok():
        close([soft(1.5, 0.5), soft(-0.3, 0.5), soft(-2.0, 0.5), soft(0.5, 0.5)], [1, 0, -1.5, 0], rtol=0, atol=1e-12)
    R.test("soft: üç durum, |z| ≤ t → tam 0", soft_ok)

    def cd_ok():
        r = np.random.default_rng(7)
        Xr, yr = r.normal(size=(25, 4)), r.normal(size=25)
        w = enet_cd(Xr, yr, 0.2, 0.5, iters=100)
        close(w, ref_enet_cd(Xr.tolist(), yr.tolist(), 0.2, 0.5, 100), rtol=1e-6, atol=1e-10, what="referans CD")
        ensure(tuple(w.shape) == (4,), f"shape {w.shape} — (d,) bekleniyordu")
        return "katsayılar: " + ", ".join(f"{v:.3f}" for v in w)
    R.test("enet_cd: referans koordinat inişiyle aynı", cd_ok)

    def zeros_ok():
        w = enet_cd(X, y, lmax + 1.0, 0.5, iters=50)
        ensure(all(v == 0.0 for v in w), f"λ ≥ λ_max'ta tam sıfır bekleniyordu, geldi: {np.round(w, 6)}")
        r = np.random.default_rng(11)
        Xg, yg = r.normal(size=(30, 4)), r.normal(size=30)  # iyi koşullu: λ = 0'da OLS'ye yakınsamalı
        close(enet_cd(Xg, yg, 0.0, 0.0, iters=300), np.linalg.lstsq(Xg, yg, rcond=None)[0],
              rtol=1e-4, atol=1e-8, what="λ = 0: OLS")
        return "λ ≥ λ_max → hepsi tam 0; λ = 0 → OLS (iyi koşullu veride)"
    R.test("enet_cd: uç λ'larda davranış", zeros_ok)

    def group_ok():
        w = enet_cd(X, y, 0.1, 0.5, iters=200)
        g = grouping_gap(w)
        ensure(g < 1e-6, f"özdeş ikilinin farkı {g:.2e}: λ₂ > 0 iken eşit paylaştırılmalı (gruplama etkisi)")
        ensure(np.abs(w[:3]).min() > 0.5, f"etkili katsayılar çok küçük kaldı: {np.round(w, 3)}")
        ensure(np.abs(w[3:]).max() < 0.2, f"gerçekten 0 olan katsayılar girdi: {np.round(w, 3)}")
        wL = enet_cd(X, y, 0.1, 0.0, iters=400)
        gL = grouping_gap(wL)
        ensure(min(abs(wL[0]), abs(wL[1])) == 0.0, f"saf lasso ikiliden birini tam 0 yapmalı, geldi: {np.round(wL, 4)}")
        ensure(max(abs(wL[0]), abs(wL[1])) > 2.0, f"saf lasso seçtiği ikiliye tüm ağırlığı vermeli: {np.round(wL, 4)}")
        return f"λ₂ = 0.5: gap {g:.1e} (eşit paylaştırma) — λ₂ = 0: gap {gL:.2f} (tek seçim)"
    R.test("gruplama etkisi: elastic net paylaştırır, lasso seçer", group_ok)

    def path_ok():
        lams = np.logspace(0.5, -3, 31)
        W = enet_path(X, y, lams, lam2=0.5)
        ensure(W.shape == (31, 6), f"shape {W.shape} — (31, 6) bekleniyordu")
        ensure(all(v == 0.0 for v in W[0]), "ilk λ (3.16 ≥ λ_max) satırı tam 0 olmalı")
        nrm = np.linalg.norm(W, axis=1)
        ensure(np.all(np.diff(nrm) >= -1e-9), "λ küçüldükçe ‖w‖ artmalı (monoton yol)")
        sıcak = enet_path(X, y, lams, lam2=0.5)[-1]
        close(sıcak, ref_enet_cd(X.tolist(), y.tolist(), lams[-1], 0.5, 400), rtol=1e-4, atol=1e-8,
              what="yolun sonu: uzun CD ile aynı")
        return f"‖w‖: {nrm[0]:.3f} → {nrm[-1]:.3f}, λ = {lams[-1]:.3g}'de {int((np.abs(W[-1]) > 1e-8).sum())} katsayı sıfırdan farklı"
    R.test("enet_path: şekil, tam sıfır başlangıç, monotonluk", path_ok)
