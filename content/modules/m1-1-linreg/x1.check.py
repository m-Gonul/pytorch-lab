import numpy as np
from labcheck import close, max_loops, need, ensure, uses_any

FUNCS = ["phi", "fit_poly", "rmse", "resid", "quad_sig"]


def check(ns, R):
    src = ns["__source__"]
    for f in FUNCS:
        R.test(f"{f} içinde döngü yok", lambda f=f: max_loops(src, 0, func=f))
    R.test("torch kullanılmıyor, inv yok", lambda: (need(ns, "phi"),
           ensure(not uses_any(src, {"torch", "inv", "polyfit"}), "lstsq kullan; inv/polyfit yasak")))

    phi = need(ns, "phi")
    fit_poly = need(ns, "fit_poly")
    rmse = need(ns, "rmse")
    resid = need(ns, "resid")
    quad_sig = need(ns, "quad_sig")

    def phi_ok():
        close(phi(np.array([1.0, 2.0, 3.0]), 2),
              [[1, 1, 1], [1, 2, 4], [1, 3, 9]], rtol=0, atol=1e-12, what="phi([1,2,3], 2)")
        close(phi(np.array([2.0, -1.0]), 0), [[1], [1]], rtol=0, atol=1e-12, what="deg=0: birler sütunu")
        rng = np.random.default_rng(3)
        v = np.linspace(-3, 3, 17)
        close(phi(v, 4), np.vander(v, 5, increasing=True), rtol=1e-12, atol=1e-12, what="np.vander ile")
        ensure(phi(np.array([1.0]), 2).shape == (1, 3), "shape (n, deg+1) olmalı")
    R.test("phi: elle örnek, np.vander ve shape", phi_ok)

    def fit_ok():
        v = np.linspace(-2, 2, 25)
        close(fit_poly(v, 3 * v + 0.8 * v**2 - 1, 2), [-1, 3, 0.8], rtol=0, atol=1e-9,
              what="gürültüsüz karesel: [−1, 3, 0.8]")
        close(fit_poly(v, 1.5 * v - 1, 1), [-1, 1.5], rtol=0, atol=1e-9, what="gürültüsüz doğrusal")
        t = fit_poly(v, 1.5 * v - 1, 4)
        close(t, [-1, 1.5, 0, 0, 0], rtol=0, atol=1e-7, what="aşırı derece: gereksiz katsayılar ~0")
        return "derece artsa da gereksiz katsayılar ~0: lstsq en küçük normlu çözümü seçer"
    R.test("fit_poly: katsayıları geri kazandırır", fit_ok)

    def rmse_ok():
        ensure(isinstance(rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])), float), "float döndürmeli")
        close(rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])), 0, rtol=0, atol=1e-12, what="mükemmel uydurma")
        close(rmse(np.array([1.0, 2.0]), np.array([2.0, 1.0])), 1.0, rtol=0, atol=1e-12, what="[1,2] vs [2,1]")
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=13), rng.normal(size=13)
        close(rmse(a, b), np.sqrt(np.mean((b - a) ** 2)), rtol=1e-12, atol=1e-12, what="rastgele")
    R.test("rmse: tanımla birebir", rmse_ok)

    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 40)
    y = 3 * x + 0.8 * x**2 - 1 + 0.4 * rng.normal(size=40)

    def resid_ok():
        v = np.array([1.0, 2.0, 3.0])
        r = resid(v, np.array([2.0, 4.0, 5.0]), 1)
        close(r, [1 / 6, -1 / 3, 1 / 6], rtol=0, atol=1e-9, what="r = ŷ − y (l4 işareti)")
    R.test("resid: r = ŷ − y işareti", resid_ok)

    def sig_ok():
        v = np.linspace(-2, 2, 31)
        close(quad_sig(v, 0.8 * (v**2 - (v**2).mean())), 0.8, rtol=0, atol=1e-12,
              what="r = 0.8(x²−x̄²) ise imza 0.8")
        close(quad_sig(v, 0.8 * (v**2 - (v**2).mean()) + 2.0), 0.8, rtol=0, atol=1e-12,
              what="sabit ekleme imzayı değiştirmemeli (merkezleme işi görür)")
        close(quad_sig(v, 3.0 * v), 0.0, rtol=0, atol=1e-12, what="doğrusal artıkta imza 0")
    R.test("quad_sig: merkezlenmiş x²'ye uydurur", sig_ok)

    def story():
        t1, t2 = fit_poly(x, y, 1), fit_poly(x, y, 2)
        r1 = phi(x, 1) @ t1 - y
        s1, s2 = quad_sig(x, r1), quad_sig(x, resid(x, y, 2))
        m1 = rmse(y, phi(x, 1) @ t1)
        m2 = rmse(y, phi(x, 2) @ t2)
        ensure(s1 < -0.5, f"doğrusal uydurmanın U imzası {s1:.3f}: −0.8'e yakın negatif bekleniyordu")
        ensure(abs(s2) < 0.02, f"derece 2'de imza {s2:.4f} ≈ 0 olmalıydı")
        ensure(abs(t2[2] - 0.8) < 0.15, f"x² katsayısı {t2[2]:.3f}: 0.8'e yakın bekleniyordu")
        ensure(abs(t2[1] - 3) < 0.25 and abs(t2[0] + 1) < 0.4, f"katsayılar {t2}: ≈ [−1, 3, 0.8] bekleniyordu")
        ensure(m2 < m1 / 3, f"RMSE düşüşü yetersiz: {m1:.3f} → {m2:.3f} (~4 kat bekleniyordu)")
        r2lin = 1 - m1**2 / np.var(y)
        ensure(r2lin > 0.9, f"doğrusal R² {r2lin:.3f} — yanlış biçime rağmen ~0.9 kalmalıydı")
        return f"imza {s1:.2f} → {s2:+.1e}; RMSE {m1:.3f} → {m2:.3f}; R²(1) ≈ {r2lin:.2f} ama biçim yanlıştı"
    R.test("U desenini kodla: imza ve RMSE hikâyesi", story)
