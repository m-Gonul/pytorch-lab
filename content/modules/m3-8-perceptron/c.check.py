import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 0, func="relu_features"), max_loops(src, 0, func="relu_interpolant")))

    def interp():
        rng = np.random.default_rng(0)
        k = np.sort(rng.uniform(-1, 1, 9))
        v = rng.normal(size=9)
        a, c0, c1 = need(ns, "relu_interpolant")(k, v)
        ensure(len(a) == 7, "k − 2 iç kırılma → 7 ReLU (uçlardaki iki düğüm doğrusal terimde)")
        x = np.linspace(k[0], k[-1], 400)
        f = c0 + c1 * x + (a[None, :] * np.maximum(0, x[:, None] - k[None, 1:-1])).sum(1)
        close(f, np.interp(x, k, v), atol=1e-9, what="parçalı doğrusal fonksiyon = ReLU toplamı")
    R_.test("ReLU ile parçalı doğrusal inşa", interp)

    def dd():
        F = np.random.default_rng(1).normal(size=(5, 12))
        y = np.arange(5.0)
        w = need(ns, "min_norm_fit")(F, y)
        close(F @ w, y, atol=1e-9, what="enterpolasyon")
        close(w, F.T @ np.linalg.solve(F @ F.T, y), atol=1e-9, what="en küçük norm = Fᵀ(FFᵀ)⁻¹y")
        res = need(ns, "double_descent")(trials=10)
        ps = sorted(res)
        peak = max(res, key=res.get)
        ensure(90 <= peak <= 110, f"tepe p ≈ n = 100'de olmalı (bulunan {peak})")
        ensure(res[3000] < res[100] / 10 and res[3000] < res[75], "p ≫ n'de hata yeniden düşer")
        R_.table("Rastgele ReLU öznitelikleri, n = 100, d = 10, en küçük normlu çözüm (10 deneme ortancası)", ["p (öznitelik)", "test MSE"], [[p, f"{res[p]:.3f}"] for p in ps],
                 note="Klasik bölge (p < n): önce iyileşme, sonra aşırı uyum. Enterpolasyon eşiği (p = n): tek bir çözüm, gürültüye tam uyum, devasa norm. Aşırı parametreli bölge (p > n): sonsuz çözüm, en küçük normlu olan pürüzsüz — derin ağların neden aşırı parametreleyince bozulmadığının en basit modeli.")
    R_.test("çift iniş", dd)
