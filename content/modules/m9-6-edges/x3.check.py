import numpy as np
from labcheck import close, need, ensure


def check(ns, R):
    def size():
        for s, n in [(0.5, 3), (1.0, 7), (1.4, 9), (2.0, 13), (2.5, 15)]:
            ensure(need(ns, "gaussian_size")(s) == n, f"σ = {s}: {n} bekleniyordu")
    R.test("gaussian_size", size)

    def kern():
        rows = []
        for s in (1.0, 2.0, 3.0):
            a = need(ns, "log_formula")(s); b = need(ns, "log_via_laplacian")(s)
            n = ns["gaussian_size"](s)
            ensure(a.shape == (n, n) and b.shape == (n + 2, n + 2), "boyutlar")
            ensure(abs(a.sum()) < 1e-9 and abs(b.sum()) < 1e-12, "toplam sıfır")
            c = float(np.corrcoef(a.ravel(), b[1:-1, 1:-1].ravel())[0, 1])
            ensure(c > 0.97, f"σ = {s}: iki çekirdeğin korelasyonu {c:.3f}")
            mid = a[n // 2]
            zc = np.nonzero(np.diff(np.sign(mid[n // 2:])))[0]
            rows.append([s, n, f"{c:.4f}", f"{zc[0] + 0.5 if len(zc) else '—'}", f"{np.sqrt(2) * s:.2f}"])
        R.table("LoG çekirdekleri", ["σ", "n", "korelasyon", "ilk işaret değişimi (piksel)", "√2σ"], rows)
    R.test("iki LoG yolu aynı şekli verir", kern)

    def noise():
        rows = []
        for v in (0.1, 1.0, 10.0):
            d1, d2 = need(ns, "derivative_noise")(v)
            s = np.sqrt(v)
            ensure(abs(d1 / (np.sqrt(2) * s) - 1) < 0.25 and abs(d2 / (np.sqrt(6) * s) - 1) < 0.25, f"σ² = {v}: türev gürültüsü √2σ ve √6σ civarında olmalı")
            e1, e2 = ns["derivative_noise"](v, smooth=2.0)
            ensure(e1 < d1 / 3 and e2 < d2 / 5, "yumuşatma türev gürültüsünü azaltmalı")
            rows.append([v, f"{s:.3f}", f"{d1:.3f}", f"{d2:.3f}", f"{e1:.4f}", f"{e2:.4f}"])
        R.table("Düz bölgede türev gürültüsü (rampa yüksekliği 200, eğim 3.1/piksel)", ["σ²", "σ", "1. fark", "2. fark", "1. fark (Gauss σ=2)", "2. fark (Gauss σ=2)"], rows,
                note="İkinci farkın gürültüsü √6σ: σ² = 10'da bile 7.7, yani rampanın kendi ikinci türevini (yalnız iki uçta ≈ 3) gömer.")
    R.test("gürültü türevleri", noise)
