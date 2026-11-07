import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="conv1d_winograd"))

    def t():
        d, g = [1.0, 2, 3, 4], [1.0, 0, -1]
        close(np.array(need(ns, "winograd_f23")(d, g)), np.array([1 - 3, 2 - 4.0]), what="el örneği")
        rng = np.random.default_rng(0)
        x = rng.normal(size=102)
        g = rng.normal(size=3)
        ref = np.correlate(x, g, mode="valid")
        close(need(ns, "conv1d_winograd")(x, g), ref, atol=1e-12, what="1B geçerli korelasyon")
        R_.note("Çıktı çifti başına 4 çarpma (6 yerine); filtre dönüşümü (g₀+g₁+g₂)/2 vb. bir kez önceden hesaplanır, girdi/çıktı dönüşümleri yalnız toplama. 2B F(2×2, 3×3): 16 çarpma vs 36 (2.25×).")
    R_.test("Winograd", t)
