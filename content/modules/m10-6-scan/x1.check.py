import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız seviye döngüsü", lambda: max_loops(src, 1, func="segmented_scan"))

    def t():
        close(need(ns, "segmented_scan")(np.array([1, 2, 3, 4, 5]), np.array([1, 0, 1, 0, 0])), np.array([1, 3, 3, 7, 12.0]), what="el örneği")
        rng = np.random.default_rng(0)
        x = rng.integers(0, 10, 2000)
        f = (rng.random(2000) < 0.05).astype(int)
        f[0] = 1
        ref = np.zeros(2000)
        acc = 0
        for i in range(2000):
            acc = x[i] if f[i] else acc + x[i]
            ref[i] = acc
        close(ns["segmented_scan"](x, f), ref, what="rastgele segmentler")
        R_.note("İşlecin birleşmeliliği: ((f₁,v₁)⊕(f₂,v₂))⊕(f₃,v₃) = (f₁,v₁)⊕((f₂,v₂)⊕(f₃,v₃)) — 'bayrak varsa sol tarafı unut'. Segmentli tarama, seyrek matris-vektör çarpımı (CSR satırları) ve değişken uzunluklu dizilerin toplu işlenmesinin temelidir.")
    R_.test("segmentli tarama", t)
