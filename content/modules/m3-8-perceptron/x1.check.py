import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        X, y = need(ns, "make_rings")(400, 1)
        ensure(X.shape == (400, 2), "şekil")
        L = need(ns, "lift")(X)
        ensure(L.shape == (400, 3), "3B")
        (pa, pm), (sa, sm) = need(ns, "fit_both")(L, y)
        ensure(pa == 1.0 and sa == 1.0, "3B'de ikisi de ayırmalı")
        ensure(sm >= pm - 1e-9, "SVC kenar payı en büyük olmalı")
        R_.table("Kaldırılmış halkalar (x₁, x₂, x₁² + x₂²)", ["model", "doğruluk", "geometrik kenar payı"], [["perceptron", f"{pa:.2f}", f"{pm:.3f}"], ["doğrusal SVC (C = 1000)", f"{sa:.2f}", f"{sm:.3f}"]],
                 note="İkisi de 3B'de bir düzlem bulur; perceptron'unki ilk hatasız düzlemdir, SVC en geniş kenarlıyı arar — yeni verilere daha sağlam. 2B'ye geri izdüşürülen karar sınırı bir çemberdir: x₁² + x₂² = sabit.")
    R_.test("perceptron vs SVC", t)
