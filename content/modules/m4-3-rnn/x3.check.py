import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        L0, g0 = need(ns, "loss_grad")(1.5, -0.5, 30, 0.5, 0.0)
        Lp = ns["loss_grad"](1.5 + 1e-6, -0.5, 30, 0.5, 0.0)[0]
        close(g0, (Lp - L0) / 1e-6, rtol=1e-3, what="BPTT gradyanı")
        rows = []
        res = {}
        for T in (5, 10, 20, 40):
            wc = need(ns, "find_wall")(-1.0, T, 0.5, 2.5, 4.0)
            far = abs(ns["loss_grad"](wc - 0.3, -1.0, T, 0.5, 2.0)[1])
            near = max(abs(ns["loss_grad"](wc - dd, -1.0, T, 0.5, 2.0)[1]) for dd in [0.0] + [10.0 ** -k for k in range(2, 15)])
            jump = abs(need(ns, "step")(wc, near, 0.01) - wc)
            jc = abs(ns["step"](wc, near, 0.01, 1.0) - wc)
            res[T] = near
            rows.append([T, f"{wc:.4f}", f"{far:.2e}", f"{near:.2e}", f"{jump:.2e}", f"{jc:.2e}"])
        ensure(res[40] > 100 * res[10] and abs(float(rows[3][1]) - 1.5 / np.tanh(0.5)) < 0.01, "duvarda gradyan T ile patlar; duvar ≈ 1.5/tanh(0.5)")
        R_.table("h_t = w·tanh(h_(t−1)) − 1, h₀ = 0.5, hedef 2", ["T", "duvar w*", "|∂L/∂w| (w* − 0.3)", "duvarda maks. |∂L/∂w|", "SGD adımı (η = 0.01)", "kırpılmış adım (|g| ≤ 1)"], rows,
                 note="w* ≈ 1.5/tanh(0.5) ≈ 3.25'te kararsız sabit nokta h₀'ın üzerinden geçer: altında yörünge negatif çekere, üstünde pozitif çekere gider. Duvara yakın gradyan uzun dizilerde astronomikleşir — tek bir kırpmasız adım parametreyi bütün anlamlı aralığın dışına atar. Kırpma, gradyan yönünü koruyup adımı sınırlar (Pascanu ve ark. Şekil 6).")
    R_.test("uçurum ve kırpma", t)
