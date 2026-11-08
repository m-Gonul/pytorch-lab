import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = np.random.default_rng(0)
        x = rng.normal(size=60)
        L = lambda w: 0.5 * (need(ns, "forward")(w, 1.0, x)[-1] - 0.3) ** 2
        g = need(ns, "grad_w")(0.9, 1.0, x, 0.3)
        close(g, (L(0.9 + 1e-7) - L(0.9 - 1e-7)) / 2e-7, rtol=1e-5, what="tam BPTT = sayısal")
        rows = []
        for w in (0.5, 0.9, 0.98):
            full = ns["grad_w"](w, 1.0, x, 0.3)
            r = [w]
            for k in (1, 5, 20, 50):
                r.append(f"{abs(ns['grad_w'](w, 1.0, x, 0.3, k) - full) / abs(full):.3f}")
            rows.append(r)
        ensure(float(rows[0][3]) < 1e-4 and float(rows[2][3]) > 0.05, "w küçükse kısa pencere yeter, w ≈ 1'de yetmez")
        R_.table("Skaler doğrusal RNN, T = 60: kesilmiş BPTT gradyanının göreli hatası", ["w", "k = 1", "k = 5", "k = 20", "k = 50"], rows,
                 note="t adım önceki katkı w^t ile söner: |w| < 1 ve küçükse birkaç adım yeter, |w| → 1 (uzun bellek) olduğunda kesme yanlılığı büyür — model, pencereden uzun bağımlılıkları öğrenemez. Pratikte (Williams & Peng 1990) uzun diziler parçalara bölünür, gizli durum parçalar arasında taşınır (gradyan taşınmaz): ileri yön tam bellek, geri yön k adım.")
    R_.test("kesilmiş BPTT", t)
