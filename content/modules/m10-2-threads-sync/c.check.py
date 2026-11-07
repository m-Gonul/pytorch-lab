import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: (max_loops(src, 0, func="lost_update_prob"), max_loops(src, 0, func="expected_lost")))

    def t():
        rows = []
        for L in (0.001, 0.01, 0.05, 0.2):
            e = need(ns, "lost_update_prob")(L, 1.0, 400000, 1)
            a = 1 - (1 - L) ** 2
            close(e, a, atol=0.003, what=f"L = {L}")
            rows.append([L, f"{e:.4f}", f"{a:.4f}"])
        close(need(ns, "expected_lost")(2, 0.1, 1.0, 400000, 2), 1 - 0.81, atol=0.003, what="k = 2 ile tutarlı")
        k_rows = [[k, f"{ns['expected_lost'](k, 0.001, 1.0, 20000, 3):.3f}"] for k in (2, 8, 32, 128)]
        ensure(float(k_rows[-1][1]) > float(k_rows[0][1]) * 20, "kayıp thread sayısıyla süperdoğrusal")
        R_.table("İki thread, kritik pencere L (çalışma süresi T = 1)", ["L", "deney", "1 − (1 − L/T)²"], rows,
                 note="Pencere küçükse yarış nadirdir — testlerden kaçar ama milyonlarca işlemde kesin olur. Aşağıda: k thread, L = 0.001: " + ", ".join(f"k = {a}: {b}" for a, b in k_rows) + " kayıp/çalıştırma (~k²L).")
    R_.test("yarış olasılığı", t)
