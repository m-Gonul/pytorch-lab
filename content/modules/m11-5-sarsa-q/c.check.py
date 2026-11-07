import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız bölüm döngüsü", lambda: max_loops(src, 1, func="max_bias"))

    def p():
        Q = np.array([[1.0, 3.0, 2.0], [0.0, -1.0, 5.0]])
        close(need(ns, "eps_greedy_probs")(Q, 0.3), np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), what="ε-açgözlü olasılıklar")
    R_.test("eps_greedy_probs", p)

    def mb():
        q = need(ns, "max_bias")(10000, 300, seed=1)
        d = ns["max_bias"](10000, 300, seed=1, double=True)
        ensure(q.shape == (300,), "şekil")
        ensure(q[5:30].max() > 0.75, "Q-öğrenme başta solu fazlasıyla seçmeli")
        ensure(d.max() < 0.55 and d[30] < d[5], "çift Q-öğrenme solu tercih etmemeli")
        ensure(d[-50:].mean() < 0.09 and q[-50:].mean() > d[-50:].mean(), "çift Q optimal ε-açgözlü orana (%5) yaklaşmalı")
        R_.table("Sol seçme oranı, 10 000 deney (optimal: %5)", ["bölüm", "Q-öğrenme", "çift Q-öğrenme"],
                 [[e + 1, f"{100 * q[e]:.1f}%", f"{100 * d[e]:.1f}%"] for e in (0, 10, 20, 50, 100, 200, 299)],
                 note="B'deki 10 eylemin gürültülü kestirimlerinden en büyüğü pozitif çıkar; Q-öğrenme bu yanlılığı A'ya taşır (Sutton Şekil 6.5).")
    R_.test("maksimizasyon yanlılığı", mb)
