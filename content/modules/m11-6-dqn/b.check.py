import random
import numpy as np
from labcheck import close, no_imports, need, ensure


def true_values(n=1000, jump=100):
    s = np.arange(1, n + 1)[:, None]
    k = np.arange(1, jump + 1)[None, :]
    P = np.zeros((n, n))
    r = np.zeros(n)
    p = 1 / (2 * jump)
    for sgn in (1, -1):
        s2 = s + sgn * k
        inside = (s2 >= 1) & (s2 <= n)
        rows, cols = np.nonzero(inside)
        np.add.at(P, (rows, s2[rows, cols] - 1), p)
        r += p * ((s2 > n).sum(1) - (s2 < 1).sum(1))
    return np.linalg.solve(np.eye(n) - P, r)


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def ep():
        e = need(ns, "walk_episode")(random.Random(2))
        ensure(e[0][0] == 500 and e[-1][2] is None and e[-1][1] in (-1.0, 1.0), "başlangıç ve son")
        ensure(all(1 <= abs(c - a) <= 100 for a, _, c in e[:-1]), "sıçrama 1–100")
        ensure(need(ns, "group")(1) == 0 and ns["group"](100) == 0 and ns["group"](101) == 1 and ns["group"](1000) == 9, "gruplar")
    R_.test("ortam ve gruplar", ep)

    def fa():
        V = true_values()
        rng = random.Random(0)
        eps = [ns["walk_episode"](rng) for _ in range(5000)]
        wm = need(ns, "gradient_mc")(eps, 2e-4)
        wt = need(ns, "semi_gradient_td")(eps, 2e-3)
        mu = np.zeros(1000)
        for e in eps:
            for s, _, _ in e:
                mu[s - 1] += 1
        mu /= mu.sum()
        g = (np.arange(1000)) // 100
        vm, vt = np.array(wm)[g], np.array(wt)[g]
        em, et = np.sqrt(mu @ (vm - V) ** 2), np.sqrt(mu @ (vt - V) ** 2)
        ensure(em < 0.1 and et < 0.15, f"değer hatası: MC {em:.3f}, TD {et:.3f}")
        ensure(abs(wt[0]) < abs(wm[0]), "TD uçlarda sıfıra doğru yanlı olmalı")
        R_.table("Durum toplama (10 grup), 5000 bölüm", ["grup", "gerçek ort. değer", "gradyan MC", "yarı-gradyan TD"],
                 [[i, f"{V[100 * i:100 * i + 100].mean():+.3f}", f"{wm[i]:+.3f}", f"{wt[i]:+.3f}"] for i in range(10)],
                 note=f"μ ağırlıklı √VE: MC {em:.3f}, TD {et:.3f}. MC VE'nin yerel en küçüğüne yakınsar; yarı-gradyan TD, TD sabit noktasına — ve VE ≤ (1/(1−γ))·min VE sınırı γ = 1 için anlamsızdır: uçlarda belirgin yanlılık (Sutton Şekil 9.1–9.2).")
    R_.test("gradyan MC ve yarı-gradyan TD", fa)
