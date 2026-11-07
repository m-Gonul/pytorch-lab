import random
import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def ep():
        e = need(ns, "walk_episode")(random.Random(1))
        ensure(e[0][0] == 3 and e[-1][2] in (0, 6) and all(abs(a - c) == 1 for a, _, c in e), "geçişler")
        ensure(all(r == (1.0 if s2 == 6 else 0.0) for _, r, s2 in e), "ödüller")
    R_.test("walk_episode", ep)

    def cmp():
        runs = 100
        curves = {}
        for name, fn, a in [("MC α = 0.02", "mc", 0.02), ("MC α = 0.04", "mc", 0.04), ("TD α = 0.05", "td0", 0.05), ("TD α = 0.1", "td0", 0.1)]:
            acc = np.zeros(100)
            for k in range(runs):
                rng = random.Random(k)
                eps = [ns["walk_episode"](rng) for _ in range(100)]
                _, h = need(ns, fn)(eps, a)
                acc += np.array(h)
            curves[name] = acc / runs
        ensure(curves["TD α = 0.05"][-1] < curves["MC α = 0.02"][-1], "100 bölümde TD MC'den düşük hata vermeli")
        R_.table("RMS hata, 100 deney ortalaması (Sutton Örnek 6.2)", ["yöntem", "10. bölüm", "50. bölüm", "100. bölüm"],
                 [[k, f"{v[9]:.3f}", f"{v[49]:.3f}", f"{v[99]:.3f}"] for k, v in curves.items()],
                 note="TD bu Markov görevde MC'den hızlı öğrenir; büyük α'larda iki yöntemde de sabit bir gürültü tabanı kalır.")
    R_.test("MC ve TD öğrenme eğrileri", cmp)

    def one():
        eps = [[(3, 0.0, 4), (4, 1.0, 6)]]
        V, _ = ns["td0"](eps, 0.5)
        close(np.array(V[3:5]), np.array([0.5 + 0.5 * (0 + 0.5 - 0.5), 0.5 + 0.5 * (1 - 0.5)]), what="TD adımları")
        V, _ = ns["mc"](eps, 0.5)
        close(np.array(V[3:5]), np.array([0.75, 0.75]), what="MC adımları")
    R_.test("tek bölümde güncellemeler", one)
