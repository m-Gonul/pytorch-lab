import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "sklearn"}))
    P = [(0, 0), (1, 0), (4, 4), (5, 4)]

    def hand():
        C, lab, J = need(ns, "kmeans")(P, [(0.0, 0.0), (1.0, 0.0)], 2)
        close([list(c) for c in C], [[0.5, 0.0], [4.5, 4.0]]); ensure(list(lab) == [0, 0, 1, 1], f"etiketler {lab}"); close(J, 1.0)
    R.test("elle örnek: merkezler, etiketler, inertia 1.0", hand)

    def tie():
        ensure(list(need(ns, "assign")([(1, 0)], [(0, 0), (2, 0)])) == [0], "eşitlikte küçük indeks")
        new = need(ns, "update")([(0, 0), (1, 1)], [0, 0], [(5, 5), (9, 9)])
        close([list(c) for c in new], [[0.5, 0.5], [9, 9]], what="boş küme eski merkezini korur")
    R.test("eşitlik ve boş küme kuralları", tie)

    def mono():
        rng = random.Random(0)
        pts = [(rng.gauss(cx, 0.5), rng.gauss(cy, 0.5)) for cx, cy in [(0, 0), (5, 5), (0, 5)] for _ in range(20)]
        C0 = [pts[0], pts[1], pts[2]]
        Js = [need(ns, "kmeans")(pts, C0, it)[2] for it in range(6)]
        ensure(all(a >= b - 1e-9 for a, b in zip(Js, Js[1:])), f"inertia arttı: {Js}")
        return "inertia (0–5 iterasyon): " + ", ".join(f"{v:.1f}" for v in Js)
    R.test("inertia iterasyonlarla artmıyor", mono)
