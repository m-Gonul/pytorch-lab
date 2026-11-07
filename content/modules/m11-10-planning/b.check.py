from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def env():
        f = need(ns, "maze_step")
        ensure(f((2, 0), 2) == ((2, 0), 0.0, False) and f((2, 1), 3) == ((2, 1), 0.0, False), "kenar ve duvar")
        ensure(f((1, 8), 0) == ((0, 8), 1.0, True), "hedef")
    R_.test("maze_step", env)

    def dq():
        rows, res = [], {}
        for n in (0, 5, 50):
            S = [need(ns, "dyna_q")(50, n, s)[1] for s in range(10)]
            res[n] = [sum(x[k] for x in S) / 10 for k in range(50)]
            rows.append([n] + [f"{res[n][k]:.0f}" for k in (0, 1, 2, 5, 10, 49)])
        ensure(res[50][1] < res[5][1] < res[0][1], "2. bölümde planlama adımı çoğaldıkça daha kısa")
        ensure(res[50][5] < 25 and res[0][5] > 50, "n = 50 birkaç bölümde optimal yola yaklaşmalı")
        R_.table("Dyna labirenti, bölüm başına adım (10 deney; en kısa yol 14)", ["n", "1", "2", "3", "6", "11", "50. bölüm"], rows,
                 note="İlk bölüm planlama olmadan da aynı uzunluktadır (ödül görülmemiştir). Ödül bir kez görülünce n = 50 değer bilgisini model üzerinden geriye hızla yayar (Sutton Şekil 8.2–8.3).")
    R_.test("Dyna-Q", dq)
