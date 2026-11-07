from labcheck import close, no_imports, need, ensure

LN = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
MEMO = {}


def term(b):
    for l in LN:
        s = b[l[0]] + b[l[1]] + b[l[2]]
        if abs(s) == 3:
            return s // 3
    return 0 if all(b) else None


def perfect(b, p):
    k = (tuple(b), p)
    if k in MEMO:
        return MEMO[k]
    t = term(b)
    if t is not None:
        return t, []
    vals = {}
    for m in range(9):
        if b[m] == 0:
            b2 = b[:]
            b2[m] = p
            vals[m] = perfect(b2, -p)[0]
    best = max(v * p for v in vals.values()) * p
    MEMO[k] = (best, [m for m, v in vals.items() if v == best])
    return MEMO[k]


def oracle(board, player):
    return [1 / 9] * 9, float(perfect(list(board), player)[0] * player)


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def vp():
        close(need(ns, "visit_policy")([10, 30, 60], 0.5)[2], 3600 / 4600, what="τ = 0.5")
        ensure(ns["visit_policy"]([3, 7, 7], 0) == [0.0, 1.0, 0.0], "τ = 0: argmax")
        ensure(need(ns, "terminal")([1, 1, 1, 0, -1, -1, 0, 0, 0]) == 1 and ns["terminal"]([0] * 9) is None, "terminal")
    R_.test("visit_policy ve terminal", vp)

    def search():
        u = lambda b, p: ([1 / 9] * 9, 0.0)
        N = need(ns, "puct_search")([1, 1, 0, -1, -1, 0, 0, 0, 0], 1, u, 60)
        ensure(max(range(9), key=lambda i: N[i]) == 2, "anında kazanç bulunmalı")
        N = ns["puct_search"]([1, 1, 0, -1, 0, 0, 0, 0, 0], -1, u, 200)
        ensure(max(range(9), key=lambda i: N[i]) == 2, "rakibin kazancı engellenmeli")
        import random
        rng = random.Random(0)
        rows, ok = [], 0
        positions = []
        while len(positions) < 30:
            b, p = [0] * 9, 1
            for _ in range(rng.randint(0, 5)):
                if term(b) is not None:
                    break
                m = rng.choice([i for i in range(9) if b[i] == 0])
                b[m] = p
                p = -p
            if term(b) is None:
                positions.append((b, p))
        okz = 0
        for b, p in positions:
            Nn = ns["puct_search"](b, p, oracle, 100)
            ok += max(range(9), key=lambda i: Nn[i]) in perfect(b, p)[1]
            Nz = ns["puct_search"](b, p, u, 100)
            okz += max(range(9), key=lambda i: Nz[i]) in perfect(b, p)[1]
        ensure(ok >= 29, f"kusursuz değerle arama optimal hamleyi seçmeli ({ok}/30)")
        R_.table("30 rastgele konum, 100 benzetim", ["ağ", "optimal hamle seçimi"], [["önsel düzgün, değer = minimax (kusursuz)", f"{ok}/30"], ["önsel düzgün, değer = 0 (bilgisiz)", f"{okz}/30"]],
                 note="Bilgisiz değerle PUCT yalnız terminal düğümlerden bilgi alır: sığ taktikleri bulur, derin planı bulamaz. AlphaZero'da değer başlığının işi bu boşluğu doldurmaktır.")
    R_.test("puct_search", search)
