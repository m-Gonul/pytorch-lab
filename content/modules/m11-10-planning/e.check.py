import random
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time

LN = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]


def term(b):
    for l in LN:
        s = b[l[0]] + b[l[1]] + b[l[2]]
        if abs(s) == 3:
            return s // 3
    return 0 if all(b) else None


MEMO = {}


def perfect(b, p):
    k = (tuple(b), p)
    if k in MEMO:
        return MEMO[k]
    t = term(b)
    if t is not None:
        return t, None
    best = None
    for m in range(9):
        if b[m] == 0:
            b2 = b[:]
            b2[m] = p
            v = perfect(b2, -p)[0]
            if best is None or v * p > best[0] * p:
                best = (v, m)
    MEMO[k] = best
    return best


def loop_playouts(board, player, n, rng):
    tot = 0
    for _ in range(n):
        b, p = board[:], player
        while term(b) is None:
            m = rng.choice([i for i in range(9) if b[i] == 0])
            b[m] = p
            p = -p
        tot += term(b)
    return tot / n


def play(px, po, seed, ns):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    b, p = [0] * 9, 1
    while term(b) is None:
        f = px if p == 1 else po
        if f == "random":
            m = rng.choice([i for i in range(9) if b[i] == 0])
        elif f == "perfect":
            m = perfect(b, p)[1]
        else:
            m = ns["mcts_move"](b, p, 200, 16, g)
        b[m] = p
        p = -p
    return term(b)


def check(ns, R_):
    src = ns["__source__"]
    R_.test("winner döngüsüz, playout yalnız hamle döngüsü", lambda: (max_loops(src, 0, func="winner"), max_loops(src, 1, func="batched_playouts")))

    def w():
        B = torch.tensor([[1, 1, 1, 0, -1, -1, 0, 0, 0], [-1, 1, 0, -1, 1, 0, -1, 0, 1], [1, -1, 1, 1, -1, -1, -1, 1, 1], [0] * 9])
        ensure(need(ns, "winner")(B).tolist() == [1, -1, 0, 0], "kazanan")
    R_.test("winner", w)

    def po():
        g = torch.Generator().manual_seed(0)
        r = need(ns, "batched_playouts")([0] * 9, 1, 100000, g)
        ensure(abs(float((r == 1).double().mean()) - 0.585) < 0.01 and abs(float((r == -1).double().mean()) - 0.288) < 0.01, "boş tahtada X %58.5, O %28.8")
        b = [1, 1, 0, -1, -1, 0, 0, 0, 0]
        close(float(ns["batched_playouts"](b, 1, 20000, g).mean()), loop_playouts(b, 1, 20000, random.Random(0)), atol=0.03, what="döngülü oyunlarla aynı dağılım")
        t1 = bench(lambda: ns["batched_playouts"]([0] * 9, 1, 10000, g), repeat=3, number=1)
        t2 = bench(lambda: loop_playouts([0] * 9, 1, 10000, random.Random(0)), repeat=1, number=1)
        R_.table("10 000 rastgele oyun", ["yöntem", "süre"], [["tensör (paralel)", fmt_time(t1)], ["Python döngüsü", fmt_time(t2)]])
    R_.test("batched_playouts", po)

    def mc():
        ensure(need(ns, "mcts_move")([1, 1, 0, -1, -1, 0, 0, 0, 0], 1, 100, 8, torch.Generator().manual_seed(0)) == 2, "kazanan hamleyi bulmalı")
        ensure(ns["mcts_move"]([1, 1, 0, -1, 0, 0, 0, 0, 0], -1, 200, 8, torch.Generator().manual_seed(0)) == 2, "rakibin kazanmasını engellemeli")
        rows = []
        r1 = [play("mcts", "random", s, ns) for s in range(10)]
        r2 = [play("random", "mcts", s, ns) for s in range(10)]
        r3 = [play("mcts", "perfect", s, ns) for s in range(4)] + [-play("perfect", "mcts", s, ns) for s in range(4)]
        ensure(all(v == 1 for v in r1) and all(v <= 0 for v in r2), "rastgele oyuncuya karşı kaybetmemeli")
        ensure(all(v >= 0 for v in r3), "kusursuz oyuncuya karşı kaybetmemeli (beraberlik)")
        rows.append(["MCTS (X) – rastgele", f"{r1.count(1)}G {r1.count(0)}B {r1.count(-1)}M"])
        rows.append(["rastgele – MCTS (O)", f"{r2.count(-1)}G {r2.count(0)}B {r2.count(1)}M (MCTS bakışından)"])
        rows.append(["MCTS – kusursuz (her iki renk)", f"{r3.count(1)}G {r3.count(0)}B {r3.count(-1)}M"])
        R_.table("UCT, hamle başına 200 benzetim × 16 paralel oyun", ["eşleşme", "sonuç"], rows,
                 note="Tic-tac-toe kusursuz oyunda beraberedir. MCTS hiçbir değerlendirme fonksiyonu bilmeden yalnız rastgele oyunların istatistikleriyle bu seviyeye ulaşır.")
    R_.test("mcts_move", mc)
