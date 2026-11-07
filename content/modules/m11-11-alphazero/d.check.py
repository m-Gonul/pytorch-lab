import random
import numpy as np
import torch
from labcheck import close, need, ensure

MEMO = {}


def perfect(b, p, term):
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
            vals[m] = perfect(b2, -p, term)[0]
    best = max(v * p for v in vals.values()) * p
    MEMO[k] = (best, [m for m, v in vals.items() if v == best])
    return MEMO[k]


def states(term):
    out, stack = set(), [([0] * 9, 1)]
    while stack:
        b, p = stack.pop()
        if (tuple(b), p) in out or term(b) is not None:
            continue
        out.add((tuple(b), p))
        for m in range(9):
            if b[m] == 0:
                b2 = b[:]
                b2[m] = p
                stack.append((b2, -p))
    return out


def policy_acc(net, ns, S):
    ok = 0
    for b, p in S:
        with torch.no_grad():
            lo = ns["Net"].forward(net, ns["encode"](list(b), p)[None])[0][0].numpy()
        lo[np.array(b) != 0] = -1e9
        ok += int(lo.argmax()) in perfect(list(b), p, ns["term"])[1]
    return ok / len(S)


def play(ns, net, opp, net_player, seed):
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    b, pl = [0] * 9, 1
    while ns["term"](b) is None:
        if pl == net_player:
            a = int(ns["mcts"](net, b, pl, 50, rng, noise=False).argmax())
        elif opp == "random":
            a = r.choice([i for i in range(9) if b[i] == 0])
        else:
            a = perfect(b, pl, ns["term"])[1][0]
        b[a] = pl
        pl = -pl
    return ns["term"](b) * net_player


def check(ns, R_):
    def parts():
        net = need(ns, "Net")()
        x = need(ns, "encode")([1, -1, 0, 0, 0, 0, 0, 0, 0], -1)
        ensure(x.tolist()[:2] == [0.0, 1.0] and x.tolist()[9:11] == [1.0, 0.0], "sıradaki oyuncunun bakışı")
        lo, v = net(x[None])
        ensure(tuple(lo.shape) == (1, 9) and -1 <= float(v) <= 1, "çıktılar")
        N = need(ns, "mcts")(net, [1, 1, 0, -1, -1, 0, 0, 0, 0], 1, 50, np.random.default_rng(0), noise=False)
        ensure(int(np.argmax(N)) == 2 and N.sum() == 50, "eğitimsiz ağla bile anında kazancı bulur")
        data = need(ns, "self_play")(net, 10, np.random.default_rng(1))
        ensure(all(abs(z) <= 1 and abs(sum(pi) - 1) < 1e-6 for _, _, pi, z in data), "örnekler")
        ensure(len({z * p for _, p, _, z in data}) == 1, "z sıradaki oyuncunun bakışından tutarlı")
    R_.test("bileşenler", parts)

    def full():
        S = states(ns["term"])
        torch.manual_seed(0)
        acc0 = policy_acc(ns["Net"](), ns, S)
        net = need(ns, "alphazero")(15, 20, 30, 0)
        acc = policy_acc(net, ns, S)
        rp = [play(ns, net, "perfect", 1, s) for s in range(3)] + [play(ns, net, "perfect", -1, s) for s in range(3)]
        rr = [play(ns, net, "random", 1, s) for s in range(5)] + [play(ns, net, "random", -1, s) for s in range(5)]
        ensure(acc > 0.8 and acc > acc0 + 0.2, f"politika başlığı optimal hamleleri öğrenmeli ({acc0:.2f} → {acc:.2f})")
        ensure(all(v >= 0 for v in rp), "kusursuz oyuncuya karşı kaybetmemeli")
        ensure(all(v >= 0 for v in rr) and sum(v == 1 for v in rr) >= 7, "rastgeleyi yenmeli")
        R_.table("Mini AlphaZero, 15 yineleme × 20 oyun × 30 benzetim", ["ölçü", "değer"],
                 [[f"politika başlığı (aramasız) optimal hamle oranı, {len(S)} durum", f"{acc0:.2f} → {acc:.2f}"],
                  ["MCTS(50) – kusursuz oyuncu (6 oyun, her iki renk)", f"{rp.count(1)}G {rp.count(0)}B {rp.count(-1)}M"],
                  ["MCTS(50) – rastgele (10 oyun)", f"{rr.count(1)}G {rr.count(0)}B {rr.count(-1)}M"]],
                 note="İnsan verisi, rollout ve el yapımı değerlendirme yok: yalnız kurallar ve kendine karşı oyun. Ziyaret sayıları ağın önselinden daha güçlü bir politikadır; ağ onu taklit ederek iyileşir.")
    R_.test("alphazero", full)
