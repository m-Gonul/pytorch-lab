import math
import numpy as np
import torch
from labcheck import need, ensure, bench, fmt_time


def seq_self_play(net, n_games, n_sims, rng, term, c=1.5):
    out = []
    for _ in range(n_games):
        b, pl, hist = [0] * 9, 1, []
        while term(b) is None:
            root = {"b": b[:], "pl": pl}

            def node(bb, p):
                legal = np.array([x == 0 for x in bb])
                with torch.no_grad():
                    B = torch.tensor(bb, dtype=torch.float32) * p
                    lo, v = net(torch.cat([(B == 1).float(), (B == -1).float()])[None])
                pr = torch.softmax(lo[0], 0).numpy() * legal
                return {"b": bb, "pl": p, "legal": legal, "P": pr / pr.sum(), "v": float(v[0]), "N": np.zeros(9), "W": np.zeros(9), "ch": {}}

            root = node(b[:], pl)
            eta = np.zeros(9)
            eta[root["legal"]] = rng.dirichlet([0.3] * int(root["legal"].sum()))
            root["P"] = 0.75 * root["P"] + 0.25 * eta
            for _ in range(n_sims):
                nd, path = root, []
                while True:
                    Q = np.where(nd["N"] > 0, nd["W"] / np.maximum(nd["N"], 1), 0.0)
                    U = c * nd["P"] * math.sqrt(nd["N"].sum() + 1) / (1 + nd["N"])
                    a = int(np.argmax(np.where(nd["legal"], Q + U, -1e9)))
                    path.append((nd, a))
                    if a in nd["ch"]:
                        nd = nd["ch"][a]
                        continue
                    b2 = nd["b"][:]
                    b2[a] = nd["pl"]
                    t = term(b2)
                    if t is not None:
                        v = -abs(t)
                    else:
                        nd["ch"][a] = node(b2, -nd["pl"])
                        v = nd["ch"][a]["v"]
                    break
                for n_, a in reversed(path):
                    v = -v
                    n_["N"][a] += 1
                    n_["W"][a] += v
            pi = root["N"] / root["N"].sum()
            a = int(rng.choice(9, p=pi)) if sum(1 for x in b if x) < 3 else int(root["N"].argmax())
            hist.append(pi)
            b[a] = pl
            pl = -pl
        out += hist
    return out


def check(ns, R_):
    def t():
        torch.manual_seed(0)
        net = ns["Net"]()
        X = need(ns, "encode_batch")([[1, -1, 0, 0, 0, 0, 0, 0, 0], [0] * 9], [-1, 1])
        ensure(tuple(X.shape) == (2, 18) and X[0, :2].tolist() == [0.0, 1.0] and X[0, 9:11].tolist() == [1.0, 0.0], "kodlama")
        data = need(ns, "batched_self_play")(net, 64, 30, np.random.default_rng(0))
        ensure(5 * 64 <= len(data) <= 9 * 64, "her oyun 5–9 hamle")
        ensure(all(abs(sum(pi) - 1) < 1e-6 and abs(z) <= 1 for _, _, pi, z in data), "örnekler")
        X0 = [d for d in data if not any(d[0])]
        ensure(len(X0) == 64, "her oyunun boş tahta örneği")
        rows = []
        for G in (16, 64, 128):
            tb = bench(lambda: ns["batched_self_play"](net, G, 30, np.random.default_rng(1)), repeat=1, number=1)
            ts = bench(lambda: seq_self_play(net, min(G, 16), 30, np.random.default_rng(1), ns["term"]), repeat=1, number=1) * G / min(G, 16)
            rows.append([G, fmt_time(ts), fmt_time(tb), f"{ts / tb:.1f}×"])
        R_.table("Kendine karşı oyun, 30 benzetim/hamle", ["oyun", "sıralı (oyun başına ağ çağrısı)", "toplu yaprak değerlendirme", "hızlanma"], rows,
                 note="CPU'da ve küçük ağda kazanç, ağ çağrısı başına sabit maliyetin (Python → C++ → çekirdek) G'ye bölünmesinden gelir; GPU'da ve büyük ağda oran çok daha büyüktür. AlphaGo Zero toplu değerlendirmeyi 8'lik mini-batch'lerle ve sanal kayıpla yapar.")
    R_.test("batched_self_play", t)
