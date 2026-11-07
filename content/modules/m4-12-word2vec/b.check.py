import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t_vocab():
        S = [["a", "b", "a", "c"], ["b", "a", "d"]]
        ensure(need(ns, "build_vocab")(S) == {"a": 0, "b": 1, "c": 2, "d": 3}, "sıralama: sayı, sonra alfabe")
        ensure(ns["build_vocab"](S, 2) == {"a": 0, "b": 1}, "min_count")
        v = ns["build_vocab"](S, 2)
        ensure(need(ns, "oov_hash")("ab") == 31 * 97 + 98, "hash")
        e = need(ns, "encode")(["a", "zebra", "b", "zebra", "c"], v, 3)
        ensure(e[0] == 0 and e[2] == 1 and e[1] == e[3] == 2 + ns["oov_hash"]("zebra") % 3 and 2 <= e[4] < 5, "OOV kovaları")
    R_.test("sözlük ve OOV", t_vocab)

    def t_pairs():
        P = need(ns, "skipgram_pairs")([0, 1, 2, 3, 4], 2)
        ensure(len(P) == 14 and P[:2] == [(0, 1), (0, 2)] and (2, 0) in P and (4, 2) in P and (4, 1) not in P, "skip-gram çiftleri")
        ensure(need(ns, "cbow_examples")([5, 6, 7], 1) == [([6], 5), ([5, 7], 6), ([6], 7)], "CBOW")
    R_.test("çiftler", t_pairs)

    def t_grad():
        J, dv, duo, duk = need(ns, "sgns_loss_grad")([1.0], [1.0], [[0.0], [-1.0]])
        close(J, 1.3196705557466014, what="kayıp")
        close(dv[0], -0.5378828427399902, what="dJ/dv")
        rnd = random.Random(0)
        v = [rnd.gauss(0, 1) for _ in range(4)]
        uo = [rnd.gauss(0, 1) for _ in range(4)]
        un = [[rnd.gauss(0, 1) for _ in range(4)] for _ in range(3)]
        J, dv, duo, duk = ns["sgns_loss_grad"](v, uo, un)
        h = 1e-6
        for i in range(4):
            vp = v[:]
            vp[i] += h
            close((ns["sgns_loss_grad"](vp, uo, un)[0] - J) / h, dv[i], atol=1e-4, what="sayısal dv")
            up = [r[:] for r in un]
            up[1][i] += h
            close((ns["sgns_loss_grad"](v, uo, up)[0] - J) / h, duk[1][i], atol=1e-4, what="sayısal du_k")
        close(need(ns, "noise_dist")([81, 16]), [27 / 35, 8 / 35], what="3/4 kuvvet")
        close(need(ns, "discard_prob")(1e-3), 0.9, what="alt-örnekleme")
        ensure(ns["discard_prob"](1e-6) == 0, "nadir kelime atılmaz")
        rows = []
        zipf = [1 / r for r in range(1, 1001)]
        tot = sum(zipf)
        for pw in (1.0, 0.75, 0.5, 0.0):
            p = ns["noise_dist"](zipf, pw)
            rows.append([pw, f"{p[0]:.3f}", f"{sum(p[:10]):.3f}", f"{sum(p[500:]):.3f}"])
        R_.table("Zipf dağılımlı 1000 kelimede gürültü dağılımı", ["kuvvet", "P(en sık)", "P(ilk 10)", "P(son 500)"], rows,
                 note="Kuvvet 1 (ham sıklık) negatif örneklerin çoğunu birkaç işlev kelimesine harcar; 0 (düzgün) nadir kelimelere aşırı ağırlık verir. 3/4 arada bir uzlaşmadır.")
    R_.test("SGNS kaybı ve gradyanı", t_grad)
