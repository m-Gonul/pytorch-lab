import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        ensure(need(ns, "causal")(4).sum() == 10, "nedensel")
        sw = need(ns, "sliding_window")(8, 3)
        ensure(sw[5].tolist() == [False, False, False, True, True, True, False, False], "kayan pencere")
        pl = need(ns, "prefix_lm")(6, 3)
        ensure(pl[0, 2] and not pl[3, 4] and pl[4, 1], "prefix-LM")
        st = need(ns, "strided")(12, 4)
        ensure(st[9, 8] and st[9, 5] and st[9, 1] and not st[9, 4], "strided (Sparse Transformer)")
        T = 256
        rows = []
        for name, M in (("nedensel", ns["causal"](T)), ("kayan pencere w = 8", ns["sliding_window"](T, 8)), ("strided s = 16", ns["strided"](T, 16))):
            r = [name, f"{M.sum() / ns['causal'](T).sum():.0%}"]
            for L in (1, 2, 4, 8):
                R = need(ns, "reach")(M, L)
                r.append(int(R[T - 1].sum()))
            rows.append(r)
        Rw = ns["reach"](ns["sliding_window"](T, 8), 4)
        ensure(int(Rw[T - 1].sum()) == 4 * 7 + 1, "w = 8, 4 katman: 4·(w − 1) + 1 = 29 token")
        ensure(int(ns["reach"](ns["strided"](T, 16), 2)[T - 1].sum()) == T, "strided: 2 katmanda tüm geçmiş")
        R_.table("T = 256: son token'ın L katman sonra 'görebildiği' girdi sayısı", ["maske", "nedensele göre çift", "L = 1", "L = 2", "L = 4", "L = 8"], rows,
                 note="Seyrek maskeler attention maliyetini düşürür ama bilgi katmanlar boyunca yayılarak yine uzağa ulaşır: kayan pencerede erişim L·(w − 1) + 1 (CNN'in receptive field'ı gibi, Longformer/Mistral), strided desende (Child ve ark. 2019) iki katmanda tüm geçmiş — O(T√T) maliyetle.")
    R_.test("maske ailesi", t)
