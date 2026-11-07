from labcheck import need, ensure


def check(ns, R_):
    def t():
        v, m, n = need(ns, "minimax")([0] * 9, 1)
        ensure(v == 0 and n == 549946, f"boş tahta: beraberlik, 549 946 düğüm (bulunan {v}, {n})")
        va, ma, na = need(ns, "alphabeta")([0] * 9, 1)
        ensure(va == 0 and na < n / 10, "α–β aynı değer, çok daha az düğüm")
        b = [1, -1, 0, 0, 1, 0, 0, 0, 0]
        ensure(ns["minimax"](b, -1)[0] == 1 and ns["alphabeta"](b, -1)[0] == 1, "O 8'i kapatsa da X kazanır (çatal)")
        R_.table("Tic-tac-toe oyun ağacı", ["yöntem", "değer", "ziyaret edilen düğüm"], [["minimax", v, n], ["α–β (hamle sırası 0..8)", va, na]],
                 note="Go'da dallanma ~250, derinlik ~150: tam arama imkânsız. MCTS aramayı umut veren dallara yoğunlaştırır, AlphaGo da politika ağıyla dallanmayı, değer ağıyla derinliği kısar.")
    R_.test("minimax ve α–β", t)
