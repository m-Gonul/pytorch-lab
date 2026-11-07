from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        res = {}
        for n in (8, 16, 64):
            a = need(ns, "win_rate")(False, 200, n, 0)
            b = ns["win_rate"](True, 200, n, 0)
            res[n] = (a, b)
            rows.append([n, f"{a:.2f}", f"{b:.2f}"])
        ensure(res[8][1] > res[8][0] + 0.15, "gürültü kazanan hamleyi az benzetimde bulmayı sağlamalı")
        R_.table("Önseli 0.001 olan kazanan hamle, 200 deneme", ["benzetim", "gürültüsüz", "Dirichlet(0.3), ε = 0.25"], rows,
                 note="PUCT'ta ziyaretsiz bir kenarın puanı c·P·√N: P ≈ 0 ise birçok benzetim boyunca seçilmez. Dirichlet(0.3) çoğu kütleyi birkaç rastgele hamleye koyar: her oyunda başka 'şanslı' hamleler denenir. Benzetim çoğaldıkça gürültüsüz arama da sonunda bulur.")
    R_.test("Dirichlet gürültüsü", t)
