from labcheck import need, ensure


def check(ns, R_):
    def t():
        ensure(need(ns, "greedy_len")({}) == 999, "boş Q yol bulmaz")
        rows = []
        res = {}
        for n in (5, 20):
            ps = [need(ns, "prioritized")(n, 1e-4, s) for s in range(5)]
            dq = [need(ns, "dyna_updates")(n, s) for s in range(5)]
            res[n] = (sum(ps) / 5, sum(dq) / 5)
            rows.append([n, f"{res[n][0]:.0f}", f"{res[n][1]:.0f}", f"{res[n][1] / res[n][0]:.1f}×"])
        ensure(all(v[0] < v[1] for v in res.values()), "öncelikli tarama daha az güncellemeyle optimal yolu bulmalı")
        R_.table("Optimal yola (14 adım) kadar Q güncellemesi, 5 tohum", ["n", "öncelikli tarama", "Dyna-Q (düzgün)", "oran"], rows,
                 note="Düzgün örnekleme güncellemelerin çoğunu değeri hiç değişmeyecek çiftlere harcar; öncelikli tarama değişimi hedeften geriye, öncüller boyunca yayar.")
    R_.test("öncelikli tarama", t)
