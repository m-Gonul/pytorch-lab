from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for p, n, c in [(0.5, 100, 0.1), (0.5, 100, 0.2), (0.9, 100, 0.1), (0.5, 1000, 0.05), (0.99, 50, 0.05)]:
            emp, bd = need(ns, "coverage")(p, n, c, 200000, 1)
            ensure(emp <= bd + 0.005, f"Hoeffding ihlal edilmemeli ({p}, {n}, {c})")
            rows.append([p, n, c, f"{emp:.4f}", f"{bd:.4f}"])
        R_.table("P(|x̄ − p| ≥ c): deney vs Hoeffding", ["p", "n", "c", "deneysel", "2e^{−2nc²}"], rows,
                 note="Hoeffding dağılımdan bağımsızdır, bu yüzden özellikle p uçlarda (varyans küçük) gevşektir; UCB-V, KL-UCB gibi varyant'lar bu boşluğu kapatır.")
    R_.test("Hoeffding", t)
