import random
from itertools import product
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rng = random.Random(0)
        rows = []
        for n in (2, 3, 4, 5):
            for _ in range(5):
                table = {row: rng.randint(0, 1) for row in product((0, 1), repeat=n)}
                f = lambda *r: table[r]
                net = need(ns, "truth_table_net")(f, n)
                ensure(all(need(ns, "evaluate")(net, r) == table[r] for r in table), f"n = {n}: rastgele fonksiyon tam temsil")
            parity = lambda *r: sum(r) % 2
            rows.append([n, len(ns["truth_table_net"](parity, n)[0]), 2 ** n])
        R_.table("Eşlik (parity) fonksiyonu için DNF gizli birim sayısı", ["n", "gizli TLU", "2ⁿ"], rows,
                 note="İki katman her Boole fonksiyonunu temsil eder ama eşlik için 2ⁿ⁻¹ birim gerekir; derin bir ağ (XOR'ların ağacı) aynı işi ~n birimle yapar — derinliğin üstel verimliliği.")
    R_.test("DNF inşası", t)
