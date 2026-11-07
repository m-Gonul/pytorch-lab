import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    rng = np.random.default_rng(0)
    S, A = 60, 4
    P = rng.random((A, S, S)) ** 6; P /= P.sum(-1, keepdims=True)
    R = rng.normal(size=(S, A))

    def same():
        g = 0.95
        V1, p1, *_ = need(ns, "vi")(P, R, g, 1e-10)
        V2, p2, *_ = need(ns, "pi_exact")(P, R, g)
        V3, p3, *_ = need(ns, "mpi")(P, R, g, 5, 1e-10)
        close(V1, V2, atol=1e-7, what="VI ve PI"); close(V3, V2, atol=1e-7, what="MPI ve PI")
        ensure(np.array_equal(p1, p2) and np.array_equal(p3, p2), "aynı politika")
    R_.test("üçü aynı V* ve π*", same)

    def acct():
        rows = []
        for g in (0.9, 0.99):
            _, _, i1, b1 = ns["vi"](P, R, g, 1e-8)
            _, _, i2, b2 = ns["pi_exact"](P, R, g)
            row = [g, f"{i1} / {b1:,}", f"{i2} / {b2:,}"]
            for k in (2, 5, 20):
                _, _, i3, b3 = ns["mpi"](P, R, g, k, 1e-8)
                row.append(f"{i3} / {b3:,}")
            rows.append(row)
        ensure(int(rows[1][2].split(" / ")[0]) < int(rows[1][1].split(" / ")[0]), "PI daha az iterasyon")
        R_.table("İterasyon / backup (60 durum, 4 eylem, ε = 1e-8)", ["γ", "VI", "PI (tam çözüm)", "MPI k = 2", "MPI k = 5", "MPI k = 20"], rows,
                 note="γ → 1'de VI'nın iterasyonu ~1/(1−γ) büyür; PI birkaç iterasyonda biter ama her biri bir doğrusal sistem.")
    R_.test("hesap muhasebesi", acct)
