import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        p = np.array([0.5, 0.3, 0.2])
        r = np.array([0.0, 2.0, 1.0])
        q = need(ns, "bon_dist")(p, r, 2)
        close(q, [0.25, 1 - 0.7 ** 2, 0.7 ** 2 - 0.25], what="n = 2 örnek")
        close(need(ns, "kl")(p, p), 0.0, what="KL(p‖p)")
        rng = np.random.default_rng(0)
        P = rng.dirichlet(np.ones(5000))
        R = rng.normal(size=5000)
        rows = []
        for n in (1, 2, 4, 16, 64, 256):
            qn = ns["bon_dist"](P, R, n)
            k = ns["kl"](qn, P)
            bound = np.log(n) - (n - 1) / n
            ensure(k <= bound + 1e-9, "KL ≤ log n − (n − 1)/n")
            rows.append([n, f"{qn @ R:.3f}", f"{k:.3f}", f"{bound:.3f}"])
        R_.table("Best-of-n: 5000 çıktılı referans, ödül ~ N(0, 1)", ["n", "beklenen ödül", "KL(BoN‖ref)", "log n − (n−1)/n"], rows,
                 note="RLHF'e alternatif en basit yöntem: n tamamlama örnekle, ödül modeline göre en iyisini seç. KL n ile yalnız logaritmik büyür, beklenen ödül ise ~√(2 log n) gibi: KL bütçesi başına çok verimlidir ama çıkarımda n kat hesap ister. Sınır sürekli (bağsız) dağılımlarda sıkıdır (Beirami ve ark. 2024).")
    R_.test("best-of-n", t)
