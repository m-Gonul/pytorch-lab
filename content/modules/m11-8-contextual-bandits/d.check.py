import torch
from labcheck import close, need, ensure


def check(ns, R_):
    def blr():
        g = torch.Generator().manual_seed(0)
        Z = torch.randn(40, 3, generator=g, dtype=torch.float64)
        A = torch.randint(0, 2, (40,), generator=g)
        R = torch.randn(40, generator=g, dtype=torch.float64)
        m1 = need(ns, "BLR")(2, 3, 1.0, 0.5)
        for z, a, r in zip(Z, A, R):
            m1.update(int(a), z, r)
        m2 = ns["BLR"](2, 3, 1.0, 0.5)
        m2.refit(Z, A, R)
        close(m1.P.numpy(), m2.P.numpy(), atol=1e-10, what="update = refit (kesinlik)")
        close(m1.b.numpy(), m2.b.numpy(), atol=1e-10, what="update = refit (b)")
        Zk = Z[A == 0]
        mu = torch.linalg.solve(Zk.T @ Zk / 0.25 + torch.eye(3, dtype=torch.float64), Zk.T @ R[A == 0] / 0.25)
        q = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
        s = torch.stack([m2.sample(q, torch.Generator().manual_seed(i))[0] for i in range(4000)])
        ensure(abs(float(s.mean() - q @ mu)) < 0.05, "örneklerin ortalaması öngörü ortalaması")
    R_.test("BLR", blr)

    def cmp():
        rl = [need(ns, "run_lints")(2000, s) for s in range(2)]
        rn = [need(ns, "run_neural_linear")(2000, s) for s in range(2)]
        late_l = sum(r[-1] - r[1499] for r in rl) / 2
        late_n = sum(r[-1] - r[1499] for r in rn) / 2
        ensure(late_n < 0.3 * late_l, f"NeuralLinear XOR'u öğrenmeli (son 500 adım pişmanlığı {late_n:.1f} vs {late_l:.1f})")
        R_.table("XOR bağlamsal haydut, 2 tohum", ["yöntem", "pişmanlık t = 500", "t = 2000", "son 500 adım / adım"],
                 [["doğrusal Thompson (x, 1)", f"{sum(r[499] for r in rl) / 2:.0f}", f"{sum(r[-1] for r in rl) / 2:.0f}", f"{late_l / 500:.3f}"],
                  ["NeuralLinear", f"{sum(r[499] for r in rn) / 2:.0f}", f"{sum(r[-1] for r in rn) / 2:.0f}", f"{late_n / 500:.3f}"]],
                 note="Doğrusal model XOR'u temsil edemez: belirsizliği sıfıra gider ama yanlış modele; pişmanlık doğrusal büyür. NeuralLinear öznitelikleri ağla öğrenir, belirsizliği yalnız son katmanda kapalı formda tutar (Riquelme ve ark. 2018).")
    R_.test("LinTS vs NeuralLinear", cmp)
