import numpy as np
import torch
from labcheck import close, need, ensure


def check(ns, R_):
    g = torch.Generator().manual_seed(0)
    S, A = 20, 4
    P = torch.rand(A, S, S, generator=g, dtype=torch.float64) ** 4
    P /= P.sum(-1, keepdim=True)
    R = torch.rand(S, A, generator=g, dtype=torch.float64)

    def qvi():
        Q = need(ns, "q_value_iteration")(P, R, 0.9)
        V = Q.max(1).values
        ref = R + 0.9 * torch.stack([P[a] @ V for a in range(A)], 1)
        close(Q.numpy(), ref.numpy(), atol=1e-8, what="Bellman optimallik denklemi")
        # aynı sonuç: politika değerlendirmesi ile
        pi = Q.argmax(1)
        Ppi = P[pi, torch.arange(S)]
        Vpi = torch.linalg.solve(torch.eye(S, dtype=torch.float64) - 0.9 * Ppi, R[torch.arange(S), pi])
        close(Vpi.numpy(), V.numpy(), atol=1e-8, what="V^{π*} = max Q*")
    R_.test("q_value_iteration", qvi)

    def sq():
        Qs = ns["q_value_iteration"](P, R, 0.9)
        rows = []
        errs = {}
        for rate in ("rescaled", "1/n"):
            for n in (100, 1000, 5000):
                Q = need(ns, "sync_q_learning")(P, R, 0.9, n, rate, 1)
                e = float((Q - Qs).abs().max())
                errs[(rate, n)] = e
                rows.append([rate, n, f"{e:.3f}", f"{float((need(ns, 'greedy')(Q) == Qs.argmax(1)).double().mean()):.2f}"])
        ensure(errs[("rescaled", 5000)] < 0.06, "1/(1+(1−γ)n) ile Q* yakınına gelmeli")
        ensure(errs[("1/n", 5000)] > 1.0, "1/n ile hata büyük kalmalı")
        R_.table("Eşzamanlı Q-öğrenme, 20 durum × 4 eylem, γ = 0.9 (‖Q* ‖∞ ≈ %.1f)" % float(Qs.abs().max()),
                 ["α_n", "yineleme", "‖Q − Q*‖∞", "açgözlü politika uyumu"], rows,
                 note="İkisi de Robbins–Monro koşullarını sağlar; ama 1/n ile önyükleme yanlılığı n^{−(1−γ)} hızında söner. Politika genelde değerlerden çok önce doğrulanır.")
    R_.test("sync_q_learning ve öğrenme oranları", sq)
