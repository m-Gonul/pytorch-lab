import math
import torch
from labcheck import close, need, ensure


def check(ns, R_):
    g = torch.Generator().manual_seed(0)
    B, A, S = 300, 3, 10
    P = torch.rand(B, A, S, S, generator=g, dtype=torch.float64) ** 3; P /= P.sum(-1, keepdim=True)
    R = torch.randn(B, S, A, generator=g, dtype=torch.float64)

    def bv():
        V, d = need(ns, "batched_vi")(P, R, 0.9, 200)
        ensure(tuple(V.shape) == (B, S) and tuple(d.shape) == (200, B), "şekiller")
        Q = R + 0.9 * torch.einsum("bast,bt->bsa", P, V)
        close(V.numpy(), Q.max(-1).values.numpy(), atol=1e-8, what="sabit nokta")
        ratio = (d[1:] / d[:-1].clamp_min(1e-300))[:60]
        ensure(bool((ratio <= 0.9 + 1e-9).all()), "ardışık farklar her adımda en az γ katına inmeli")
        need_n = [int((d[:, b] > 1e-6).sum()) + 1 for b in range(B)]
        theory = math.ceil(math.log(1e-6 / d[0].max().item()) / math.log(0.9))
        return f"1e-6'ya kadar iterasyon: en az {min(need_n)}, en çok {max(need_n)}; teorik üst sınır log(ε/‖V₁−V₀‖)/log γ ≈ {theory}"
    R_.test("batched_vi (300 MDP)", bv)

    def cr():
        r = need(ns, "contraction_ratio")(P[0], R[0], 0.9)
        ensure(float(r.max()) <= 0.9 + 1e-12, f"oran {float(r.max()):.4f} > γ")
        return f"en büyük oran {float(r.max()):.4f} (γ = 0.9), ortalama {float(r.mean()):.3f}"
    R_.test("contraction_ratio ≤ γ", cr)

    def grad():
        Pm, Rm = P[1], R[1].clone().requires_grad_(True)
        V = need(ns, "soft_v_star")(Pm, Rm, 0.9, 400)
        V[0].backward()
        with torch.no_grad():
            pi = (Rm + 0.9 * torch.einsum("ast,t->sa", Pm, V)).argmax(-1)
            Tp = Pm[pi, torch.arange(S)]
            d = torch.linalg.solve(torch.eye(S, dtype=torch.float64) - 0.9 * Tp.T, torch.eye(S, dtype=torch.float64)[0])
            occ = torch.zeros(S, A, dtype=torch.float64); occ[torch.arange(S), pi] = d
        close(Rm.grad.numpy(), occ.numpy(), atol=1e-8, what="∂V*(s0)/∂R = indirimli ziyaret ölçüsü")
        close(float(Rm.grad.sum()), 1 / (1 - 0.9), atol=1e-8, what="toplam 1/(1 − γ)")
    R_.test("autograd: ∂V*/∂R = d^π*", grad)
