import numpy as np
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        betas = np.linspace(1e-4, 0.02, 1000)
        rows = []
        for v0 in (0.0, 1.0, 25.0):
            vs = need(ns, "var_recursion")(v0, betas, True)
            vu = ns["var_recursion"](v0, betas, False)
            rows.append([v0, f"{vs[99]:.3f}", f"{vs[-1]:.4f}", f"{vu[-1]:.2f}"])
            close(vs[-1], 1.0, atol=1e-3, what=f"v₀ = {v0}: ölçekli varyans → 1")
        rng = np.random.default_rng(0)
        x = need(ns, "forward_chain")(5.0, betas, rng, 20000)
        ab = np.cumprod(1 - betas)[-1]
        close((x.mean(), x.var()), (5 * np.sqrt(ab), 1 - ab), atol=0.03, what="zincir = kısa yol: N(√ᾱ x₀, 1 − ᾱ)")
        s = need(ns, "snr")(np.cumprod(1 - betas))
        ensure(s[0] > 1e3 and s[-1] < 1e-3, "SNR yüksekten ~0'a")
        R_.table("Varyans özyinelemesi, doğrusal β (10⁻⁴ → 0.02), T = 1000", ["başlangıç varyansı", "t = 100 (ölçekli)", "t = 1000 (ölçekli)", "t = 1000 (ölçeksiz)"], rows,
                 note="Ölçeklenmiş süreçte varyans (1 − β)v + β ile 1'e doğru kasılır: başlangıç dağılımı ne olursa olsun x_T ≈ N(0, 1) — örneklemenin başlangıç noktası verinin ölçeğinden bağımsızdır. Ölçeksiz süreçte varyans Σβ kadar artar ve veri ölçeğine bağlı kalır (bu, 'variance exploding' SDE'lerin seçimidir; DDPM 'variance preserving'dir).")
    R_.test("ileri süreç", t)
