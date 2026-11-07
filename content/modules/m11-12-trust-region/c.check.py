import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 0, func="fisher"), max_loops(src, 0, func="fvp"), max_loops(src, 1, func="cg"), max_loops(src, 0, func="natural_step")))
    rng = np.random.default_rng(0)
    B, K = 200, 5
    th = rng.normal(0, 1, (B, K))
    r = rng.random((B, K))

    def fis():
        pi = need(ns, "softmax")(th)
        F = need(ns, "fisher")(pi)
        a = np.array([rng.choice(K, 20000, p=p) for p in pi[:3]])
        for b in range(3):
            sc = np.eye(K)[a[b]] - pi[b]
            close(F[b], sc.T @ sc / 20000, atol=0.01, what="F = E[∇logπ ∇logπᵀ] (örneklem)")
        v = rng.normal(0, 1, (B, K))
        close(need(ns, "fvp")(pi, v), np.einsum("bij,bj->bi", F, v), what="matrissiz Fv")
        d = 1e-3 * rng.normal(0, 1, (B, K))
        close(need(ns, "kl")(pi, ns["softmax"](th + d)), 0.5 * np.einsum("bi,bij,bj->b", d, F, d), rtol=0.01, atol=1e-12, what="KL ≈ ½ dᵀFd")
        g = need(ns, "policy_grad")(th, r)
        h = 1e-6
        num = np.stack([((ns["softmax"](th + h * np.eye(K)[k]) * r).sum(1) - (ns["softmax"](th - h * np.eye(K)[k]) * r).sum(1)) / (2 * h) for k in range(K)], 1)
        close(g, num, atol=1e-8, what="politika gradyanı")
    R_.test("Fisher, KL, gradyan", fis)

    def nat():
        pi = ns["softmax"](th)
        A = np.eye(K)[None] * 3 + 0.5 * np.ones((B, K, K))
        bvec = rng.normal(0, 1, (B, K))
        x = need(ns, "cg")(lambda v: np.einsum("bij,bj->bi", A, v), bvec, 50)
        close(x, np.linalg.solve(A, bvec[..., None])[..., 0], atol=1e-8, what="CG = doğrudan çözüm (toplu)")
        delta = 0.005
        tn = need(ns, "natural_step")(th, r, delta)
        g = ns["policy_grad"](th, r)
        tv = th + g * np.sqrt(2 * delta / np.einsum("bi,bi->b", g, ns["fvp"](pi, g)))[:, None]
        kn, kv = ns["kl"](pi, ns["softmax"](tn)), ns["kl"](pi, ns["softmax"](tv))
        ensure(np.abs(kn / delta - 1).max() < 0.15 and np.abs(kv / delta - 1).max() < 0.15, "iki adım da KL ≈ δ")
        gain_n = (ns["softmax"](tn) * r).sum(1) - (pi * r).sum(1)
        gain_v = (ns["softmax"](tv) * r).sum(1) - (pi * r).sum(1)
        ensure((gain_n >= gain_v - 1e-9).mean() > 0.95, "aynı KL'de doğal gradyan daha çok iyileştirir")
        s = ns["cg"](lambda v: ns["fvp"](pi, v) + 1e-3 * v, g)
        adv = r - (pi * r).sum(1, keepdims=True)
        cos = (s * (adv - adv.mean(1, keepdims=True))).sum(1) / (np.linalg.norm(s, axis=1) * np.linalg.norm(adv - adv.mean(1, keepdims=True), axis=1))
        R_.table(f"{B} softmax haydut, KL bütçesi δ = {delta}", ["ölçü", "doğal gradyan", "düz gradyan (aynı KL)"],
                 [["ort. KL / δ", f"{kn.mean() / delta:.3f}", f"{kv.mean() / delta:.3f}"], ["ort. beklenen ödül artışı", f"{gain_n.mean():.5f}", f"{gain_v.mean():.5f}"],
                  ["F⁻¹g ile (avantaj − ort.) arasındaki kosinüs", f"{cos.mean():.4f}", ""]],
                 note="Softmax'ta Fisher tekildir (θ'ya sabit eklemek politikayı değiştirmez); sönümleme bunu çözer. Doğal gradyan, merkezlenmiş avantajın kendisidir: parametrelemeden bağımsız 'en dik' yön.")
    R_.test("doğal gradyan", nat)
