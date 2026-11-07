import numpy as np
from labcheck import close, max_loops, need, ensure


def rand_mdp(rng, S=8, A=3):
    P = rng.random((A, S, S)) ** 3; P /= P.sum(-1, keepdims=True)
    return P, rng.normal(size=(S, A))


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız yineleme döngüsü", lambda: max_loops(src, 1, func="value_iteration"))
    rng = np.random.default_rng(0)

    def pe():
        P, R = rand_mdp(rng)
        pi = rng.integers(0, 3, 8)
        V = need(ns, "policy_eval")(P, R, pi, 0.9)
        Tp = P[pi, np.arange(8)]; Rp = R[np.arange(8), pi]
        close(V, Rp + 0.9 * Tp @ V, atol=1e-10, what="V = R_π + γT_πV")
    R_.test("policy_eval (doğrusal sistem)", pe)

    def bd():
        rows, worst = [], 0
        for g in (0.5, 0.9, 0.99):
            for _ in range(10):
                P, R = rand_mdp(rng)
                Vstar, pistar, _, _ = need(ns, "value_iteration")(P, R, g, 1e-12)
                for eps in (1e-1, 1e-3):
                    V, pi, n, d = ns["value_iteration"](P, R, g, eps)
                    err = np.abs(Vstar - V).max()
                    ensure(err <= need(ns, "bound")(g, d) + 1e-9, f"γ = {g}: hata {err:.2e} > sınır {ns['bound'](g, d):.2e}")
                    loss = np.abs(Vstar - ns["policy_eval"](P, R, pi, g)).max()
                    ensure(loss <= 2 * g * eps / (1 - g) + 1e-9, "politika kaybı sınırı")
                    worst = max(worst, err / ns["bound"](g, d))
            P, R = rand_mdp(rng)
            _, _, n, d = ns["value_iteration"](P, R, g, 1e-6)
            rows.append([g, n, f"{d:.1e}", f"{ns['bound'](g, d):.1e}"])
        R_.table("Değer yinelemesi, ε = 1e-6", ["γ", "iterasyon", "son fark", "V*'a uzaklık sınırı"], rows,
                 note=f"Gerçek hata / sınır oranının en büyüğü {worst:.2f}: sınır geçerli ve oldukça sıkı. γ → 1'de iterasyon ~ log ε / log γ ile patlar.")
    R_.test("durma sınırı ve politika kaybı", bd)
