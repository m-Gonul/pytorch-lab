import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 0, func="pg_theorem"), max_loops(src, 0, func="evaluate"), max_loops(src, 1, func="finite_diff")))
    rng = np.random.default_rng(5)
    S, A = 6, 3
    P = rng.random((A, S, S)) ** 3
    P /= P.sum(-1, keepdims=True)
    R = rng.normal(0, 1, (S, A))
    d0 = rng.dirichlet(np.ones(S))

    def thm():
        for gamma in (0.5, 0.9, 0.99):
            th = rng.normal(0, 1, (S, A))
            close(need(ns, "pg_theorem")(th, P, R, d0, gamma), need(ns, "finite_diff")(th, P, R, d0, gamma), atol=1e-5, rtol=1e-5, what=f"teorem = sonlu fark (γ = {gamma})")
        pi = ns["softmax_policy"](th)
        V, Q = ns["evaluate"](P, R, pi, 0.9)
        close((pi * Q).sum(1), V, atol=1e-10, what="V = Σπ Q")
    R_.test("politika gradyanı teoremi", thm)

    def ascent():
        gamma = 0.9
        th = np.zeros((S, A))
        rows = []
        Vs = np.zeros(S)
        for _ in range(3000):
            Vs = (R + gamma * np.einsum("ast,t->sa", P, Vs)).max(1)
        Jstar = float(d0 @ Vs)
        for it in range(401):
            if it in (0, 10, 50, 100, 400):
                rows.append([it, f"{ns['J'](th, P, R, d0, gamma):.4f}"])
            th = th + 5.0 * ns["pg_theorem"](th, P, R, d0, gamma)
        ensure(Jstar - ns["J"](th, P, R, d0, gamma) < 0.05, "gradyan yükselişi J*'a yaklaşmalı")
        R_.table(f"Tam gradyanla politika yükselişi (J* = {Jstar:.4f}, değer yinelemesi)", ["yineleme", "J(θ)"], rows,
                 note="Softmax parametrelemede optimal deterministik politika ancak θ → ∞'da elde edilir; yaklaşım yavaşlar (gradyan π(1−π) ile söner).")
    R_.test("gradyan yükselişi", ascent)
