import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("bellman_q döngüsüz", lambda: max_loops(src, 0, func="bellman_q"))
    R_.test("policy_eval_exact döngüsüz", lambda: max_loops(src, 0, func="policy_eval_exact"))
    R_.test("policy_iteration: tek döngü", lambda: max_loops(src, 1, func="policy_iteration"))
    R_.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    S, A = 12, 3
    P = rng.random((S, A, S)) ** 4; P /= P.sum(2, keepdims=True)
    R = rng.normal(size=(S, A))
    V = rng.normal(size=S)

    def bq():
        Q = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                Q[s, a] = R[s, a] + 0.9 * sum(P[s, a, t] * V[t] for t in range(S))
        close(need(ns, "bellman_q")(V, P, R, 0.9), Q)
    R_.test("bellman_q", bq)

    def pe():
        pi = rng.random((S, A)); pi /= pi.sum(1, keepdims=True)
        Vp = need(ns, "policy_eval_exact")(P, R, pi, 0.9)
        Q = R + 0.9 * P @ Vp
        close(Vp, (pi * Q).sum(1), what="Bellman beklenti denklemi")
    R_.test("policy_eval_exact (stokastik π)", pe)

    def pi_():
        V, pol = need(ns, "policy_iteration")(P, R, 0.95)
        Vr = np.zeros(S)
        for _ in range(3000):
            Vr = (R + 0.95 * P @ Vr).max(1)
        close(V, Vr, atol=1e-8, what="V* (value iteration ile)")
        ensure(np.array_equal(pol, (R + 0.95 * P @ Vr).argmax(1)), "optimal politika")
        return f"policy iteration {S} durumda yakınsadı; V*(0) = {V[0]:.4f}"
    R_.test("policy_iteration = value iteration", pi_)
