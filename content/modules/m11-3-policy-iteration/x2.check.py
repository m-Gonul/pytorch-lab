import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    rng = np.random.default_rng(0)

    def t():
        worst = 0
        for _ in range(200):
            S, A = 12, 3
            P = rng.random((A, S, S)) ** 3; P /= P.sum(-1, keepdims=True)
            R = rng.normal(size=(S, A)); g = rng.uniform(0.5, 0.99)
            pi = rng.integers(0, A, S)
            dV, dQ = need(ns, "check_improvement")(P, R, g, pi)
            ensure(dV.min() >= -1e-9 and dQ.min() >= -1e-9, "iyileştirme değeri düşürmemeli")
            new = need(ns, "random_improving_policy")(P, R, g, pi, int(rng.integers(1e6)))
            idx = np.arange(S)
            V0 = np.linalg.solve(np.eye(S) - g * P[pi, idx], R[idx, pi]); V1 = np.linalg.solve(np.eye(S) - g * P[new, idx], R[idx, new])
            worst = min(worst, (V1 - V0).min())
        ensure(worst >= -1e-9, "tek durumlu iyileştirme de değeri düşürmemeli")
        return "200 rastgele MDP'de hem tam açgözlü hem tek durumlu değişiklik hiçbir durumun değerini düşürmedi"
    R_.test("politika iyileştirme teoremi", t)
