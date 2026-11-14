from labcheck import close, need, ensure
import numpy as np


def check(ns, R_):
    def t():
        transitions = [([1., 0.], 1., [0., 2.]), ([0., 2.], 2., [2., 1.]), ([2., 1.], 3., [0., 0.])]
        X = np.array([t[0] for t in transitions]); Z = np.array([t[2] for t in transitions]); rewards = np.array([t[1] for t in transitions])
        gamma, epsilon = .7, .25
        A = epsilon * np.eye(2) + X.T @ (X - gamma * Z)
        expected = np.linalg.solve(A, X.T @ rewards)
        # A is deliberately nonsymmetric; missing the right transpose fails.
        ensure(not np.allclose(A, A.T), "bağımsız denetim nonsimetrik olmalı")
        for name in ("batch_lstd", "recursive_lstd"):
            f = need(ns, name)
            close(f(transitions, gamma, epsilon), expected, rtol=1e-10, what=name + ": NumPy solve")
            try:
                f([([1.], 1., [2.25])], .5, .125)
            except ValueError:
                pass
            else:
                ensure(False, name + ": tekillik bildirilmeli")
        R_.table("Aynı moment sistemi", ["yöntem", "w1", "w2"], [[name, *ns[name](transitions, gamma, epsilon)] for name in ("batch_lstd", "recursive_lstd")])
    R_.test("Bağımsız nonsimetrik çözüm ve tekillik", t)
