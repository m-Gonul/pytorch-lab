import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 0, func="posterior"), max_loops(src, 1, func="sequential_posterior"), max_loops(src, 0, func="predictive"), max_loops(src, 0, func="joint_features")))
    rng = np.random.default_rng(3)
    w = np.array([0.5, -1.0, 2.0])
    X = rng.normal(0, 1, (200, 3))
    r = X @ w + 0.5 * rng.normal(0, 1, 200)

    def post():
        mu, S = need(ns, "posterior")(np.array([[2.0]]), np.array([3.0]), 1.0, 1.0)
        close(mu, np.array([1.2]), what="el hesabı μ")
        close(S, np.array([[0.2]]), what="el hesabı Σ")
        mu, S = ns["posterior"](X, r, 1.0, 0.5)
        mu2, S2 = need(ns, "sequential_posterior")(X, r, 1.0, 0.5)
        close(mu2, mu, atol=1e-9, what="sıralı = toplu")
        close(S2, S, atol=1e-12, what="sıralı Σ")
        mean, var = need(ns, "predictive")(np.eye(3), mu, S, 0.5)
        close(var, np.diag(S) + 0.25, what="öngörü varyansı")
        rows = []
        for n in (5, 20, 200):
            m_, S_ = ns["posterior"](X[:n], r[:n], 1.0, 0.5)
            rows.append([n, np.array2string(m_, precision=2), np.array2string(np.sqrt(np.diag(S_)), precision=3)])
        R_.table("Sonsal daralması, gerçek w = (0.5, −1, 2)", ["n", "μ", "√diag Σ"], rows, note="Belirsizlik ~ σ/√n hızında daralır; LinUCB'nin keşif bonusu bununla birlikte söner.")
    R_.test("posterior, sequential_posterior, predictive", post)

    def jf():
        xs = np.array([1.0, 2.0])
        XA = np.array([[1.0, 0, 3], [0, 1, 1]])
        F = need(ns, "joint_features")(xs, XA)
        close(F, np.array([[1, 0, 3, 2, 0, 6], [0, 1, 1, 0, 2, 2.0]]), what="dış çarpım öznitelikleri")
    R_.test("joint_features", jf)
