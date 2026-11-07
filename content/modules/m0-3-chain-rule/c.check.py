import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü/comprehension yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    f = lambda X: (X[:, 0] + X[:, 1]) * X[:, 1]
    g = lambda X: np.stack([X[:, 0] * X[:, 1], X[:, 0] + X[:, 1] ** 2], axis=1)
    calls = []

    def counting(F):
        def w(X):
            calls.append(np.asarray(X).shape)
            return F(X)
        return w

    def grad2():
        out = need(ns, "num_grad_np")(counting(f), np.array([2.0, 3.0]))
        close(out, [3, 8], rtol=1e-6, atol=1e-6)
        ensure(len(calls) == 1, f"f {len(calls)} kez çağrıldı; tek batch çağrısı bekleniyordu")
        return "f tek çağrıda (4×2 batch) değerlendirildi"
    R.test("num_grad_np: [3, 8], tek çağrı", grad2)

    def grad_n():
        A = np.random.default_rng(0).normal(size=(6, 6)); A = A + A.T
        q = lambda X: 0.5 * np.einsum("ki,ij,kj->k", X, A, X) + np.sin(X).sum(1)
        x = np.linspace(-1, 1, 6)
        close(need(ns, "num_grad_np")(q, x), A @ x + np.cos(x), rtol=1e-6, atol=1e-6, what="½xᵀAx + Σ sin xᵢ")
    R.test("6 boyutlu kuadratik + sin", grad_n)

    def jac():
        J = need(ns, "num_jacobian_np")(g, np.array([2.0, 3.0]))
        ensure(np.asarray(J).shape == (2, 2), f"shape {np.asarray(J).shape}, (m, n) = (2, 2) bekleniyordu")
        close(J, [[3, 2], [1, 6]], rtol=1e-6, atol=1e-6)
        W = np.random.default_rng(1).normal(size=(3, 4))
        J2 = need(ns, "num_jacobian_np")(lambda X: X @ W.T, np.ones(4))
        ensure(np.asarray(J2).shape == (3, 4), f"ℝ⁴→ℝ³ için shape {np.asarray(J2).shape}, (3, 4) bekleniyordu")
        close(J2, W, rtol=1e-6, atol=1e-6, what="lineer g(x) = Wx'in Jacobian'ı W")
    R.test("num_jacobian_np: [[3,2],[1,6]] ve lineer haritada J = W", jac)
