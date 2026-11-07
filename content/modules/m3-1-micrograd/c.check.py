import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))

    def hand():
        o, L, dw, db = need(ns, "neuron_fb")(np.array([[2.0, 0.0]]), np.array([-3.0, 1.0]), 6.8813735870195432)
        close(o, [2 ** -0.5]); close(dw, [1.0, 0.0], atol=1e-9); close(db, 0.5)
    R.test("elle nöron: o = 1/√2, ∂w = [1, 0], ∂b = 0.5", hand)

    def rnd():
        rng = np.random.default_rng(0)
        X, w, b = rng.normal(size=(50, 4)), rng.normal(size=4), 0.3
        o, L, dw, db = need(ns, "neuron_fb")(X, w, b)
        h = 1e-6
        f = lambda w_, b_: np.tanh(X @ w_ + b_).sum()
        E = h * np.eye(4)
        num_w = np.array([(f(w + E[j], b) - f(w - E[j], b)) / (2 * h) for j in range(4)])
        num_b = (f(w, b + h) - f(w, b - h)) / (2 * h)
        close(L, f(w, b), what="L"); close(dw, num_w, rtol=1e-6, atol=1e-6, what="dw"); close(db, num_b, rtol=1e-6, atol=1e-6, what="db")
        ensure(np.asarray(o).shape == (50,), f"o shape {np.asarray(o).shape}")
    R.test("50 örnek: sayısal gradyanla eşleşiyor", rnd)
