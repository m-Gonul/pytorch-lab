import numpy as np
from labcheck import close, no_imports, need, ensure


def ref(Q, K, V, causal):
    Q, K, V = map(np.array, (Q, K, V))
    S = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        S = np.where(np.triu(np.ones(S.shape), 1) > 0, -np.inf, S)
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    return A @ V


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    R.test("softmax_row: büyük değer ve -inf", lambda: close(need(ns, "softmax_row")([1000.0, 1000.0, float("-inf")]), [0.5, 0.5, 0.0]))
    Q = K = [[1, 0], [0, 1], [1, 1]]; V = [[1, 2], [3, 4], [5, 6]]
    R.test("elle örnek", lambda: close(need(ns, "attention")(Q, K, V), ref(Q, K, V, False)))
    R.test("elle örnek, causal", lambda: close(need(ns, "attention")(Q, K, V, causal=True), ref(Q, K, V, True)))

    def rnd():
        rng = np.random.default_rng(0)
        Q, K, V = rng.normal(size=(5, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
        for c in (False, True):
            close(need(ns, "attention")(Q.tolist(), K.tolist(), V.tolist(), causal=c), ref(Q, K, V, c), what=f"causal={c}")
    R.test("rastgele 5×4, d_v = 3", rnd)
