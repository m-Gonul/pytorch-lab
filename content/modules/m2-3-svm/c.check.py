import numpy as np
from labcheck import close, no_loops, no_imports, need


def naive(W, X, y, reg, delta=1.0):
    n, C = X.shape[0], W.shape[1]
    loss, dW = 0.0, np.zeros_like(W)
    for i in range(n):
        s = X[i] @ W
        for j in range(C):
            if j == y[i]:
                continue
            m = s[j] - s[y[i]] + delta
            if m > 0:
                loss += m; dW[:, j] += X[i]; dW[:, y[i]] -= X[i]
    return loss / n + reg * (W * W).sum(), dW / n + 2 * reg * W


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(1)
    W = rng.normal(size=(8, 4)) * 0.3
    X = rng.normal(size=(30, 8))
    y = rng.integers(0, 4, 30)

    def eq():
        L, dW = need(ns, "svm_loss")(W, X, y, reg=0.05)
        Ln, dWn = naive(W, X, y, 0.05)
        close(L, Ln, what="kayıp"); close(dW, dWn, what="dW")
    R.test("naive (çift döngü) ile aynı", eq)

    def num():
        h = 1e-6
        f = lambda Wp: need(ns, "svm_loss")(Wp, X, y, 0.05)[0]
        _, dW = need(ns, "svm_loss")(W, X, y, 0.05)
        for (i, j) in [(0, 0), (3, 2), (7, 1)]:
            E = np.zeros_like(W); E[i, j] = h
            close(dW[i, j], (f(W + E) - f(W - E)) / (2 * h), rtol=1e-4, atol=1e-6, what=f"dW[{i},{j}]")
    R.test("sayısal gradyanla aynı", num)
    R.test("elle örnek: 2.9", lambda: close(need(ns, "svm_loss")(np.eye(3), np.array([[3.2, 5.1, -1.7]]), np.array([0]))[0], 2.9))
