import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def spiral(n=100, K=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.0, 1, n)
        t = np.linspace(k * 4, (k + 1) * 4, n) + rng.normal(size=n) * 0.2
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [k] * n
    return np.concatenate(X), np.array(y)


def check(ns, R):
    src = ns["__source__"]
    for fn in ("affine_forward", "affine_backward", "relu_forward", "relu_backward", "softmax_loss", "two_layer_loss"):
        R.test(f"{fn}: döngü yok", lambda fn=fn: max_loops(src, 0, func=fn))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)

    def aff():
        x, w, b = rng.normal(size=(5, 4)), rng.normal(size=(4, 3)), rng.normal(size=3)
        out, cache = need(ns, "affine_forward")(x, w, b)
        close(out, x @ w + b)
        dout = rng.normal(size=(5, 3))
        dx, dw, db = need(ns, "affine_backward")(dout, cache)
        close(dx, dout @ w.T); close(dw, x.T @ dout); close(db, dout.sum(0))
    R.test("affine forward/backward", aff)

    def grads():
        X, y = rng.normal(size=(7, 3)), rng.integers(0, 4, 7)
        p = {"W1": rng.normal(size=(3, 5)), "b1": rng.normal(size=5) * 0.1, "W2": rng.normal(size=(5, 4)), "b2": rng.normal(size=4) * 0.1}
        L, g = need(ns, "two_layer_loss")(X, y, p, reg=0.05)
        h = 1e-6
        for k in p:
            flat = p[k].reshape(-1)
            for idx in rng.choice(flat.size, min(4, flat.size), replace=False):
                old = flat[idx]
                flat[idx] = old + h; Lp = need(ns, "two_layer_loss")(X, y, p, 0.05)[0]
                flat[idx] = old - h; Lm = need(ns, "two_layer_loss")(X, y, p, 0.05)[0]
                flat[idx] = old
                close(g[k].reshape(-1)[idx], (Lp - Lm) / (2 * h), rtol=1e-4, atol=1e-7, what=f"d{k}[{idx}]")
    R.test("two_layer_loss: sayısal gradyanla aynı (reg dahil)", grads)

    def tr():
        X, y = spiral()
        p = need(ns, "train")(X, y, 100, 3)
        h = np.maximum(0, X @ p["W1"] + p["b1"])
        acc = ((h @ p["W2"] + p["b2"]).argmax(1) == y).mean()
        ensure(acc >= 0.95, f"spiral doğruluğu {acc:.2f}")
        return f"spiral (3 kol): eğitim doğruluğu {acc:.3f}"
    R.test("spiral veride eğitim ≥ 0.95", tr)
