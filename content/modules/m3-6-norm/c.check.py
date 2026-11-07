import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def num(f, x, dout, h=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]; x[idx] = old + h; p = (f(x) * dout).sum(); x[idx] = old - h; m = (f(x) * dout).sum(); x[idx] = old
        g[idx] = (p - m) / (2 * h)
    return g


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, size=(8, 4)); gamma, beta = rng.uniform(0.5, 2, 4), rng.normal(size=4); dout = rng.normal(size=(8, 4))

    def bnf():
        p = {"mode": "train"}
        out, _ = need(ns, "batchnorm_forward")(x, gamma, beta, p)
        close(out.mean(0), beta, atol=1e-9, what="çıktı ortalaması β"); close(out.std(0), gamma * np.sqrt(x.var(0) / (x.var(0) + 1e-5)), rtol=1e-6, what="çıktı std'si ≈ γ")
        close(p["running_mean"], 0.1 * x.mean(0), what="running_mean (momentum 0.9)"); close(p["running_var"], 0.9 + 0.1 * x.var(0), what="running_var")
        p["mode"] = "test"
        out_t, _ = need(ns, "batchnorm_forward")(x, gamma, beta, p)
        close(out_t, gamma * (x - p["running_mean"]) / np.sqrt(p["running_var"] + 1e-5) + beta, what="test modu")
    R.test("batchnorm_forward: train/test ve running istatistikler", bnf)

    def bnb():
        f = lambda xx: need(ns, "batchnorm_forward")(xx, gamma, beta, {"mode": "train"})[0]
        _, cache = need(ns, "batchnorm_forward")(x, gamma, beta, {"mode": "train"})
        dx, dg, db = need(ns, "batchnorm_backward")(dout, cache)
        close(dx, num(f, x.copy(), dout), rtol=1e-5, atol=1e-7, what="dx")
        close(db, dout.sum(0), what="dβ")
    R.test("batchnorm_backward sayısal gradyanla aynı", bnb)

    def ln():
        g2, b2 = rng.uniform(0.5, 2, 4), rng.normal(size=4)
        out, cache = need(ns, "layernorm_forward")(x, g2, b2)
        close(((out - b2) / g2).mean(1), np.zeros(8), atol=1e-9, what="satır ortalaması")
        dx, dg, db = need(ns, "layernorm_backward")(dout, cache)
        close(dx, num(lambda xx: need(ns, "layernorm_forward")(xx, g2, b2)[0], x.copy(), dout), rtol=1e-5, atol=1e-7, what="dx")
        xh = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)
        close(dg, (dout * xh).sum(0), what="dγ")
    R.test("layernorm forward/backward", ln)
