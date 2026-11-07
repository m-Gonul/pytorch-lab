import numpy as np
from labcheck import close, max_loops, no_imports, need


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız zaman döngüsü", lambda: (max_loops(src, 1, func="rnn_forward"), max_loops(src, 1, func="rnn_backward")))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    N, T, D, H = 3, 5, 4, 6
    x = rng.normal(size=(N, T, D)); h0 = rng.normal(size=(N, H)) * 0.5
    Wx, Wh, b = rng.normal(size=(D, H)) * 0.5, rng.normal(size=(H, H)) * 0.5, rng.normal(size=H) * 0.1
    dout = rng.normal(size=(N, T, H))

    def fw():
        h, _ = need(ns, "rnn_forward")(x, h0, Wx, Wh, b)
        prev = h0
        for t in range(T):
            prev = np.tanh(x[:, t] @ Wx + prev @ Wh + b)
            close(h[:, t], prev, what=f"t = {t}")
    R.test("rnn_forward", fw)

    def bw():
        _, cache = need(ns, "rnn_forward")(x, h0, Wx, Wh, b)
        dx, dh0, dWx, dWh, db = need(ns, "rnn_backward")(dout, cache)
        f = lambda: (need(ns, "rnn_forward")(x, h0, Wx, Wh, b)[0] * dout).sum()
        e = 1e-6
        for arr, g, name in ((x, dx, "dx"), (h0, dh0, "dh0"), (Wx, dWx, "dWx"), (Wh, dWh, "dWh"), (b, db, "db")):
            for idx in list(np.ndindex(arr.shape))[::max(1, arr.size // 6)]:
                old = arr[idx]; arr[idx] = old + e; p = f(); arr[idx] = old - e; m = f(); arr[idx] = old
                close(g[idx], (p - m) / (2 * e), rtol=1e-5, atol=1e-7, what=f"{name}{idx}")
    R.test("rnn_backward sayısal gradyanla aynı (tüm gradyanlar)", bw)
