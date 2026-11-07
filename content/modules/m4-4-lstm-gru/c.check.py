import numpy as np
from labcheck import close, no_loops, no_imports, need


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    N, D, H = 3, 4, 5
    x, h, c = rng.normal(size=(N, D)), rng.normal(size=(N, H)), rng.normal(size=(N, H))
    Wx, Wh, b = rng.normal(size=(D, 4 * H)) * 0.5, rng.normal(size=(H, 4 * H)) * 0.5, rng.normal(size=4 * H) * 0.3
    dh, dc = rng.normal(size=(N, H)), rng.normal(size=(N, H))

    def fw():
        hn, cn, _ = need(ns, "lstm_step_forward")(x, h, c, Wx, Wh, b)
        a = x @ Wx + h @ Wh + b; s = lambda v: 1 / (1 + np.exp(-v))
        cc = s(a[:, H:2 * H]) * c + s(a[:, :H]) * np.tanh(a[:, 2 * H:3 * H])
        close(cn, cc, what="c"); close(hn, s(a[:, 3 * H:]) * np.tanh(cc), what="h")
    R.test("lstm_step_forward", fw)

    def bw():
        _, _, cache = need(ns, "lstm_step_forward")(x, h, c, Wx, Wh, b)
        grads = need(ns, "lstm_step_backward")(dh, dc, cache)
        def f():
            hn, cn, _ = need(ns, "lstm_step_forward")(x, h, c, Wx, Wh, b)
            return (hn * dh).sum() + (cn * dc).sum()
        e = 1e-6
        for arr, g, name in zip((x, h, c, Wx, Wh, b), grads, ("dx", "dprev_h", "dprev_c", "dWx", "dWh", "db")):
            for idx in list(np.ndindex(arr.shape))[::max(1, arr.size // 8)]:
                old = arr[idx]; arr[idx] = old + e; p = f(); arr[idx] = old - e; m = f(); arr[idx] = old
                close(g[idx], (p - m) / (2 * e), rtol=1e-5, atol=1e-7, what=f"{name}{idx}")
    R.test("lstm_step_backward: 6 gradyan sayısal türevle aynı", bw)
