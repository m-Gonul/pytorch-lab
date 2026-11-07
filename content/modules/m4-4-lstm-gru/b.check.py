import numpy as np
from labcheck import close, no_imports, need


def sig(v):
    return 1 / (1 + np.exp(-v))


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    rng = np.random.default_rng(0)
    D, H = 3, 4
    x, h, c = rng.normal(size=D), rng.normal(size=H) * 0.5, rng.normal(size=H)

    def ls():
        Wi, Wh, bi, bh = rng.normal(size=(4 * H, D)), rng.normal(size=(4 * H, H)), rng.normal(size=4 * H), rng.normal(size=4 * H)
        a = Wi @ x + bi + Wh @ h + bh
        i, f, g, o = sig(a[:H]), sig(a[H:2 * H]), np.tanh(a[2 * H:3 * H]), sig(a[3 * H:])
        cn = f * c + i * g
        hn, cc = need(ns, "lstm_step")(x.tolist(), h.tolist(), c.tolist(), Wi.tolist(), Wh.tolist(), bi.tolist(), bh.tolist())
        close(cc, cn, what="c"); close(hn, o * np.tanh(cn), what="h")
    R.test("lstm_step (kapı sırası i, f, g, o)", ls)

    def gr():
        Wi, Wh, bi, bh = rng.normal(size=(3 * H, D)), rng.normal(size=(3 * H, H)), rng.normal(size=3 * H), rng.normal(size=3 * H)
        gi, gh = Wi @ x + bi, Wh @ h + bh
        r, z = sig(gi[:H] + gh[:H]), sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        close(need(ns, "gru_step")(x.tolist(), h.tolist(), Wi.tolist(), Wh.tolist(), bi.tolist(), bh.tolist()), (1 - z) * n + z * h)
    R.test("gru_step (r yalnız W_hn·h + b_hn'yi çarpar)", gr)

    def hand():
        h1, c1 = need(ns, "lstm_step")([0.0], [0.0], [0.5], [[0.0], [0.0], [0.0], [0.0]], [[0.0]] * 4, [0.0, 1.0, 1.0, 0.0], [0.0] * 4)
        close([c1[0], h1[0]], [0.746326, 0.316476], atol=2e-6)
    R.test("elle örnek", hand)
