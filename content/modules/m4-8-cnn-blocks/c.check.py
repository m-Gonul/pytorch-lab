import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("conv2d", "depthwise", "pointwise", "grouped_conv", "se_block", "lrn")])
    rng = np.random.default_rng(0)

    def eq():
        x = rng.normal(size=(8, 9, 10))
        dw = rng.normal(size=(8, 3, 3))
        pw = rng.normal(size=(6, 8))
        full = np.zeros((8, 8, 3, 3))
        full[np.arange(8), np.arange(8)] = dw
        close(need(ns, "depthwise")(x, dw), need(ns, "conv2d")(x, full), atol=1e-10, what="derinlemesine = köşegen tam konv.")
        sep = ns["pointwise"](ns["depthwise"](x, dw), pw)
        close(sep, ns["conv2d"](x, np.einsum("oc,cuv->ocuv", pw, dw)), atol=1e-10, what="ayrılabilir = rank-yapılı tam konv.")
        close(need(ns, "pointwise")(x, pw), ns["conv2d"](x, pw[:, :, None, None]), atol=1e-10, what="1×1 konv. = noktasal")
        wg = rng.normal(size=(8, 2, 3, 3))
        g = need(ns, "grouped_conv")(x, wg, 4)
        blk = np.zeros((8, 8, 3, 3))
        for gi in range(4):
            blk[2 * gi:2 * gi + 2, 2 * gi:2 * gi + 2] = wg[2 * gi:2 * gi + 2]
        close(g, ns["conv2d"](x, blk), atol=1e-10, what="grup konv. = blok köşegen tam konv.")
        close(ns["grouped_conv"](x, dw[:, None], 8), ns["depthwise"](x, dw), atol=1e-10, what="g = C → derinlemesine")
    R_.test("eşdeğerlikler", eq)

    def se_lrn():
        x = rng.normal(size=(16, 5, 5))
        W1, b1, W2, b2 = rng.normal(size=(4, 16)), rng.normal(size=4), rng.normal(size=(16, 4)), rng.normal(size=16)
        y = need(ns, "se_block")(x, W1, b1, W2, b2)
        r = y / x
        ensure(np.allclose(r, r[:, :1, :1]) and (r > 0).all() and (r < 1).all(), "SE kanal başına (0, 1) ölçek")
        a = rng.normal(size=(7, 3, 3)) * 5
        ref = np.stack([a[i] / (2 + 1e-4 * (a[max(0, i - 2):i + 3] ** 2).sum(0)) ** 0.75 for i in range(7)])
        close(need(ns, "lrn")(a), ref, rtol=1e-10, what="LRN")
    R_.test("SE ve LRN", se_lrn)
