import numpy as np
from labcheck import close, max_loops, need, ensure


def _tconv(x, w, s):
    Ci, H, W = x.shape
    _, Co, k, _ = w.shape
    out = np.zeros((Co, (H - 1) * s + k, (W - 1) * s + k))
    for i in range(H):
        for j in range(W):
            out[:, i * s:i * s + k, j * s:j * s + k] += np.einsum("c,couv->ouv", x[:, i, j], w)
    return out


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("pixel_shuffle", "pixel_unshuffle", "subpixel_to_tconv")])

    def t():
        rng = np.random.default_rng(0)
        x = np.arange(2 * 4 * 3 * 3.).reshape(8, 3, 3)
        y = need(ns, "pixel_shuffle")(x, 2)
        ensure(y.shape == (2, 6, 6), "şekil")
        ensure(y[1, 2 * 1 + 1, 2 * 2 + 0] == x[1 * 4 + 1 * 2 + 0, 1, 2], "kanal c·r² + a·r + b → (h·r + a, w·r + b)")
        close(need(ns, "pixel_unshuffle")(y, 2), x, what="ters")
        for r in (2, 3):
            Ci, Co = 5, 3
            w = rng.normal(size=(Co * r * r, Ci))
            xin = rng.normal(size=(Ci, 4, 5))
            sub = ns["pixel_shuffle"](np.einsum("oc,chw->ohw", w, xin), r)
            close(_tconv(xin, need(ns, "subpixel_to_tconv")(w, r), r), sub, atol=1e-10, what=f"1×1 konv. + PixelShuffle = tconv(k = s = {r})")
        R_.note("Shi ve ark. (ESPCN): tüm konvolüsyonlar düşük çözünürlükte yapılır, en sonda kanallar uzaya 'açılır'. 1×1 durumda bu, k = s = r'lik transposed conv'un ta kendisidir (örtüşme yok → dama tahtası yok); k > 1'li alt-piksel konvolüsyonu, düşük çözünürlükte k, yüksekte r·k'lik bir tconv'a karşılık gelir ama r² kat daha ucuzdur.")
    R_.test("eşdeğerlik", t)
