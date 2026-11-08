import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("fold_linear", "fold_conv")])

    def t():
        rng = np.random.default_rng(0)
        W, b = rng.normal(size=(6, 4)), rng.normal(size=6)
        g, be, mu, var = rng.random(6) + 0.5, rng.normal(size=6), rng.normal(size=6), rng.random(6) + 0.1
        X = rng.normal(size=(10, 4))
        ref = g * ((X @ W.T + b) - mu) / np.sqrt(var + 1e-5) + be
        W2, b2 = need(ns, "fold_linear")(W, b, g, be, mu, var)
        close(X @ W2.T + b2, ref, atol=1e-10, what="doğrusal + BN = katlanmış doğrusal")
        Wc, bc = rng.normal(size=(5, 3, 3, 3)), rng.normal(size=5)
        g, be, mu, var = rng.random(5) + 0.5, rng.normal(size=5), rng.normal(size=5), rng.random(5) + 0.1
        x = rng.normal(size=(3, 8, 8))
        conv = lambda W_, b_: np.einsum("chwuv,ocuv->ohw", sliding_window_view(x, (3, 3), axis=(1, 2)), W_) + b_[:, None, None]
        ref = g[:, None, None] * (conv(Wc, bc) - mu[:, None, None]) / np.sqrt(var + 1e-5)[:, None, None] + be[:, None, None]
        Wf, bf = need(ns, "fold_conv")(Wc, bc, g, be, mu, var)
        close(conv(Wf, bf), ref, atol=1e-10, what="konv + BN = katlanmış konv")
        R_.note("ResNet-50'de 53 BN katmanı çıkarımda tamamen kaybolur: her biri bir okuma + bir yazma + iki aritmetik işlem demekti (bellek-bağımlı, M7.1 roofline). torch.ao.quantization.fuse_modules ve TensorRT bu katlamayı otomatik yapar; nicemlemeden (M7.5) önce yapılması da gerekir, yoksa BN'nin ölçeği nicemleme aralığını bozar.")
    R_.test("BN katlama", t)
