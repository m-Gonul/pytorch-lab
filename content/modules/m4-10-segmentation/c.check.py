import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("conv_matrix", "tconv2d", "tconv_zero_insert", "bilinear_kernel", "confusion", "miou")])
    rng = np.random.default_rng(0)

    def t_eq():
        for (H, k, s, p) in ((5, 3, 1, 0), (6, 3, 2, 1), (5, 4, 2, 1), (7, 5, 3, 2)):
            w = rng.normal(size=(k, k))
            M = need(ns, "conv_matrix")(w, H, s, p)
            x = rng.normal(size=(H, H))
            xp = np.pad(x, p)
            Ho = (H + 2 * p - k) // s + 1
            ref = np.array([[(xp[i * s:i * s + k, j * s:j * s + k] * w).sum() for j in range(Ho)] for i in range(Ho)])
            close(M @ x.ravel(), ref.ravel(), atol=1e-10, what=f"C·x = konv. (H={H}, k={k}, s={s}, p={p})")
            y = rng.normal(size=(Ho, Ho))
            T = (Ho - 1) * s + k - 2 * p
            close(need(ns, "tconv2d")(y[None], w[None, None], s, p)[0], (M.T @ y.ravel()).reshape(H, H)[:T, :T], atol=1e-10, what="transposed conv = Cᵀ·y")
        x = rng.normal(size=(3, 5, 6))
        w = rng.normal(size=(3, 4, 3, 3))
        for s, p in ((1, 0), (2, 0), (2, 1), (3, 2)):
            a = ns["tconv2d"](x, w, s, p)
            b = need(ns, "tconv_zero_insert")(x, w, s, p)
            ensure(a.shape == (4, (5 - 1) * s + 3 - 2 * p, (6 - 1) * s + 3 - 2 * p), "çıkış şekli")
            close(a, b, atol=1e-10, what=f"scatter = sıfır ekleme + konv. (s={s}, p={p})")
    R_.test("dört görünüm", t_eq)

    def t_bil():
        K = need(ns, "bilinear_kernel")(2)
        close(K[1, 1], 0.5625, what="merkez")
        ramp = np.add.outer(np.arange(8.), 2 * np.arange(8.))[None]
        for f in (2, 4):
            Kf = ns["bilinear_kernel"](f)
            k = Kf.shape[0]
            up = ns["tconv2d"](ramp, Kf[None, None], f, (k - f) // 2 if k % 2 == 0 else (k - f + 1) // 2)
            inner = up[0, f:-f, f:-f]
            dr, dc = np.diff(inner, axis=0), np.diff(inner, axis=1)
            ensure(np.allclose(dr, dr[0, 0]) and np.allclose(dc, dc[0, 0]), f"×{f} bilineer: doğrusal rampa doğrusal kalır")
            close(dc[0, 0] * f, 2.0, what="eğim korunur")
        pred = rng.integers(0, 3, (20, 20))
        gt = rng.integers(0, 3, (20, 20))
        cm = need(ns, "confusion")(pred, gt, 3)
        ensure(cm.sum() == 400 and cm[1, 2] == ((gt == 1) & (pred == 2)).sum(), "karışıklık matrisi (satır gerçek)")
        iou, m = need(ns, "miou")(cm)
        close(iou[0], ((gt == 0) & (pred == 0)).sum() / ((gt == 0) | (pred == 0)).sum(), what="sınıf IoU")
    R_.test("bilineer çekirdek ve mIoU", t_bil)
