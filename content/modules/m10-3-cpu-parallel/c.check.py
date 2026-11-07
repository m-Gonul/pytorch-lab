import numpy as np
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("scan_blocks", "integral_image", "conv2d", "conv2d_separable", "matmul_tiled")])
    rng = np.random.default_rng(0)

    def t():
        a = rng.integers(-100, 100, 4096)
        close(need(ns, "scan_blocks")(a, 8), np.cumsum(a), what="blok taraması")
        img = rng.integers(0, 256, (64, 80)).astype(np.uint8)
        close(need(ns, "integral_image")(img), np.cumsum(np.cumsum(img.astype(np.int64), 0), 1), what="integral görüntü")
        g = np.exp(-((np.arange(7) - 3) ** 2) / 4.0).astype(np.float32)
        g /= g.sum()
        K = np.outer(g, g).astype(np.float32)
        c2 = need(ns, "conv2d")(img, K)
        f = np.pad(img.astype(np.float64), 3)
        direct = np.array([[(f[y:y + 7, x:x + 7] * K).sum() for x in range(80)] for y in range(64)])
        ensure(c2.dtype == np.uint8 and np.abs(c2.astype(int) - np.clip(np.floor(direct + 0.5), 0, 255)).max() <= 1, "2B konvolüsyon")
        cs = need(ns, "conv2d_separable")(img, g, g)
        ensure(np.abs(cs.astype(int) - c2.astype(int)).max() <= 1, "ayrılabilir = 2B (yuvarlama payı)")
        A = rng.random((96, 96)).astype(np.float32)
        B = rng.random((96, 96)).astype(np.float32)
        close(need(ns, "matmul_tiled")(A, B, 16), A @ B, rtol=1e-4, what="karolu matmul")
        big = rng.integers(0, 256, (256, 256)).astype(np.uint8)
        g21 = np.exp(-((np.arange(21) - 10) ** 2) / 50.0).astype(np.float32)
        g21 /= g21.sum()
        t2 = bench(lambda: ns["conv2d"](big, np.outer(g21, g21).astype(np.float32)), repeat=2, number=1)
        t1 = bench(lambda: ns["conv2d_separable"](big, g21, g21), repeat=2, number=1)
        R_.table("256×256, 21×21 Gauss", ["yöntem", "süre"], [["2B (441 çarpım/piksel)", fmt_time(t2)], ["ayrılabilir (42 çarpım/piksel)", fmt_time(t1)]])
    R_.test("vektörel çekirdekler", t)
