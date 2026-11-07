import numpy as np
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def direct(img, k):
    K = k.shape[0]
    r = K // 2
    f = np.pad(img.astype(np.float64), r)
    H, W = img.shape
    return np.array([[(f[y:y + K, x:x + K] * k).sum() for x in range(W)] for y in range(H)])


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("im2col", "conv_im2col", "conv_multi", "conv_fft", "tile_intensity")])
    rng = np.random.default_rng(0)

    def t():
        img = rng.integers(0, 256, (30, 34)).astype(np.float32)
        for K in (3, 7):
            k = rng.random((K, K)).astype(np.float32)
            ref = direct(img, k)
            ensure(need(ns, "im2col")(img, K).shape == (30 * 34, K * K), "im2col şekli")
            close(need(ns, "conv_im2col")(img, k), ref, rtol=1e-5, what=f"im2col K = {K}")
            close(need(ns, "conv_fft")(img, k), ref, rtol=1e-6, atol=1e-6, what=f"FFT K = {K}")
        x = rng.normal(size=(3, 12, 13)).astype(np.float32)
        w = rng.normal(size=(5, 3, 3, 3)).astype(np.float32)
        ref = np.stack([sum(direct(x[c], w[o, c]) for c in range(3)) for o in range(5)])
        close(need(ns, "conv_multi")(x, w), ref, atol=1e-4, what="çok kanallı konvolüsyon = GEMM")
        close(need(ns, "tile_intensity")(32, 32, 32), 8.0, what="yoğunluk")
        big = rng.integers(0, 256, (256, 256)).astype(np.float32)
        rows = []
        for K in (3, 11, 31):
            k = rng.random((K, K)).astype(np.float32)
            rows.append([K, fmt_time(bench(lambda: ns["conv_im2col"](big, k), repeat=2, number=1)), fmt_time(bench(lambda: ns["conv_fft"](big, k), repeat=2, number=1))])
        R_.table("256×256: im2col + matmul vs FFT", ["K", "im2col", "FFT"], rows,
                 note="im2col maliyeti K² ile büyür; FFT neredeyse K'den bağımsız. Derin ağlardaki 3×3'ler için GEMM/Winograd, sinyal işlemedeki büyük çekirdekler için FFT.")
    R_.test("konvolüsyon = matmul", t)
