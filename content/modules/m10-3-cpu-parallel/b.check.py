import random
import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    rng = random.Random(0)

    def scan():
        ensure(need(ns, "block_scan")(list(range(1, 9)), 4) == [1, 3, 6, 10, 15, 21, 28, 36], "el örneği")
        for n, p in ((1000, 7), (13, 4), (5, 8)):
            a = [rng.randint(-50, 50) for _ in range(n)]
            ensure(ns["block_scan"](a, p) == list(np.cumsum(a)), f"n = {n}, p = {p}")
        img = [[rng.randint(0, 255) for _ in range(23)] for _ in range(17)]
        for p in (1, 3, 5):
            ensure(need(ns, "integral_image")(img, p) == np.cumsum(np.cumsum(img, 0), 1).tolist(), f"integral görüntü p = {p}")
    R_.test("taramalar", scan)

    def conv():
        img = [[rng.randint(0, 255) for _ in range(20)] for _ in range(18)]
        g = np.exp(-((np.arange(5) - 2) ** 2) / 2.0)
        k = (np.outer(g, g) / np.outer(g, g).sum()).astype(np.float32).tolist()
        ref = need(ns, "conv_bands")(img, k, 1)
        for p in (2, 3, 7):
            ensure(ns["conv_bands"](img, k, p) == ref, f"bant sayısı sonucu değiştirmemeli (p = {p})")
        f = np.pad(np.array(img, float), 2)
        direct = np.array([[(f[y:y + 5, x:x + 5] * np.array(k)).sum() for x in range(20)] for y in range(18)])
        ensure(np.abs(np.array(ref) - np.clip(np.floor(direct + 0.5), 0, 255)).max() <= 1, "doğrudan konvolüsyonla aynı")
        A = [[rng.random() for _ in range(9)] for _ in range(11)]
        B = [[rng.random() for _ in range(7)] for _ in range(9)]
        close(np.array(need(ns, "matmul_rows")(A, B, 4)), np.array(A) @ np.array(B), what="matmul")
    R_.test("konvolüsyon ve matmul", conv)
