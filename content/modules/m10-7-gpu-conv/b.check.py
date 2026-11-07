import random
import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    rng = random.Random(0)

    def conv():
        ensure(need(ns, "conv_tile_loads")(16, 5) == (400, 1.5625), "el hesabı")
        img = [[rng.randint(0, 255) for _ in range(40)] for _ in range(36)]
        rows = []
        for K in (3, 7, 11):
            k = [[rng.random() for _ in range(K)] for _ in range(K)]
            d, rd = need(ns, "direct_conv")(img, k)
            t, rt = need(ns, "tiled_conv")(img, k, 8)
            close(np.array(t), np.array(d), atol=1e-9, what=f"K = {K}: karolu = doğrudan")
            rows.append([K, rd, rt, f"{rd / rt:.1f}", f"{K * K / ns['conv_tile_loads'](8, K)[1]:.1f}"])
        R_.table("36×40 görüntü, 8×8 çıktı karoları: global okuma", ["K", "doğrudan", "karolu", "oran (ölçülen)", "oran (iç karo formülü)"], rows,
                 note="Ölçülen oran formülden biraz düşük: kenar karolarında görüntü dışı (sıfır) halo okunmuyor ve kenar çıktıları da daha az okuyor. Büyük K'de karolama vazgeçilmez.")
    R_.test("karolu konvolüsyon", conv)

    def mm():
        A = [[rng.random() for _ in range(37)] for _ in range(29)]
        B = [[rng.random() for _ in range(23)] for _ in range(37)]
        for T in (1, 4, 8):
            C, reads = need(ns, "tiled_matmul")(A, B, T)
            close(np.array(C), np.array(A) @ np.array(B), what=f"T = {T}")
        n = 32
        A2 = [[1.0] * n for _ in range(n)]
        r1 = ns["tiled_matmul"](A2, A2, 1)[1]
        r8 = ns["tiled_matmul"](A2, A2, 8)[1]
        ensure(r1 == 2 * n ** 3 and r8 == 2 * n ** 3 // 8, "global okuma 2n³/T")
    R_.test("karolu matmul", mm)
