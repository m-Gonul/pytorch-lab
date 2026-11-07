import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def sym():
        q, s = need(ns, "quantize_sym")([0.5, -1.2, 2.54])
        ensure(list(q) == [25, -60, 127], f"q = {q}")
        close(s, 0.02)
        q, s = ns["quantize_sym"]([0.0, 0.0])
        ensure(list(q) == [0, 0], "sıfır vektör")
        q, s = ns["quantize_sym"]([1.0, -0.3], bits=4)
        ensure(list(q) == [7, -2], f"4 bit: {q}")
    R.test("quantize_sym", sym)

    def aff():
        q, s, z = need(ns, "quantize_affine")([-1.0, 0.5, 3.0])
        ensure(list(q) == [0, 96, 255] and z == 64, f"q = {q}, z = {z}")
        close(s, 4 / 255)
        q, s, z = ns["quantize_affine"]([0.2, 0.9])
        ensure(z == 0 and q[0] > 0, "aralık 0'ı içerecek şekilde genişlemeli (min → 0)")
        close(ns["dequantize"](*ns["quantize_affine"]([0.0, 0.7, 1.4])), [0.0, 0.7, 1.4], atol=1.4 / 255)
    R.test("quantize_affine (0 tam temsil edilir)", aff)

    def err():
        rng = random.Random(0)
        xs = [rng.gauss(0, 1) for _ in range(1000)]
        q, s = ns["quantize_sym"](xs)
        d = need(ns, "dequantize")(q, s)
        ensure(max(abs(a - b) for a, b in zip(xs, d)) <= s / 2 + 1e-12, "hata ≤ s/2")
    R.test("yuvarlama hatası ≤ s/2", err)

    def pr():
        ensure(list(need(ns, "magnitude_prune")([0.1, -2, 0.3, 1.5], 0.5)) == [0, -2, 0, 1.5], "elle örnek")
        ensure(list(ns["magnitude_prune"]([1, -1, 1, 2], 0.5)) == [0, 0, 1, 2], "eşitlikte küçük indeks önce")
        w = [random.Random(1).uniform(-1, 1) for _ in range(100)]
        ensure(sum(v == 0 for v in ns["magnitude_prune"](w, 0.3)) == 30, "%30 seyreklik")
    R.test("magnitude_prune", pr)
