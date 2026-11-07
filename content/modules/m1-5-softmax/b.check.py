import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    e = [math.exp(2), math.exp(1), 1.0]
    ref = [v / sum(e) for v in e]
    R.test("softmax([2,1,0])", lambda: close(need(ns, "softmax")([2, 1, 0]), ref))
    R.test("cross_entropy([2,1,0], 0) = 0.4076", lambda: close(need(ns, "cross_entropy")([2, 1, 0], 0), math.log(sum(e)) - 2))

    def stable():
        close(need(ns, "softmax")([1000, 999, 998]), ref, what="z + 998")
        close(need(ns, "logsumexp")([1000, 999, 998]), 998 + math.log(sum(e)), what="LSE")
        close(need(ns, "softmax")([-1000, -1001]), [1 / (1 + math.exp(-1)), 1 / (1 + math.e)], what="büyük negatif")
        close(need(ns, "cross_entropy")([0, 800], 0), 800.0, what="CE, doğru sınıf çok gerideyken")
    R.test("taşma yok (±1000)", stable)

    def grad():
        rng = random.Random(0)
        for _ in range(5):
            z = [rng.uniform(-3, 3) for _ in range(4)]
            y = rng.randrange(4)
            g = need(ns, "grad_z")(z, y)
            h = 1e-6
            num = []
            for k in range(4):
                zp, zm = list(z), list(z)
                zp[k] += h; zm[k] -= h
                num.append((need(ns, "cross_entropy")(zp, y) - need(ns, "cross_entropy")(zm, y)) / (2 * h))
            close(g, num, rtol=1e-5, atol=1e-7)
            ensure(abs(sum(g)) < 1e-9, "gradyan bileşenleri toplamı 0 olmalı")
    R.test("grad_z = p − e_y, sayısal türevle aynı, toplamı 0", grad)
