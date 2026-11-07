import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def gr():
        f = lambda th, ps: -math.log(1 + math.exp(-ps * th))
        h = 1e-6
        for th, ps in ((1.0, 1.0), (0.3, -2.0), (-1.5, 0.7)):
            g = need(ns, "grads")(th, ps)
            close(g, [(f(th + h, ps) - f(th - h, ps)) / (2 * h), (f(th, ps + h) - f(th, ps - h)) / (2 * h)], atol=1e-6, what=f"({th}, {ps})")
    R.test("grads = sayısal türev", gr)

    def one():
        th, ps = need(ns, "simulate")(1.0, 1.0, 0.1, 1)
        s = 1 / (1 + math.exp(1.0))
        close([th, ps], [1 - 0.1 * s, 1 + 0.1 * s])
        th, ps = ns["simulate"](1.0, 1.0, 0.1, 1, alternating=True)
        th1 = 1 - 0.1 * s
        close([th, ps], [th1, 1 + 0.1 * th1 / (1 + math.exp(th1))], what="alternatif adım")
    R.test("tek adım (eşzamanlı ve alternatif)", one)

    def dyn():
        r = lambda t: math.hypot(*t)
        sim = r(need(ns, "simulate")(1.0, 1.0, 0.1, 3000))
        alt = r(ns["simulate"](1.0, 1.0, 0.1, 3000, alternating=True))
        reg = r(ns["simulate"](1.0, 1.0, 0.1, 3000, gamma=1.0))
        ensure(sim > 2.5, f"eşzamanlı GD uzaklaşmalı: yarıçap {sim:.3f}")
        ensure(0.5 < alt < 2.5, f"alternatif sınırlı kalmalı: {alt:.3f}")
        ensure(reg < 1e-3, f"R1 ile yakınsamalı: {reg:.2e}")
        return f"başlangıç yarıçapı 1.414 → eşzamanlı {sim:.3f}, alternatif {alt:.3f}, R1 {reg:.1e}"
    R.test("üç dinamik", dyn)
