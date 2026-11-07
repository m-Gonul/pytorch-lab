import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def fb():
        out = need(ns, "f_backward")(2.0, 3.0)
        ensure(isinstance(out, (tuple, list)) and len(out) == 3, "(out, dfdx, dfdy) üçlüsü döndürmeli")
        close(out, [15, 3, 8], what="(f, ∂f/∂x, ∂f/∂y)")
    R.test("elle örnek: (15, 3, 8)", fb)

    def fb_rand():
        rng = random.Random(2)
        for _ in range(10):
            x, y = rng.uniform(-4, 4), rng.uniform(-4, 4)
            close(need(ns, "f_backward")(x, y), [(x + y) * y, y, x + 2 * y], what=f"(x, y) = ({x:.2f}, {y:.2f})")
    R.test("rastgele noktalar: ∂f/∂x = y, ∂f/∂y = x + 2y", fb_rand)

    def ng():
        g = need(ns, "num_grad")
        close(g(lambda x, y: (x + y) * y, [2.0, 3.0]), [3, 8], rtol=1e-6, atol=1e-6)
        fn = lambda a, b, c: math.sin(a) * b ** 2 + math.exp(c) * a
        xs = [0.7, -1.3, 0.2]
        want = [math.cos(0.7) * 1.69 + math.exp(0.2), 2 * math.sin(0.7) * -1.3, math.exp(0.2) * 0.7]
        close(g(fn, xs), want, rtol=1e-6, atol=1e-6, what="sin(a)·b² + eᶜ·a")
        ensure(xs == [0.7, -1.3, 0.2], "num_grad girdi listesini değiştirdi")
    R.test("num_grad: polinom ve sin/exp fonksiyonu, girdi değişmiyor", ng)
