import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    pts = [-3.0, -1.2, -0.3, 0.4, 1.0, 2.5]
    for name in ("relu", "leaky_relu", "sigmoid", "tanh", "gelu", "silu"):
        def t(name=name):
            f, df = need(ns, name), need(ns, "d_" + name)
            h = 1e-6
            close([df(x) for x in pts], [(f(x + h) - f(x - h)) / (2 * h) for x in pts], rtol=1e-5, atol=1e-7, what=f"d_{name}")
        R.test(f"d_{name} sayısal türevle aynı", t)
    R.test("değerler: GELU(1) = 0.8412, SiLU(−2) = −0.2384, sigmoid(−800) = 0",
           lambda: close([need(ns, "gelu")(1.0), need(ns, "silu")(-2.0), need(ns, "sigmoid")(-800.0)], [0.8411919906082768, -2 / (1 + math.e ** 2), 0.0], rtol=1e-7, atol=1e-300))
