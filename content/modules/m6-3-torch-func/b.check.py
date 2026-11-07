import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    D = need(ns, "Dual", "sınıf")

    def ops():
        a, b = D(2.0, 1.0), D(3.0, 0.5)
        p = a * b + 2 * a - b * 1.5 + 1
        close([p.val, p.dot], [6 + 4 - 4.5 + 1, 1 * 3 + 2 * 0.5 + 2 - 0.75])
    R.test("Dual aritmetiği (sağdan sayılar dahil)", ops)

    def fns():
        d = need(ns, "dsin")(D(0.5, 2.0)); e = need(ns, "dexp")(D(0.5, 2.0))
        close([d.val, d.dot, e.val, e.dot], [math.sin(0.5), 2 * math.cos(0.5), math.exp(0.5), 2 * math.exp(0.5)])
    R.test("dsin, dexp", fns)
    R.test("jvp (elle örnek)", lambda: close(need(ns, "jvp")(lambda x: [x[0] * x[1], x[0] * x[0]], [1.0, 2.0], [1.0, 1.0]), [[2, 1], [3, 2]]))

    def jac():
        sin, exp = ns["dsin"], ns["dexp"]
        f = lambda x: [sin(x[0] * x[1]) + x[2], exp(x[0] - x[2]) * x[1]]
        x = [0.3, -1.2, 0.7]
        J = need(ns, "jacobian_fwd")(f, x)
        g = lambda x: [math.sin(x[0] * x[1]) + x[2], math.exp(x[0] - x[2]) * x[1]]
        h = 1e-6
        num = [[(g(x[:j] + [x[j] + h] + x[j + 1:])[i] - g(x[:j] + [x[j] - h] + x[j + 1:])[i]) / (2 * h) for j in range(3)] for i in range(2)]
        close(J, num, atol=1e-6)
    R.test("jacobian_fwd = sayısal Jacobian (2 × 3)", jac)
