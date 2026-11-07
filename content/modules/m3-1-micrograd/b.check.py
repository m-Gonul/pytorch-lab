import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    R.test("`Value` sınıfı tanımlı", lambda: need(ns, "Value", "sınıf"))
    V = ns.get("Value")
    if V is None:
        return

    def hand():
        a, b, c, f = V(2), V(-3), V(10), V(-2)
        L = (a * b + c) * f
        L.backward()
        close([L.data, a.grad, b.grad, c.grad, f.grad], [-8, 6, -4, -2, 4])
    R.test("elle örnek: L = −8; ∂a = 6, ∂b = −4, ∂c = −2, ∂f = 4", hand)

    def neuron():
        x1, x2, w1, w2, b = V(2), V(0), V(-3), V(1), V(6.8813735870195432)
        o = (x1 * w1 + x2 * w2 + b).tanh()
        o.backward()
        close([o.data, w1.grad, w2.grad, x1.grad, x2.grad], [2 ** -0.5, 1.0, 0.0, -1.5, 0.5], atol=1e-6)
    R.test("tanh nöronu: o = 1/√2, ∂w₁ = 1, ∂x₁ = −1.5", neuron)

    def reuse():
        a = V(3.0)
        b = a + a
        b.backward()
        ensure(abs(a.grad - 2) < 1e-12, f"a + a için ∂/∂a = {a.grad}, 2 bekleniyordu (+= yerine = mi yazdın?)")
        a2 = V(-4.0)
        c = a2 * a2
        c.backward()
        ensure(abs(a2.grad + 8) < 1e-12, f"a·a için ∂/∂a = {a2.grad}, 2a = −8 bekleniyordu")
    R.test("paylaşılan düğümde gradyan birikiyor (a + a, a·a)", reuse)

    def deep():
        def f(x, y):
            z = x * y + x ** 3 - y / 2.5
            q = (z.tanh() * 2 + 1) ** 2
            return q - 3 * x + (1 - y) * z
        def fv(x, y):
            z = x * y + x ** 3 - y / 2.5
            q = (math.tanh(z) * 2 + 1) ** 2
            return q - 3 * x + (1 - y) * z
        for x0, y0 in [(0.5, -0.3), (-0.8, 1.1), (0.1, 0.2)]:
            x, y = V(x0), V(y0)
            out = f(x, y)
            out.backward()
            h = 1e-6
            nx = (fv(x0 + h, y0) - fv(x0 - h, y0)) / (2 * h)
            ny = (fv(x0, y0 + h) - fv(x0, y0 - h)) / (2 * h)
            close([out.data, x.grad, y.grad], [fv(x0, y0), nx, ny], rtol=1e-5, atol=1e-6, what=f"({x0}, {y0})")
        return "sayısal türevle 6 basamak eşleşti"
    R.test("uzun ifade (−, /, **, r-işlemleri) sayısal türevle eşleşiyor", deep)
