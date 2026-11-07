import math
import random
from labcheck import close, no_imports, need


def raises(f, exc=ValueError):
    try:
        f()
    except exc:
        return
    raise AssertionError(f"{exc.__name__} bekleniyordu")


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "pandas"}))
    for n in ("dot", "norm", "cosine"):
        R.test(f"`{n}` tanımlı", lambda n=n: need(ns, n))
    if not all(n in ns for n in ("dot", "norm", "cosine")):
        return
    dot, norm, cosine = ns["dot"], ns["norm"], ns["cosine"]
    R.test("elle örnek: u·v = 12", lambda: close(dot([1, 2, 3], [4, -5, 6]), 12))
    R.test("‖u‖ = √14", lambda: close(norm([1, 2, 3]), math.sqrt(14)))
    R.test("cos θ = 12/√1078", lambda: close(cosine([1, 2, 3], [4, -5, 6]), 12 / math.sqrt(1078)))

    def rnd():
        rng = random.Random(1)
        for d in (1, 5, 17):
            u = [rng.uniform(-3, 3) for _ in range(d)]
            v = [rng.uniform(-3, 3) for _ in range(d)]
            close(dot(u, v), sum(a * b for a, b in zip(u, v)), what=f"d={d} dot")
            c = cosine(u, v)
            assert -1 - 1e-12 <= c <= 1 + 1e-12, f"kosinüs [-1,1] dışında: {c}"
    R.test("rastgele vektörler (d = 1, 5, 17)", rnd)
    R.test("kendisiyle kosinüs = 1, zıttıyla = −1", lambda: (close(cosine([2, 1], [4, 2]), 1), close(cosine([2, 1], [-2, -1]), -1)))
    R.test("farklı uzunlukta ValueError", lambda: raises(lambda: dot([1, 2], [1, 2, 3])))
    R.test("sıfır vektörde ValueError", lambda: raises(lambda: cosine([0, 0], [1, 2])))
