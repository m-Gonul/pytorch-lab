import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    tanh, exp = need(ns, "tanh"), need(ns, "exp")

    def f(x, y):
        a = tanh(x * y)
        b = tanh(y * x)
        unused = exp(x)
        return a + b * 2 - 1

    def tr():
        g = need(ns, "trace")(f, 2)
        ops = [op for _, op, _ in g.nodes]
        ensure(ops.count("input") == 2 and ops.count("tanh") == 2 and ops.count("exp") == 1, f"düğümler {ops}")
        close(need(ns, "run")(g, [0.3, 0.5]), f(0.3, 0.5))
        close(ns["run"](g, [-1.2, 0.7]), f(-1.2, 0.7))
    R.test("trace + run = Python fonksiyonu", tr)

    def cs():
        g = need(ns, "cse")(ns["trace"](f, 2))
        ops = [op for _, op, _ in g.nodes]
        ensure(ops.count("tanh") == 1 and ops.count("mul") == 2, f"CSE sonrası {ops}: x*y ve y*x tek düğüm olmalı")
        close(ns["run"](g, [0.3, 0.5]), f(0.3, 0.5))
    R.test("cse (değişmeli işlemler dahil)", cs)

    def dc():
        g = need(ns, "dce")(ns["cse"](ns["trace"](f, 2)))
        ops = [op for _, op, _ in g.nodes]
        ensure("exp" not in ops and ops.count("input") == 2, f"DCE sonrası {ops}")
        close(ns["run"](g, [0.3, 0.5]), f(0.3, 0.5))
        before = len(ns["trace"](f, 2).nodes)
        return f"{before} düğüm → {len(g.nodes)} düğüm"
    R.test("dce", dc)

    def sub():
        g = need(ns, "cse")(ns["trace"](lambda x, y: (x - y) * (y - x), 2))
        close(ns["run"](g, [3.0, 1.0]), -4.0, what="sub değişmeli değil")
    R.test("sub değişmeli sayılmıyor", sub)
