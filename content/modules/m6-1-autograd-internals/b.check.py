import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    g = {"a": ("leaf", []), "b": ("leaf", []), "c": ("mul", ["a", "b"]), "d": ("add", ["c", "a"]), "L": ("square", ["d"])}

    def fw():
        v = need(ns, "forward")(g, {"a": 2.0, "b": 3.0})
        ensure(v["L"] == 64 and v["d"] == 8, f"değerler {v}")
    R.test("forward", fw)

    def ts():
        order = need(ns, "toposort")(g, "L")
        ensure(order[-1] == "L" and set(order) == set(g), f"sıra {order}")
        for n, (_, ins) in g.items():
            for i in ins:
                ensure(order.index(i) < order.index(n), f"{i}, {n}'den önce gelmeli")
        o2 = ns["toposort"](g, "c")
        ensure(set(o2) == {"a", "b", "c"}, "yalnız erişilebilir düğümler")
    R.test("toposort", ts)

    def bw():
        v = ns["forward"](g, {"a": 2.0, "b": 3.0})
        ensure(need(ns, "backward")(g, v, "L") == {"a": 64.0, "b": 32.0}, "hook'suz")
        ensure(ns["backward"](g, v, "L", hooks={"c": lambda x: 2 * x}) == {"a": 112.0, "b": 64.0}, "c'de ×2 hook")
        ensure(ns["backward"](g, v, "L", hooks={"a": lambda x: 0.0}) == {"a": 0.0, "b": 32.0}, "yaprak hook'u")
    R.test("backward + hook'lar", bw)

    def deep():
        rng = random.Random(0)
        G = {"x": ("leaf", []), "y": ("leaf", [])}
        names = ["x", "y"]
        for k in range(30):
            op = rng.choice(["add", "mul"])
            G[f"n{k}"] = (op, [rng.choice(names), rng.choice(names)])
            names.append(f"n{k}")
        out = names[-1]
        f = lambda x, y: ns["forward"](G, {"x": x, "y": y})[out]
        v = need(ns, "forward")(G, {"x": 0.9, "y": 1.05})
        gr = need(ns, "backward")(G, v, out)
        h = 1e-6
        num = [(f(0.9 + h, 1.05) - f(0.9 - h, 1.05)) / (2 * h), (f(0.9, 1.05 + h) - f(0.9, 1.05 - h)) / (2 * h)]
        close([gr.get("x", 0.0), gr.get("y", 0.0)], num, rtol=1e-4)
    R.test("30 düğümlü rastgele DAG = sayısal türev", deep)
