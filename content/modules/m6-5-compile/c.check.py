import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    N = 10 ** 6
    G = [("x", "input", ()), ("a", "input", ()), ("b", "input", ()),
         ("t1", "mul", ("x", "a")), ("t2", "add", ("t1", "b")), ("y", "tanh", ("t2",))]
    nb = {"x": 4 * N, "a": 0, "b": 0, "t1": 4 * N, "t2": 4 * N, "y": 4 * N}

    def ev():
        rng = np.random.default_rng(0)
        inp = {"x": rng.normal(size=5), "a": 2.0, "b": -1.0}
        close(need(ns, "evaluate")(G, inp)["y"], np.tanh(inp["x"] * 2 - 1))
    R.test("evaluate", ev)

    def fs():
        groups = need(ns, "fuse")(G, ["y"])
        ensure([sorted(g) for g in groups] == [["t1", "t2", "y"]], f"gruplar {groups}")
        ensure(need(ns, "bytes_moved")(G, [["t1"], ["t2"], ["y"]], nb, ["y"]) == 24 * N, "füzyonsuz 24 MB")
        ensure(ns["bytes_moved"](G, groups, nb, ["y"]) == 8 * N, "füzyonlu 8 MB")
    R.test("elle örnek: 24 MB → 8 MB", fs)

    G2 = [("x", "input", ()), ("e", "exp", ("x",)), ("s", "sum", ("e",)), ("m", "mul", ("e", "x")),
          ("t", "tanh", ("m",)), ("u", "add", ("t", "e")), ("o", "exp", ("u",))]
    nb2 = {k: 4 * N for k, _, _ in G2}; nb2["s"] = 4

    def fs2():
        groups = [sorted(g) for g in need(ns, "fuse")(G2, ["o", "s"])]
        ensure(["e"] in groups and ["s"] in groups, f"çok tüketicili e ve indirgeme s ayrı grup: {groups}")
        ensure(["m", "o", "t", "u"] in groups, f"m → t → u → o tek grup: {groups}")
        unf = need(ns, "bytes_moved")(G2, [[n] for n, op, _ in G2 if op != "input"], nb2, ["o", "s"])
        fus = ns["bytes_moved"](G2, ns["fuse"](G2, ["o", "s"]), nb2, ["o", "s"])
        R.table("Çok tüketicili graf", ["", "bayt"], [["füzyonsuz", f"{unf:,}"], ["füzyonlu", f"{fus:,}"]])
        ensure(fus < unf, "füzyon trafiği azaltmalı")
    R.test("çok tüketicili ve indirgemeli graf", fs2)
