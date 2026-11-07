from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def bs():
        f = need(ns, "broadcast_shape")
        for a, b, want in [((2, 3), (3,), (2, 3)), ((4, 1), (1, 5), (4, 5)), ((5, 1, 3), (2, 1), (5, 2, 3)), ((), (3,), (3,)), ((7,), (1,), (7,))]:
            ensure(tuple(f(a, b)) == want, f"{a} + {b} → {f(a, b)}, {want} bekleniyordu")
        try:
            f((2, 3), (2,))
        except ValueError:
            return
        raise AssertionError("(2, 3) ve (2,) için ValueError bekleniyordu")
    R.test("broadcast_shape: kurallar ve uyumsuzluk", bs)

    def ub():
        f = need(ns, "unbroadcast2d")
        G = [[1, 2, 3], [4, 5, 6]]
        close(f(G, (3,)), [5, 7, 9]); close(f(G, (1, 3)), [[5, 7, 9]]); close(f(G, (2, 1)), [[6], [15]])
        close(f(G, ()), 21); close(f(G, (2, 3)), G)
    R.test("unbroadcast2d: (m,), (1, m), (n, 1), (), (n, m)", ub)
