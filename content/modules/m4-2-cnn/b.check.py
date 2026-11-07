from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    R.test("conv_out", lambda: close([need(ns, "conv_out")(32, 3, 1, 1), need(ns, "conv_out")(32, 3, 2, 1), need(ns, "conv_out")(224, 7, 2, 3), need(ns, "conv_out")(7, 3, 3, 0)], [32, 16, 112, 2]))
    R.test("receptive_field", lambda: close([need(ns, "receptive_field")(l) for l in ([(3, 1)] * 2, [(3, 1)] * 3, [(3, 1), (3, 2), (3, 1)], [(7, 2), (3, 2), (3, 1), (3, 1)])], [5, 7, 9, 27]))
    R.test("grad_product", lambda: close([need(ns, "grad_product")([0.1] * 10, False), need(ns, "grad_product")([0.1] * 10, True), need(ns, "grad_product")([0.5, 2.0], False)], [1e-10, 1.1 ** 10, 1.0]))

    def pl():
        ensure(list(need(ns, "plan")(224, [(7, 2, 3), (3, 2, 1), (3, 1, 1), (3, 2, 1)])) == [112, 56, 56, 28], "ResNet planı")
    R.test("plan", pl)
