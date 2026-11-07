from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def cs():
        f = need(ns, "contiguous_strides")
        for shape, want in [((2, 3), (3, 1)), ((2, 3, 4), (12, 4, 1)), ((5,), (1,)), ((1, 7, 1, 2), (14, 2, 2, 1))]:
            ensure(tuple(f(shape)) == want, f"{shape} → {tuple(f(shape))}, {want} bekleniyordu")
    R.test("contiguous_strides", cs)

    data = list(range(24))

    def gt():
        f = need(ns, "get")
        ensure(f(list(range(6)), (3, 1), 0, (1, 2)) == 5, "A[1,2] = 5 bekleniyordu")
        ensure(f(data, (12, 4, 1), 0, (1, 2, 3)) == 23, "(2,3,4) tensörde [1,2,3] = 23")
        ensure(f(data, (12, 4, 1), 4, (0, 1, 1)) == 9, "offset hesaba katılmıyor")
    R.test("get: adres formülü (offset dahil)", gt)

    def tr():
        sh, st = need(ns, "transpose")((2, 3, 4), (12, 4, 1), 0, 2)
        ensure(tuple(sh) == (4, 3, 2) and tuple(st) == (1, 4, 12), f"{sh}, {st}")
        m = need(ns, "materialize")(data, tuple(sh), tuple(st))
        want = [[[data[k * 12 + j * 4 + i] for k in range(2)] for j in range(3)] for i in range(4)]
        ensure(m == want, "transpoze edilmiş materialize yanlış")
    R.test("transpose + materialize (3 boyut)", tr)

    def win():
        sh, st = need(ns, "windows")(6, 3)
        ensure(need(ns, "materialize")(list(range(10, 16)), tuple(sh), tuple(st)) == [[10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15]], "pencereler yanlış")
        sh, st = need(ns, "windows")(5, 5)
        ensure(tuple(sh) == (1, 5), "w = n için tek pencere")
    R.test("windows: kopyasız kayan pencere", win)
