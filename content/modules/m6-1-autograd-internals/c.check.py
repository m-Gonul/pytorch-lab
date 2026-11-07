import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    Var = need(ns, "Var", "sınıf")

    def grads():
        x, w = Var([0.5, -1.0, 0.2]), Var([2.0, 3.0, -1.0])
        need(ns, "backward")(need(ns, "sum_")(need(ns, "exp")(need(ns, "mul")(x, w))))
        e = np.exp(x.data * w.data)
        close(x.grad, e * w.data); close(w.grad, e * x.data)
    R.test("sum(exp(x·w)) gradyanları", grads)

    def acc():
        x, w = Var([1.0]), Var([2.0])
        for _ in range(2):
            need(ns, "backward")(ns["sum_"](ns["mul"](x, w)))
        close(x.grad, [4.0], what="iki backward → birikim")
    R.test("yapraklarda gradyan birikimi", acc)

    def bad_exp():
        x = Var([0.3, 0.7])
        y = need(ns, "exp")(x)
        need(ns, "add_")(y, 1.0)
        try:
            need(ns, "backward")(ns["sum_"](y))
        except RuntimeError as e:
            ensure("inplace" in str(e), "hata mesajı 'inplace' içermeli")
            return "exp çıktısını sakladığı için hata: doğru"
        raise AssertionError("exp çıktısı in-place değişti, backward hata vermeliydi")
    R.test("exp(x).add_(1) → hata", bad_exp)

    def ok_mul():
        x, w = Var([0.3, 0.7]), Var([2.0, -1.0])
        y = need(ns, "mul")(x, w)
        need(ns, "add_")(y, 1.0)
        need(ns, "backward")(ns["sum_"](y))
        close(x.grad, w.data); close(w.grad, x.data)
    R.test("mul(x, w).add_(1) → sorun yok", ok_mul)

    def bad_mul():
        x, w = Var([0.3, 0.7]), Var([2.0, -1.0])
        h = need(ns, "exp")(x)
        z = need(ns, "mul")(h, w)
        need(ns, "add_")(h, 1.0)
        try:
            need(ns, "backward")(ns["sum_"](z))
        except RuntimeError:
            return
        raise AssertionError("mul'un sakladığı girdi değişti, hata bekleniyordu")
    R.test("saklanan girdi değişirse → hata", bad_mul)
