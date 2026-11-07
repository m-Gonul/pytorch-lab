import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    LSE = need(ns, "LogSumExp", "sınıf")

    def fw():
        close(LSE.apply([1.0, 2.0])[0], math.log(math.e + math.e ** 2))
        close(LSE.apply([1000.0, 1000.0])[0], 1000 + math.log(2), what="büyük girdi (taşma yok)")
    R.test("LogSumExp ileri (kararlı)", fw)

    def bw():
        out, ctx = LSE.apply([1.0, 2.0, 3.0])
        g = LSE.backward(ctx, 2.0)
        e = [math.exp(x) for x in (1, 2, 3)]
        close(g, [2 * v / sum(e) for v in e])
        out, ctx = LSE.apply([1000.0, 999.0])
        close(sum(LSE.backward(ctx, 1.0)), 1.0, what="büyük girdide softmax toplamı")
    R.test("LogSumExp geri = g · softmax", bw)

    def gc():
        ok, err = need(ns, "gradcheck")(LSE, [0.3, -1.2, 2.0, 0.0])
        ensure(ok, f"LogSumExp gradcheck'ten geçmeli (hata {err:.2e})")
        ok2, err2 = ns["gradcheck"](need(ns, "RoundSTE", "sınıf"), [0.3, 1.2, 2.7])
        ensure(not ok2, "RoundSTE gradcheck'ten GEÇMEMELİ: gradyanı bilerek gerçek türevden farklı")
        return f"LSE hata {err:.1e}; STE hata {err2:.1f} (beklenen: gerçek türev 0, STE 1)"
    R.test("gradcheck: LSE geçer, STE geçmez", gc)

    def ste():
        out, ctx = need(ns, "RoundSTE", "sınıf").apply([0.3, 1.7, -0.6])
        ensure(out == [0.0, 2.0, -1.0], f"ileri {out}")
        ensure(list(ns["RoundSTE"].backward(ctx, [2.0, 3.0, 4.0])) == [2.0, 3.0, 4.0], "geri özdeşlik")
    R.test("RoundSTE", ste)
