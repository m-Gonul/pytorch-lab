from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    inc = [("load", "x", "r"), ("add", "r", 1), ("store", "x", "r")]
    locked = [("lock", "m")] + inc + [("unlock", "m")]

    def race():
        out, dl = need(ns, "explore")([inc, inc], {"x": 0})
        ensure(out == {(("x", 1),), (("x", 2),)} and not dl, "kayıp güncelleme mümkün")
        out, dl = ns["explore"]([locked, locked, locked], {"x": 0})
        ensure(out == {(("x", 3),)} and not dl, "kilitle yalnız doğru sonuç")
        out3, _ = ns["explore"]([inc, inc, inc], {"x": 0})
        ensure({v for ((_, v),) in out3} == {1, 2, 3}, "3 thread: 1, 2, 3")
        dep = lambda k: [("load", "b", "r"), ("add", "r", k), ("store", "b", "r")]
        outb, _ = ns["explore"]([dep(1000), dep(500)], {"b": 10000})
        ensure({v for ((_, v),) in outb} == {11500, 11000, 10500}, "banka örneği")
        ensure(need(ns, "count_interleavings")([2, 2]) == 6 and ns["count_interleavings"]([3, 3]) == 20 and ns["count_interleavings"]([3, 3, 3]) == 1680, "multinom")
    R_.test("yarışlar", race)

    def dead():
        t1 = [("lock", "A"), ("lock", "B"), ("load", "x", "r"), ("add", "r", 1), ("store", "x", "r"), ("unlock", "B"), ("unlock", "A")]
        t2 = [("lock", "B"), ("lock", "A"), ("load", "x", "r"), ("add", "r", 1), ("store", "x", "r"), ("unlock", "A"), ("unlock", "B")]
        out, dl = ns["explore"]([t1, t2], {"x": 0})
        ensure(dl and out == {(("x", 2),)}, "ters sırada kilit: kilitlenme mümkün ama biten yollar doğru")
        out, dl = ns["explore"]([t1, t1], {"x": 0})
        ensure(not dl, "aynı sırada kilit: kilitlenme yok")
        R_.table("Model denetimi", ["program", "olası sonuçlar", "kilitlenme"],
                 [["2 × (oku, +1, yaz)", "x ∈ {1, 2}", "yok"], ["3 × kilitli artış", "x = 3", "yok"], ["A→B ve B→A kilit sırası", "x = 2 (bitenlerde)", "VAR"], ["A→B ve A→B", "x = 2", "yok"]],
                 note="Test çalıştırmak zamanlamalardan birini görür; durum uzayını gezmek hepsini. Kilitlenme yalnız belirli iç içe geçmelerde çıkar — bu yüzden testlerde nadiren yakalanır.")
    R_.test("kilitlenme", dead)
