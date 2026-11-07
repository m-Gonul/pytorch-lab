import random
from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def tree():
        ensure(need(ns, "reduce_tree")([3, 1, 7, 0, 4, 1, 6, 3], max, float("-inf")) == (7, 3, 7), "el örneği")
        r = ns["reduce_tree"](list(range(1, 1001)), lambda a, b: a + b, 0)
        ensure(r == (500500, 10, 999), "1000 eleman: 10 adım, 999 işlem")
        cat = ns["reduce_tree"](list("paralel"), lambda a, b: a + b, "")
        ensure(cat[0] == "paralel", "birleşmeli ama değişmeli olmayan işleç: sıra korunmalı")
    R_.test("reduce_tree", tree)

    def warps():
        s = need(ns, "warp_activity")(256, "simple")
        c = ns["warp_activity"](256, "convergent")
        ensure(sum(w for _, w, _ in s) == 55 and sum(w for _, w, _ in c) == 20, "etkin warp toplamları 55 ve 20")
        ensure(sum(d for *_, d in s) == 8 * 5 + 4 + 2 + 1 and sum(d for *_, d in c) == 5, "ayrışan warp sayıları")
        rng = random.Random(0)
        x = [rng.randint(0, 9) for _ in range(512)]
        ensure(need(ns, "run_block")(x, "simple") == sum(x) and ns["run_block"](x, "convergent") == sum(x), "iki eşleme de doğru")
        rows = [[i + 1, f"{s[i][0]} / {s[i][1]} / {s[i][2]}", f"{c[i][0]} / {c[i][1]} / {c[i][2]}"] for i in range(9)]
        R_.table("256 thread: aktif thread / aktif warp / ayrışan warp", ["iterasyon", "basit (adım ×2)", "yakınsak (adım ÷2)"], rows,
                 note="Aynı toplama sayısı, çok farklı donanım kullanımı: basit eşlemede aktif warp her zaman 'yarı boş'.")
        data = [rng.randint(0, 15) for _ in range(10000)]
        h, m = need(ns, "hist_private")(data, 8, 16)
        ensure(h == [data.count(b) for b in range(16)] and m == 8 * 16, "özel histogram")
    R_.test("warp etkinliği ve histogram", warps)
