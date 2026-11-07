import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="warp_cost"))

    def t():
        c = np.array([10.0, 30.0])
        close(need(ns, "warp_cost")(np.array([[0] * 32, [1] * 32, [0, 1] * 16]), c), np.array([10, 30, 40.0]), what="warp maliyetleri")
        th = need(ns, "kernel_cost")(1024, "thread", c)
        wp = ns["kernel_cost"](1024, "warp", c)
        ensure(th == 32 * 40 and wp == 16 * 10 + 16 * 30, "thread vs warp hizalı dallanma")
        R_.table("1024 thread, iki yol (maliyet 10 ve 30)", ["dallanma", "toplam warp maliyeti"], [["tid % 2 (her warp ayrışır)", th], ["(tid / 32) % 2 (warp hizalı)", wp], ["dal yok, yalnız yol 0", ns["kernel_cost"](1024, "none", c)]],
                 note="Aynı iş, aynı sonuç; yalnız thread–veri eşlemesi değişti. İndirgemede (M10.5) bu fikir 'ayrışmayı en aza indiren eşleme'ye dönüşür.")
    R_.test("ayrışma", t)
