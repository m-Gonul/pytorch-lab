import numpy as np
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R_):
    src = ns["__source__"]
    R_.test("box_filter döngüsüz", lambda: max_loops(src, 0, func="box_filter"))

    def t():
        img = np.random.default_rng(0).integers(0, 256, (60, 70)).astype(np.uint8)
        for r in (0, 1, 5, 40):
            close(need(ns, "box_filter")(img, r), need(ns, "box_direct")(img, r), what=f"r = {r}")
        big = np.random.default_rng(1).integers(0, 256, (512, 512)).astype(np.uint8)
        rows = [[r, fmt_time(bench(lambda: ns["box_filter"](big, r), repeat=3, number=1))] for r in (1, 10, 100)]
        ensure(True, "")
        R_.table("512×512 kutu filtresi (integral görüntüyle)", ["yarıçap r", "süre"], rows,
                 note="Süre pencere boyutundan bağımsız: 4 okuma, 3 toplama. Viola–Jones'un Haar öznitelikleri ve hızlı kutu bulanıklığı (ve iterasyonla Gauss yaklaşımı) böyle hesaplanır.")
    R_.test("kutu filtresi", t)
