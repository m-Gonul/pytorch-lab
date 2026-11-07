import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: (max_loops(src, 0, func="gap_head"), max_loops(src, 0, func="cam")))

    def t():
        rng = np.random.default_rng(0)
        F = np.maximum(0, rng.normal(size=(64, 7, 7)))
        W = rng.normal(size=(10, 64))
        b = rng.normal(size=10)
        logits = need(ns, "gap_head")(F, W, b)
        for c in (0, 3, 9):
            close(need(ns, "cam")(F, W, c).mean(), logits[c] - b[c], atol=1e-10, what=f"CAM ortalaması = logit (sınıf {c})")
        F2 = np.zeros((2, 7, 7))
        F2[0, 1, 5] = 10
        F2[1, 5, 1] = 10
        m = ns["cam"](F2, np.array([[1.0, 0.0], [0.0, 1.0]]), 0)
        ensure(np.unravel_index(m.argmax(), m.shape) == (1, 5), "CAM sınıfın kanıtını konumlandırır")
        a, g = need(ns, "params_flatten_vs_gap")(512, 7, 7, 1000, 4096)
        R_.table("VGG-16 başı vs GAP başı (512×7×7 → 1000 sınıf)", ["baş", "parametre"], [["düzleştir + FC 4096 + FC 1000", f"{a:,}".replace(",", " ")], ["GAP + FC 1000", f"{g:,}".replace(",", " ")]],
                 note="GoogLeNet ve ResNet'ten beri sınıflandırma başı GAP'tir: yüz milyonlarca parametreden kurtarır, girdi boyutundan bağımsızdır ve CAM ile 'model nereye baktı'yı ücretsiz verir.")
    R_.test("GAP ve CAM", t)
