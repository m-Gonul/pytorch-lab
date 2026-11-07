import numpy as np
import cv2
import torch
from skimage import data
from labcheck import max_loops, need, ensure, bench, fmt_time


def gray(name):
    im = getattr(data, name)()
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im


def check(ns, R):
    src = ns["__source__"]
    R.test("tek döngü (histerezis)", lambda: max_loops(src, 1, func="canny_torch"))
    crops = np.stack([gray(n)[40:200, 60:260] for n in ("camera", "coins", "astronaut", "moon")])
    for lo, hi, l2 in [(50, 150, False), (30, 90, True), (20, 60.5, False)]:
        def t(lo=lo, hi=hi, l2=l2):
            got = need(ns, "canny_torch")(torch.tensor(crops), lo, hi, l2).numpy()
            for i in range(len(crops)):
                want = cv2.Canny(crops[i], lo, hi, L2gradient=l2)
                ensure(np.array_equal(got[i], want), f"görüntü {i}: {(got[i] != want).sum()} piksel farklı")
        R.test(f"canny_torch = cv2.Canny ({lo}/{hi}, {'L2' if l2 else 'L1'})", t)
    if "canny_torch" not in ns:
        return
    big = np.stack([cv2.resize(gray(n), (256, 256)) for n in ("camera", "coins", "astronaut", "moon")] * 2)
    tb = torch.tensor(big)
    t_cv = bench(lambda: [cv2.Canny(b, 50, 150) for b in big], repeat=3)
    t_t = bench(lambda: ns["canny_torch"](tb, 50, 150), repeat=3)
    R.table("8 × 256 × 256", ["yöntem", "süre"], [["cv2.Canny döngüsü", fmt_time(t_cv)], ["canny_torch (toplu)", fmt_time(t_t)]],
            note="CPU'da OpenCV'nin elle optimize edilmiş C++'ı kazanır; histerezisin yinelemeli genişlemesi en uzun zincir kadar sürer. GPU'da ve büyük yığınlarda tablo değişir.")
