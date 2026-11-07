import numpy as np
import cv2
from skimage import data
from labcheck import no_loops, need, ensure


def ref_mh(img, sigma, thr_frac):
    n = int(np.ceil(6 * sigma)); n += 1 - n % 2
    g = cv2.Laplacian(cv2.GaussianBlur(img.astype(np.float64), (n, n), sigma), cv2.CV_64F)
    thr = thr_frac * g.max(); H, W = g.shape
    z = np.zeros((H, W), bool)
    c = lambda a, b: g[1 + a:H - 1 + a, 1 + b:W - 1 + b]
    for p, q in [(c(0, -1), c(0, 1)), (c(-1, 0), c(1, 0)), (c(-1, -1), c(1, 1)), (c(-1, 1), c(1, -1))]:
        z[1:-1, 1:-1] |= (p * q < 0) & (np.abs(p - q) > thr)
    return z


def gray(name):
    im = getattr(data, name)()
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    rows = []
    for name, lo, hi, l2 in [("camera", 50, 150, False), ("camera", 30, 90, True), ("coins", 100, 200, False), ("astronaut", 20, 60, True), ("moon", 40, 100.7, False)]:
        def t(name=name, lo=lo, hi=hi, l2=l2):
            img = gray(name)
            got = need(ns, "canny_cv")(img, lo, hi, l2)
            want = cv2.Canny(img, lo, hi, L2gradient=l2)
            diff = int((got != want).sum())
            ensure(diff == 0, f"{diff} piksel farklı (toplam kenar {int((want > 0).sum())})")
            rows.append([name, f"{lo}/{hi}", "L2" if l2 else "L1", int((want > 0).sum())])
        R.test(f"canny_cv = cv2.Canny ({name}, {lo}/{hi}, {'L2' if l2 else 'L1'})", t)
    if rows:
        R.table("Birebir eşleşen Canny çıktıları", ["görüntü", "eşikler", "norm", "kenar pikseli"], rows)

    def mh():
        img = gray("camera")[::2, ::2]
        for s, f in [(1.4, 0.0), (2.0, 0.04), (3.0, 0.04)]:
            got = need(ns, "marr_hildreth")(img, s, f)
            ensure(got.dtype == bool and np.array_equal(got, ref_mh(img, s, f)), f"σ = {s}, eşik %{int(f * 100)}")
        return f"σ = 2: eşiksiz {ref_mh(img, 2.0, 0).sum()} piksel, %4 eşikle {ns['marr_hildreth'](img, 2.0, 0.04).sum()} piksel (eşik 'spagetti' kenarları temizler)"
    R.test("marr_hildreth", mh)
