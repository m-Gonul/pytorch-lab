import numpy as np
import cv2
from skimage import data
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src_ = ns["__source__"]
    R.test("numpy/cv2 kullanılmıyor", lambda: no_imports(src_, {"numpy", "cv2", "scipy", "skimage"}))

    def n():
        close(need(ns, "ncc")([1, 2, 3, 4], [2, 1, 4, 3]), 0.6, what="elle örnek")
        close(ns["ncc"]([1, 2, 3], [10, 20, 30]), 1.0, what="kazanç değişmezliği")
        close(ns["ncc"]([1, 2, 3], [7, 8, 9]), 1.0, what="ofset değişmezliği")
        ensure(ns["ncc"]([1, 1, 1], [1, 2, 3]) == 0, "sıfır payda → 0")
    R.test("ncc", n)

    def mt():
        img = data.camera()[200:236, 240:280].astype(np.float32)
        t = img[10:22, 15:25].copy()
        got = np.array(need(ns, "match_template")(img.tolist(), t.tolist()))
        want = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        ensure(got.shape == want.shape, f"şekil {got.shape}, {want.shape} bekleniyordu")
        close(got, want, atol=1e-4, what="cv2 TM_CCOEFF_NORMED")
        ensure(np.unravel_index(np.argmax(got), got.shape) == (10, 15), "en iyi eşleşme şablonun kesildiği yer")
    R.test("match_template = cv2.matchTemplate", mt)

    def h4():
        rng = np.random.default_rng(0)
        for _ in range(5):
            Ht = np.array([[rng.uniform(0.8, 1.2), rng.uniform(-0.2, 0.2), rng.uniform(-30, 30)], [rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.2), rng.uniform(-30, 30)], [rng.uniform(-1e-3, 1e-3), rng.uniform(-1e-3, 1e-3), 1]])
            src = rng.uniform(0, 400, (4, 2))
            dst = cv2.perspectiveTransform(src[None], Ht)[0]
            H = np.array(need(ns, "solve_h4")(src.tolist(), dst.tolist()))
            close(H, Ht, atol=1e-6, what="homografi")
            close(np.array(need(ns, "apply_h")(H.tolist(), 10.0, 20.0)), cv2.perspectiveTransform(np.array([[[10.0, 20.0]]]), Ht)[0, 0], atol=1e-6, what="apply_h")
    R.test("solve_h4 ve apply_h", h4)
