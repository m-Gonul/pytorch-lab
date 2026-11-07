import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage import data
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))

    def rm():
        x = torch.randn(3, 4, 37)
        for k in (1, 3, 7, 15, 41):
            want = F.max_pool1d(F.pad(x, (k // 2, k // 2), value=-float("inf")), k, 1)
            close(need(ns, "running_max")(x, k), want, what=f"k = {k}")
    R.test("running_max = max_pool1d", rm)

    img = data.camera()
    t = torch.tensor(img)
    for kh, kw in [(3, 3), (15, 7), (61, 61), (1, 31)]:
        def tt(kh=kh, kw=kw):
            k = np.ones((kh, kw), np.uint8)
            ensure(np.array_equal(need(ns, "rect_dilate")(t, kh, kw).numpy(), cv2.dilate(img, k)), "genişleme")
            ensure(np.array_equal(need(ns, "rect_erode")(t, kh, kw).numpy(), cv2.erode(img, k)), "erozyon")
        R.test(f"= cv2 ({kh} × {kw})", tt)
    if "rect_dilate" not in ns:
        return
    rows = []
    for k in (3, 15, 61):
        naive = lambda: F.unfold(F.pad(t[None, None].float(), (k // 2,) * 4, value=-1.0), k).max(1).values
        rows.append([k, fmt_time(bench(naive, repeat=3)), fmt_time(bench(lambda: ns["rect_dilate"](t, k, k), repeat=3)), fmt_time(bench(lambda: cv2.dilate(img, np.ones((k, k), np.uint8)), repeat=3))])
    R.table("512 × 512 kare genişleme", ["k", "unfold (k² eleman/piksel)", "van Herk (3 karş./piksel)", "cv2.dilate"], rows,
            note="unfold süresi k² ile büyür; van Herk'inki k'dan bağımsız.")
