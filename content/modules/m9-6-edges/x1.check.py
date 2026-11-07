import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    ring = [5, 5, 5, -3, -3, -3, -3, -3]
    pos = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    ks = []
    for i in range(8):
        k = np.zeros((3, 3))
        for (y, x), v in zip(pos, np.roll(ring, -i)):
            k[y, x] = v
        ks.append(k)

    def exact():
        img = data.camera()[200:300, 100:220]
        Rr = np.stack([cv2.filter2D(img.astype(np.float64), -1, k, borderType=cv2.BORDER_REPLICATE) for k in ks])
        r, i = need(ns, "kirsch")(img)
        close(r, Rr.max(0), atol=1e-9, what="yanıt")
        ensure(np.array_equal(i, Rr.argmax(0)), "yön indisi")
    R.test("kirsch = 8 çekirdeğin en büyüğü", exact)

    def dirs():
        names = ["K", "KB", "B", "GB", "G", "GD", "D", "KD"]
        rows = []
        for i, (dy, dx) in enumerate([(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]):
            yy, xx = np.mgrid[-20:21, -20:21]
            img = ((yy * dy + xx * dx) > 0).astype(np.uint8) * 200
            r, idx = ns["kirsch"](img)
            got = int(idx[20, 20])
            ensure(got == i, f"parlak taraf {names[i]} yönündeyken indis {got}")
            rows.append([names[i], got, int(r[20, 20])])
        R.table("Basamak yönü → Kirsch indisi", ["parlak taraf", "indis", "yanıt"], rows)
    R.test("8 yön doğru", dirs)
