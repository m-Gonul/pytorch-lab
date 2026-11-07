import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    rows = []
    for name in ("camera", "coins", "astronaut"):
        def t(name=name):
            im = getattr(data, name)()
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            for mc, q, md in [(100, 0.01, 10), (500, 0.05, 5), (50, 0.1, 20)]:
                want = cv2.goodFeaturesToTrack(im, mc, q, md)[:, 0]
                got = need(ns, "good_features")(im, mc, q, md)
                ensure(got.shape == want.shape, f"({mc}, {q}, {md}): {len(got)} köşe, {len(want)} bekleniyordu")
                ensure(np.array_equal(got, want), f"({mc}, {q}, {md}): {int((got != want).any(1).sum())} köşe farklı")
                rows.append([name, mc, q, md, len(want)])
        R.test(f"good_features = cv2.goodFeaturesToTrack ({name})", t)
    if rows:
        R.table("Birebir eşleşen Shi-Tomasi seçimleri", ["görüntü", "max", "kalite", "min uzaklık", "köşe"], rows)
