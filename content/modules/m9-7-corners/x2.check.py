import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure, bench, fmt_time


def check(ns, R):
    rows = []
    for name in ("camera", "coins", "astronaut"):
        def t(name=name):
            im = getattr(data, name)()
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            for th in (10, 25, 40):
                kp = cv2.FastFeatureDetector_create(th, False).detect(im)
                want = {(int(k.pt[0]), int(k.pt[1])) for k in kp}
                m = need(ns, "fast")(im, th)
                ys, xs = np.nonzero(m)
                got = set(zip(xs.tolist(), ys.tolist()))
                ensure(got == want, f"t = {th}: {len(got ^ want)} nokta farklı")
                rows.append([name, th, len(want)])
        R.test(f"fast = cv2.FAST ({name})", t)
    if rows:
        R.table("FAST-9 (nonmax kapalı)", ["görüntü", "t", "köşe"], rows)
    if "fast" in ns:
        im = data.camera()
        t1 = bench(lambda: ns["fast"](im, 20), repeat=3)
        t2 = bench(lambda: cv2.FastFeatureDetector_create(20, False).detect(im), repeat=3)
        t3 = bench(lambda: cv2.cornerHarris(im, 3, 3, 0.04), repeat=3)
        R.table("512 × 512", ["yöntem", "süre"], [["senin FAST (NumPy)", fmt_time(t1)], ["cv2 FAST", fmt_time(t2)], ["cv2.cornerHarris", fmt_time(t3)]])
