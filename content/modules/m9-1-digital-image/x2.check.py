import numpy as np
import cv2
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("cv2/numpy kullanılmıyor", lambda: no_imports(src, {"cv2", "numpy", "scipy"}))
    rng = np.random.default_rng(0)
    B = (rng.random((30, 40)) > 0.03).astype(np.uint8)
    B[0, :] = 1; B[:, 0] = 1
    B[5, 7] = 0
    for metric, dist in (("d4", cv2.DIST_L1), ("d8", cv2.DIST_C)):
        def t(metric=metric, dist=dist):
            got = np.array(need(ns, "distance_transform")(B.tolist(), metric), np.float32)
            close(got, cv2.distanceTransform(B, dist, 3), atol=0)
        R.test(f"{metric} = cv2.distanceTransform", t)
