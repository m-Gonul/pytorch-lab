import numpy as np
from skimage import data
from skimage.feature import hog
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/skimage kullanılmıyor", lambda: no_imports(src, {"numpy", "skimage", "cv2", "scipy"}))

    def g():
        img = data.camera()[100:116, 200:216].astype(float)
        gx, gy = need(ns, "gradients")(img.tolist())
        wx = np.zeros_like(img); wx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        wy = np.zeros_like(img); wy[1:-1] = img[2:] - img[:-2]
        close(np.array(gx), wx, what="gx"); close(np.array(gy), wy, what="gy")
    R.test("gradients", g)

    for name, sl in [("camera", (slice(100, 116), slice(200, 216))), ("coins", (slice(50, 66), slice(60, 76))), ("moon", (slice(300, 316), slice(10, 26)))]:
        def t(name=name, sl=sl):
            img = getattr(data, name)()[sl]
            got = need(ns, "hog16")(img.astype(float).tolist())
            want = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
            close(np.array(got), want, atol=1e-6, what="skimage hog")
        R.test(f"hog16 = skimage hog ({name})", t)
