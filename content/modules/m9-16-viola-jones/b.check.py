import numpy as np
from skimage import data
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/skimage kullanılmıyor", lambda: no_imports(src, {"numpy", "skimage", "cv2", "scipy"}))
    img = data.lfw_subset()[3]

    def ii():
        got = np.array(need(ns, "integral")(img.tolist()))
        want = np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        close(got, want, atol=1e-9, what="integral")
        close(need(ns, "rect_sum")(got.tolist(), [(3, 4), (10, 12)]), img[3:11, 4:13].sum(), atol=1e-9, what="rect_sum")
    R.test("integral ve rect_sum", ii)

    for kind in ("type-2-x", "type-2-y", "type-3-x", "type-3-y", "type-4"):
        def t(kind=kind):
            coords, types = haar_like_feature_coord(25, 25, kind)
            idx = np.random.default_rng(len(kind)).choice(len(coords), 300, replace=False)
            c = coords[idx]; ty = types[idx]
            want = haar_like_feature(integral_image(img), 0, 0, 25, 25, feature_type=ty, feature_coord=c)
            got = need(ns, "features_at")(img.tolist(), [list(map(tuple, map(lambda r: list(map(tuple, r)), x))) for x in c], list(ty))
            close(np.array(got), want, atol=1e-9, what=kind)
        R.test(f"haar = skimage ({kind})", t)
