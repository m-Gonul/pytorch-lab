import numpy as np
from skimage import data
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
from labcheck import close, max_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yalnız rect_table'da", lambda: (max_loops(src, 0, func="haar_matrix"), max_loops(src, 0, func="best_stump")))
    X = data.lfw_subset()
    rng = np.random.default_rng(0)
    coords, kinds = [], []
    for k in ("type-2-x", "type-2-y", "type-3-x", "type-3-y", "type-4"):
        c, t = haar_like_feature_coord(25, 25, k)
        idx = rng.choice(len(c), 60, replace=False)
        coords += list(c[idx]); kinds += list(t[idx])
    coords_a = np.empty(len(coords), object); coords_a[:] = coords
    kinds_a = np.array(kinds, object)

    def hm():
        Rt, St = need(ns, "rect_table")(coords, kinds)
        F = need(ns, "haar_matrix")(X[:20], Rt, St)
        ensure(F.shape == (20, 300), f"şekil {F.shape}")
        for i in (0, 7, 19):
            want = haar_like_feature(integral_image(X[i]), 0, 0, 25, 25, feature_type=kinds_a, feature_coord=coords_a)
            close(F[i], want, atol=1e-9, what=f"görüntü {i}")
    R.test("haar_matrix = skimage (300 öznitelik, 20 görüntü)", hm)

    def bs():
        for trial in range(20):
            f = np.round(rng.normal(size=40), 1); y = rng.integers(0, 2, 40).astype(float); w = rng.random(40); w /= w.sum()
            err, thr, pol = need(ns, "best_stump")(f, y, w)
            best = min(((w * (((p * f) <= (p * t)).astype(float) != y)).sum(), t, p) for t in np.unique(f) for p in (1, -1))
            close(err, best[0], atol=1e-12, what="en küçük ağırlıklı hata")
            pred = ((pol * f) <= (pol * thr)).astype(float)
            close((w * (pred != y)).sum(), err, atol=1e-12, what="döndürülen eşik bu hatayı vermeli")
    R.test("best_stump = kaba kuvvet (20 rastgele durum)", bs)
