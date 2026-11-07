import numpy as np
from scipy import ndimage
from skimage import data
from sklearn.cluster import KMeans
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "scipy", "sklearn"}))

    def pt():
        rng = np.random.default_rng(0)
        img = (rng.random((30, 40)) * 20).astype(int); img[10, 12] = 200; img[20, 30] = 0; img[5, 5] = 220
        got = np.array(need(ns, "point_detect")(img.tolist(), 900))
        R_ = ndimage.correlate(img, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]), mode="constant")
        want = (np.abs(R_) >= 900).astype(int); want[0] = want[-1] = 0; want[:, 0] = want[:, -1] = 0
        ensure(np.array_equal(got, want), "nokta tespiti")
        return f"bulunan: {list(zip(*np.nonzero(got)))}"
    R.test("point_detect", pt)

    def ln():
        img = np.zeros((25, 25), int); img[12, :] = 100; np.fill_diagonal(img, 100)
        ks = {"h": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], "v": [[-1, 2, -1]] * 3, "d1": [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], "d2": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]}
        rows = []
        for kind, k in ks.items():
            got = np.array(need(ns, "line_detect")(img.tolist(), kind, 300))
            want = (ndimage.correlate(img, np.array(k), mode="constant") >= 300).astype(int); want[0] = want[-1] = 0; want[:, 0] = want[:, -1] = 0
            ensure(np.array_equal(got, want), f"{kind}")
            rows.append([kind, int(got.sum())])
        ensure(rows[0][1] > 15 and rows[2][1] > 15 and rows[1][1] == 0 and rows[3][1] <= 1, f"yatay ve d1 çizgiler bulunmalı: {rows}")
        R.table("Yatay + ana köşegen çizgili görüntü, T = 300", ["çekirdek", "piksel"], rows)
    R.test("line_detect (yalnız pozitif yanıt)", ln)

    def km():
        rng = np.random.default_rng(1)
        Z = np.concatenate([rng.normal(c, 1.0, (40, 2)) for c in ([0, 0], [6, 1], [2, 7])])
        init = Z[[0, 1, 2]]
        lab, m = need(ns, "kmeans")(Z.tolist(), init.tolist())
        sk = KMeans(3, init=init, n_init=1, tol=0, algorithm="lloyd").fit(Z)
        ensure(list(lab) == sk.labels_.tolist(), "etiketler sklearn ile farklı")
        close(np.array(m), sk.cluster_centers_, atol=1e-9, what="merkezler")
        ensure(ns["kmeans"]([[1], [2], [3], [10], [11], [12]], [[1], [2]])[1] == [[2.0], [11.0]], "elle örnek")
    R.test("kmeans = sklearn KMeans (lloyd)", km)
