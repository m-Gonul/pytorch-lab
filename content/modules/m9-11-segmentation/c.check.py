import numpy as np
from sklearn.cluster import KMeans
from skimage import data
from labcheck import close, max_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız iterasyon döngüsü", lambda: max_loops(src, 1, func="kmeans_np"))
    img = data.astronaut()[::4, ::4]
    Z = img.reshape(-1, 3).astype(np.float64)

    for k, seed in [(3, 0), (6, 1), (12, 2)]:
        def t(k=k, seed=seed):
            init = Z[np.random.default_rng(seed).choice(len(Z), k, replace=False)]
            lab, m = need(ns, "kmeans_np")(Z, init)
            sk = KMeans(k, init=init, n_init=1, tol=0, algorithm="lloyd", max_iter=300).fit(Z)
            ensure(np.array_equal(lab, sk.labels_), f"{(lab != sk.labels_).sum()} etiket farklı")
            close(m, sk.cluster_centers_, atol=1e-9, what="merkezler")
        R.test(f"kmeans_np = sklearn (k = {k})", t)

    def q():
        rows = []
        prev = np.inf
        for k in (2, 4, 8, 16):
            out, lab = need(ns, "quantize")(img, k)
            ensure(out.dtype == np.uint8 and out.shape == img.shape and lab.shape == img.shape[:2], "şekiller")
            ensure(len(np.unique(out.reshape(-1, 3), axis=0)) <= k, "en fazla k renk")
            mse = float(((out.astype(float) - img) ** 2).mean())
            rows.append([k, f"{mse:.1f}", f"{10 * np.log10(255 ** 2 / mse):.2f}"])
            prev = mse
        ensure(float(rows[-1][1]) < float(rows[0][1]), "k arttıkça hata azalmalı")
        R.table("Renk nicemleme (astronaut 128²)", ["k", "MSE", "PSNR (dB)"], rows)
    R.test("quantize", q)
