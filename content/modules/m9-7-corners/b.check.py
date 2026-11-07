import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import close, no_imports, need, ensure


def ref_resp(img, k, r):
    f = img.astype(np.float64)
    H, W = f.shape
    fx = np.zeros_like(f); fy = np.zeros_like(f)
    fx[1:-1, 1:-1] = f[1:-1, 2:] - f[1:-1, :-2]; fy[1:-1, 1:-1] = f[2:, 1:-1] - f[:-2, 1:-1]
    box = lambda a: ndimage.uniform_filter(a, 2 * r + 1, mode="constant") * (2 * r + 1) ** 2
    a, b, c = box(fx * fx), box(fx * fy), box(fy * fy)
    R = a * c - b * b - k * (a + c) ** 2
    out = np.zeros_like(R); out[r + 1:H - r - 1, r + 1:W - r - 1] = R[r + 1:H - r - 1, r + 1:W - r - 1]
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "scipy", "skimage"}))
    G = 255 ** 2

    def case():
        left = [[0, 255, 255, 255, 255], [0, 0, 255, 255, 255], [0, 0, 0, 255, 255], [0, 0, 0, 0, 255], [0, 0, 0, 0, 0]]
        right = [[255] * 5, [255] * 5, [0, 0, 0, 255, 255], [0, 0, 0, 255, 255], [0, 0, 0, 255, 255]]
        ensure(need(ns, "harris_matrix")(left) == [[5 * G, -5 * G], [-5 * G, 5 * G]], f"sol blok: {ns['harris_matrix'](left)}")
        ensure(ns["harris_matrix"](right) == [[4 * G, -G], [-G, 4 * G]], f"sağ blok: {ns['harris_matrix'](right)}")
    R.test("notlardaki vaka çalışması", case)

    img = data.camera()[200:240, 230:270].astype(int)

    def resp():
        for r in (1, 2):
            got = np.array(need(ns, "harris_response")(img.tolist(), 0.04, r), float)
            close(got, ref_resp(img, 0.04, r), rtol=1e-9, atol=1e-3, what=f"r = {r}")
    R.test("harris_response", resp)

    def lm():
        Rm = ref_resp(img, 0.04, 1)
        thr = 0.01 * Rm.max()
        got = need(ns, "local_maxima")(Rm.tolist(), thr, 2)
        mx = ndimage.maximum_filter(Rm, 5, mode="constant", cval=-np.inf)
        ys, xs = np.nonzero((Rm > thr) & (Rm >= mx))
        ensure([tuple(p) for p in got] == list(zip(xs.tolist(), ys.tolist())), "yerel en büyükler farklı")
        return f"{len(got)} köşe"
    R.test("local_maxima", lm)
