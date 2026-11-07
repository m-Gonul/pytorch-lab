import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera().astype(np.float64)[100:300, 150:330]

    def rot():
        for deg in (30, -75):
            got = need(ns, "rotate_bilinear")(img, deg)
            ref = ndimage.rotate(img, deg, reshape=False, order=1, mode="constant", cval=0.0)
            H, W = img.shape
            inner = (slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4))
            close(got[inner], ref[inner], atol=1e-6, what=f"{deg}° (merkez bölge)")
    R.test("rotate_bilinear = scipy.ndimage.rotate (saat yönünün tersine)", rot)

    def aff():
        rng = np.random.default_rng(0)
        A = np.array([[0.9, -0.3], [0.25, 1.1]]); t = np.array([5.0, -2.0])
        src_pts = rng.uniform(0, 100, (12, 2))
        dst_pts = src_pts @ A.T + t + rng.normal(0, 0.01, (12, 2))
        Ah, th = need(ns, "estimate_affine")(src_pts, dst_pts)
        close(Ah, A, atol=1e-3); close(th, t, atol=0.05)
        A3, t3 = ns["estimate_affine"](src_pts[:3], (src_pts[:3] @ A.T + t))
        close(A3, A, atol=1e-8, what="tam 3 nokta → tam çözüm")
    R.test("estimate_affine (gürültülü 12 nokta, tam 3 nokta)", aff)
