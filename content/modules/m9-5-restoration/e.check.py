import numpy as np
import torch
from skimage import data
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def ref_adaptive(img, smax=7):
    H, W = img.shape
    P = np.pad(img, smax // 2, mode="edge").astype(int)
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            k = 3
            while True:
                r = k // 2
                win = P[y + smax // 2 - r:y + smax // 2 + r + 1, x + smax // 2 - r:x + smax // 2 + r + 1]
                zmin, zmax, zmed = win.min(), win.max(), int(np.median(win))
                if zmin < zmed < zmax:
                    z = int(img[y, x])
                    out[y, x] = z if zmin < z < zmax else zmed
                    break
                k += 2
                if k > smax:
                    out[y, x] = zmed
                    break
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız pencere boyutu döngüsü", lambda: max_loops(src, 1, func="adaptive_median_vec"))
    rng = np.random.default_rng(0)
    for p in (0.1, 0.4, 0.7):
        def t(p=p):
            img = data.camera()[100:180, 200:290].copy()
            u = rng.random(img.shape); img[u < p / 2] = 0; img[(u >= p / 2) & (u < p)] = 255
            got = need(ns, "adaptive_median_vec")(torch.tensor(img)).numpy()
            ensure(np.array_equal(got, ref_adaptive(img)), f"gürültü %{int(p * 100)}: başvuru ile farklı")
        R.test(f"%{int(p * 100)} tuz-biber = döngülü başvuru", t)
    if "adaptive_median_vec" not in ns:
        return
    img = data.camera()[:128, :128].copy(); u = rng.random(img.shape); img[u < 0.2] = 0; img[(u >= 0.2) & (u < 0.4)] = 255
    t_ref = bench(lambda: ref_adaptive(img), repeat=1, number=1)
    t_vec = bench(lambda: ns["adaptive_median_vec"](torch.tensor(img)), repeat=3)
    R.table("128 × 128, %40 gürültü", ["yöntem", "süre"], [["piksel döngüsü (NumPy)", fmt_time(t_ref)], ["vektörel", fmt_time(t_vec)]])
