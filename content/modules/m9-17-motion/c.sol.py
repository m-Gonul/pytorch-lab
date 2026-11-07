import numpy as np
from scipy import ndimage


def block_match_np(prev, cur, bs, R):
    H, W = cur.shape
    nby, nbx = H // bs, W // bs
    P = np.pad(prev.astype(np.float64), R, constant_values=1e9)
    C = cur[:nby * bs, :nbx * bs].astype(np.float64).reshape(nby, bs, nbx, bs).transpose(0, 2, 1, 3)
    win = np.lib.stride_tricks.sliding_window_view(P, (bs, bs))
    oy = (np.arange(nby) * bs)[:, None, None, None] + np.arange(2 * R + 1)[None, None, :, None]
    ox = (np.arange(nbx) * bs)[None, :, None, None] + np.arange(2 * R + 1)[None, None, None, :]
    cand = win[oy, ox]
    sad = np.abs(cand - C[:, :, None, None]).sum((-2, -1)).reshape(nby, nbx, -1)
    k = sad.argmin(-1)
    return np.stack([k // (2 * R + 1) - R, k % (2 * R + 1) - R], -1)


def lk_flow(I0, I1, win=7, thr=1e-2):
    I0, I1 = I0.astype(np.float64), I1.astype(np.float64)
    Iy, Ix = np.gradient(I0)
    It = I1 - I0
    f = lambda a: ndimage.uniform_filter(a, win, mode="nearest")
    a, b, c = f(Ix * Ix), f(Ix * Iy), f(Iy * Iy)
    p, q = f(Ix * It), f(Iy * It)
    det = a * c - b * b
    lmin = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b * b)
    ok = lmin > thr
    d = np.where(ok, det, 1)
    return np.where(ok, (-c * p + b * q) / d, 0), np.where(ok, (b * p - a * q) / d, 0)


from skimage import data
p = data.camera()[:64, :64]
print(block_match_np(p, np.roll(p, 1, 1), 8, 2)[2, 2])
