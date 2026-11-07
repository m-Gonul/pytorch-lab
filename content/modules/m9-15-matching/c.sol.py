import numpy as np
from scipy import signal


def ncc_map(img, t):
    img, t = img.astype(np.float64), t.astype(np.float64)
    h, w = t.shape
    n = h * w
    tz = t - t.mean()
    corr = signal.fftconvolve(img, tz[::-1, ::-1], mode="valid")
    ii = np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(img * img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    S = lambda I: I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]
    s1 = S(ii)
    var = np.maximum(S(ii2) - s1 * s1 / n, 0)
    den = np.sqrt(var * (tz * tz).sum())
    return np.where(den > 1e-6 * np.sqrt((tz * tz).sum()) + 1e-12, corr / np.maximum(den, 1e-12), 0.0)


def normalize_points(p):
    c = p.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    q = np.c_[p, np.ones(len(p))] @ T.T
    return q[:, :2], T


def _A(src, dst):
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    r1 = np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], -1)
    r2 = np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], -1)
    return np.concatenate([r1, r2], -2)


def _h_from_A(A):
    H = np.linalg.svd(A)[2][..., -1, :].reshape(*A.shape[:-2], 3, 3)
    return H


def dlt(src, dst):
    s, Ts = normalize_points(src)
    d, Td = normalize_points(dst)
    H = np.linalg.inv(Td) @ _h_from_A(_A(s, d)) @ Ts
    return H / H[2, 2]


def ransac(src, dst, thr=3.0, iters=500, seed=0):
    rng = np.random.default_rng(seed)
    N = len(src)
    idx = rng.random((iters, N)).argsort(1)[:, :4]
    Hs = _h_from_A(_A(src[idx], dst[idx]))
    p = np.c_[src, np.ones(N)] @ Hs.transpose(0, 2, 1)
    err = np.linalg.norm(p[..., :2] / p[..., 2:] - dst, axis=-1)
    best = int(np.argmax((err < thr).sum(1)))
    H = dlt(src[err[best] < thr], dst[err[best] < thr])
    q = np.c_[src, np.ones(N)] @ H.T
    mask = np.linalg.norm(q[:, :2] / q[:, 2:] - dst, axis=1) < thr
    return H, mask


rng = np.random.default_rng(0)
s = rng.uniform(0, 100, (10, 2))
print(dlt(s, s + 5))
