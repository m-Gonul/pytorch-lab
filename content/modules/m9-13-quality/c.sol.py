import numpy as np
import cv2
from scipy import ndimage


def ssim(x, y, R=255):
    x, y = x.astype(np.float64), y.astype(np.float64)
    f = lambda a: ndimage.gaussian_filter(a, sigma=1.5, truncate=3.5, mode="reflect")
    ux, uy = f(x), f(y)
    vx, vy, vxy = f(x * x) - ux * ux, f(y * y) - uy * uy, f(x * y) - ux * uy
    C1, C2 = (0.01 * R) ** 2, (0.03 * R) ** 2
    S = (2 * ux * uy + C1) * (2 * vxy + C2) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    return float(S[5:-5, 5:-5].mean())


def _err(a, b):
    return float(((a.astype(np.float64) - b) ** 2).mean())


def mean_shift(img, mse):
    return img.astype(np.float64) + np.sqrt(mse)


def contrast(img, mse):
    f = img.astype(np.float64)
    a = 1 + np.sqrt(mse) / f.std()
    return f.mean() + a * (f - f.mean())


def noise(img, mse, seed=0):
    n = np.random.default_rng(seed).normal(size=img.shape)
    return img + n * np.sqrt(mse / (n ** 2).mean())


def impulse(img, mse, seed=0):
    rng = np.random.default_rng(seed)
    f = img.astype(np.float64).ravel()
    order = rng.permutation(f.size)
    val = np.where(rng.random(f.size) < 0.5, 0.0, 255.0)
    err = np.cumsum((val - f[order]) ** 2) / f.size
    k = int(np.searchsorted(err, mse)) + 1
    out = f.copy()
    out[order[:k]] = val[:k]
    return out.reshape(img.shape)


def blur(img, mse):
    f = img.astype(np.float64)
    lo, hi = 0.1, 20.0
    for _ in range(40):
        s = (lo + hi) / 2
        if _err(f, cv2.GaussianBlur(f, (0, 0), s)) < mse:
            lo = s
        else:
            hi = s
    return cv2.GaussianBlur(f, (0, 0), (lo + hi) / 2)


def jpeg(img, mse):
    best = None
    for q in range(1, 101):
        dec = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
        e = abs(_err(img, dec) - mse)
        if best is None or e < best[0]:
            best = (e, dec)
    return best[1].astype(np.float64)


from skimage import data
print(ssim(data.camera(), mean_shift(data.camera(), 210)))
