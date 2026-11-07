import numpy as np


def rotate_bilinear(img, deg):
    H, W = img.shape
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    # ters eşleme: çıkış (x, y) → kaynak, -θ ile döndür (görüntü koordinatlarında y aşağı)
    xr = c * (xs - cx) - s * (ys - cy) + cx
    yr = s * (xs - cx) + c * (ys - cy) + cy
    x0, y0 = np.floor(xr).astype(int), np.floor(yr).astype(int)
    a, b = xr - x0, yr - y0
    def at(yy, xx):
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        return np.where(ok, img[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)], 0.0)
    return ((1 - a) * (1 - b) * at(y0, x0) + a * (1 - b) * at(y0, x0 + 1)
            + (1 - a) * b * at(y0 + 1, x0) + a * b * at(y0 + 1, x0 + 1))


def estimate_affine(src, dst):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    M = np.zeros((2 * n, 6))
    M[0::2, 0:2] = src; M[0::2, 4] = 1
    M[1::2, 2:4] = src; M[1::2, 5] = 1
    p = np.linalg.lstsq(M, dst.reshape(-1), rcond=None)[0]
    return p[:4].reshape(2, 2), p[4:]


from skimage import data
img = data.camera().astype(np.float64)
print(rotate_bilinear(img, 30).shape)
