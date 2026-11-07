import numpy as np
import cv2


def rgb2hsi(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    tot = r + g + b
    S = np.where(tot > 0, 1 - 3 * f.min(-1) / np.where(tot > 0, tot, 1), 0.0)
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    th = np.degrees(np.arccos(np.clip(0.5 * ((r - g) + (r - b)) / np.where(den > 0, den, 1), -1, 1)))
    return np.stack([np.where(den > 0, np.where(b <= g, th, 360 - th), 0.0), S, tot / 3], -1)


def hsi2rgb(hsi):
    H, S, I = hsi[..., 0] % 360, hsi[..., 1], hsi[..., 2]
    out = np.zeros(hsi.shape)
    for lo, order in ((0, (2, 0, 1)), (120, (0, 1, 2)), (240, (1, 2, 0))):
        m = (H >= lo) & (H < lo + 120)
        h = np.radians(H - lo)
        x = I * (1 - S)
        y = I * (1 + S * np.cos(h) / np.cos(np.radians(60) - h))
        z = 3 * I - (x + y)
        for ch, v in zip(order, (x, y, z)):
            out[..., ch] = np.where(m, v, out[..., ch])
    return out


def slice_cube(img, a, W):
    out = img.copy()
    out[(np.abs(img - np.asarray(a)) > W / 2).any(-1)] = 0.5
    return out


def slice_sphere(img, a, R0):
    out = img.copy()
    out[((img - np.asarray(a)) ** 2).sum(-1) > R0 ** 2] = 0.5
    return out


def equalize_intensity(img):
    hsi = rgb2hsi(img.astype(np.float64) / 255)
    I8 = np.clip(np.round(hsi[..., 2] * 255), 0, 255).astype(np.int64)
    cdf = np.bincount(I8.ravel(), minlength=256).cumsum()
    lut = np.round(255 * cdf / cdf[-1])
    hsi[..., 2] = lut[I8] / 255
    return np.clip(np.round(hsi2rgb(hsi) * 255), 0, 255).astype(np.uint8)


def equalize_channels(img):
    return np.stack([cv2.equalizeHist(np.ascontiguousarray(img[..., c])) for c in range(3)], -1)


from skimage import data
print(equalize_intensity(data.coffee()).shape)
