import numpy as np


def rgb2hsv8(img):
    f = img.astype(np.int64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    v, vmin = f.max(-1), f.min(-1)
    diff = v - vmin
    i = np.arange(256)
    d = np.maximum(i, 1)
    sdiv = np.where(i > 0, np.round((255 << 12) / d), 0).astype(np.int64)
    hdiv = np.where(i > 0, np.round((180 << 12) / (6.0 * d)), 0).astype(np.int64)
    s = (diff * sdiv[v] + 2048) >> 12
    h = np.where(v == r, g - b, np.where(v == g, b - r + 2 * diff, r - g + 4 * diff))
    h = (h * hdiv[diff] + 2048) >> 12
    h = h + np.where(h < 0, 180, 0)
    return np.stack([h, s, v], -1).astype(np.uint8)


def rgb2ycrcb8(img):
    f = img.astype(np.int64)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    Y = (R * 4899 + G * 9617 + B * 1868 + 8192) >> 14
    d = (128 << 14) + 8192
    Cr = ((R - Y) * 11682 + d) >> 14
    Cb = ((B - Y) * 9241 + d) >> 14
    return np.clip(np.stack([Y, Cr, Cb], -1), 0, 255).astype(np.uint8)


def rgb2hsi(img):
    f = img.astype(np.float64) / (255 if img.dtype == np.uint8 else 1)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    tot = r + g + b
    I = tot / 3
    S = np.where(tot > 0, 1 - 3 * f.min(-1) / np.where(tot > 0, tot, 1), 0.0)
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    th = np.degrees(np.arccos(np.clip(0.5 * ((r - g) + (r - b)) / np.where(den > 0, den, 1), -1, 1)))
    H = np.where(den > 0, np.where(b <= g, th, 360 - th), 0.0)
    return np.stack([H, S, I], -1)


from skimage import data
print(rgb2hsv8(data.astronaut())[0, 0])
