import numpy as np
from scipy.ndimage import convolve


def bayer_mask(H, W):
    """RGGB: (0,0) R, (0,1) G, (1,0) G, (1,1) B. Dönüş (H, W, 3) bool."""
    y, x = np.mgrid[0:H, 0:W]
    r = (y % 2 == 0) & (x % 2 == 0)
    b = (y % 2 == 1) & (x % 2 == 1)
    return np.stack([r, ~(r | b), b], -1)


def mosaic(rgb):
    return (rgb * bayer_mask(*rgb.shape[:2])).sum(-1)


def demosaic_bilinear(raw):
    m = bayer_mask(*raw.shape)
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0
    r = convolve(raw * m[..., 0], k_rb, mode="mirror")
    g = convolve(raw * m[..., 1], k_g, mode="mirror")
    b = convolve(raw * m[..., 2], k_rb, mode="mirror")
    return np.stack([r, g, b], -1)


def psnr(a, b, peak=1.0):
    return 10 * np.log10(peak ** 2 / np.mean((a - b) ** 2))
