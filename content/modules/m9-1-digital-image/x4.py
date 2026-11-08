import numpy as np
from scipy.ndimage import convolve


def bayer_mask(H, W):
    """RGGB: (0,0) R, (0,1) G, (1,0) G, (1,1) B. Dönüş (H, W, 3) bool."""
    ...


def mosaic(rgb):
    ...


def demosaic_bilinear(raw):
    ...


def psnr(a, b, peak=1.0):
    ...
