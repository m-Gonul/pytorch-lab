import numpy as np
import cv2


def _win(img, k):
    r = k // 2
    return np.lib.stride_tricks.sliding_window_view(np.pad(img, r, mode="edge"), (k, k))


def median_np(img, k):
    return np.median(_win(img, k).reshape(*img.shape, k * k), -1).astype(img.dtype)


def adaptive_mean(img_u8, noise_var, k=5):
    g = img_u8.astype(np.float32) / 255
    w = _win(g, k)
    m = w.mean((-2, -1))
    v = w.var((-2, -1))
    ratio = np.minimum(noise_var / np.maximum(v, 1e-12), 1.0)
    f = g - ratio * (g - m)
    return np.clip(np.rint(f * 255), 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 20 * np.log10(255.0 / (np.sqrt(mse) + np.finfo(np.float64).eps))


from skimage import data
img = data.camera()
print(psnr(img, median_np(img, 5)))
