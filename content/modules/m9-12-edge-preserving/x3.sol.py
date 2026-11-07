import numpy as np
import cv2

FILTERS = {
    "gauss5": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    "median5": lambda x: cv2.medianBlur(x, 5),
    "bilateral": lambda x: cv2.bilateralFilter(x, 9, 50, 3),
    "nlm": lambda x: cv2.fastNlMeansDenoising(x, None, 15, 7, 21),
}


def method_noise(clean, fn):
    return clean.astype(np.float64) - fn(clean)


def compare(clean, sigma=15, seed=0):
    noisy = np.clip(clean + np.random.default_rng(seed).normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)
    return {k: (cv2.PSNR(clean, f(noisy)), float(method_noise(clean, f).std())) for k, f in FILTERS.items()}


from skimage import data
print(compare(data.camera()[:128, :128]))
