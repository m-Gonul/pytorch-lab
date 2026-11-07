import numpy as np
from scipy import ndimage


def rgb_to_y(img):
    f = img.astype(np.float64)
    return 16 + (65.481 * f[..., 0] + 128.553 * f[..., 1] + 24.966 * f[..., 2]) / 255


def psnr_y(a, b, shave=0):
    ya, yb = rgb_to_y(a), rgb_to_y(b)
    if shave:
        ya, yb = ya[shave:-shave, shave:-shave], yb[shave:-shave, shave:-shave]
    return float(10 * np.log10(255 ** 2 / ((ya - yb) ** 2).mean()))


def _ssim(x, y):
    f = lambda a: ndimage.gaussian_filter(a, 1.5, truncate=3.5, mode="reflect")
    ux, uy = f(x), f(y)
    vx, vy, vxy = f(x * x) - ux * ux, f(y * y) - uy * uy, f(x * y) - ux * uy
    S = (2 * ux * uy + 6.5025) * (2 * vxy + 58.5225) / ((ux * ux + uy * uy + 6.5025) * (vx + vy + 58.5225))
    return S[5:-5, 5:-5].mean()


def ssim_rgb(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.mean([_ssim(a[..., c], b[..., c]) for c in range(3)]))


from skimage import data
print(rgb_to_y(data.astronaut()).min())
