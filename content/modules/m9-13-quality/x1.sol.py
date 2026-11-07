import numpy as np
from scipy import ndimage


def ssim_map(x, y, win=7, sample_cov=True, R=255):
    x, y = x.astype(np.float64), y.astype(np.float64)
    f = lambda a: ndimage.uniform_filter(a, size=win, mode="reflect")
    n = win * win
    norm = n / (n - 1) if sample_cov else 1.0
    ux, uy = f(x), f(y)
    vx, vy, vxy = norm * (f(x * x) - ux * ux), norm * (f(y * y) - uy * uy), norm * (f(x * y) - ux * uy)
    C1, C2 = (0.01 * R) ** 2, (0.03 * R) ** 2
    S = (2 * ux * uy + C1) * (2 * vxy + C2) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    p = (win - 1) // 2
    return float(S[p:-p, p:-p].mean()), S


from skimage import data
print(ssim_map(data.camera(), data.camera())[0])
