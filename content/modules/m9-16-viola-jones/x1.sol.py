import numpy as np


def window_stats(img, win):
    f = img.astype(np.float64)
    I = np.pad(f, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    I2 = np.pad(f * f, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    S = lambda J: J[win:, win:] - J[:-win, win:] - J[win:, :-win] + J[:-win, :-win]
    n = win * win
    m = S(I) / n
    return m, np.sqrt(np.maximum(S(I2) / n - m * m, 0))


def min_dtype(H, W, bits=8):
    mx = (2 ** bits - 1) * H * W
    pick = lambda v: next(t for t in (np.uint16, np.uint32, np.uint64) if v <= np.iinfo(t).max)
    return pick(mx), pick((2 ** bits - 1) ** 2 * H * W)


from skimage import data
m, s = window_stats(data.camera(), 24)
print(m.shape, s.max())
