import numpy as np


def _win(f, se, pad):
    r0, r1 = se.shape[0] // 2, se.shape[1] // 2
    P = np.pad(f, ((r0, r0), (r1, r1)), constant_values=pad)
    return np.lib.stride_tricks.sliding_window_view(P, se.shape)


def gray_erode(f, se):
    return _win(f, se, 255)[..., se].min(-1)


def gray_dilate(f, se):
    s = se[::-1, ::-1]
    return _win(f, s, 0)[..., s].max(-1)


def opening(f, se):
    return gray_dilate(gray_erode(f, se), se)


def closing(f, se):
    return gray_erode(gray_dilate(f, se), se)


def gradient(f, se):
    return gray_dilate(f, se) - gray_erode(f, se)


def tophat(f, se):
    return f - opening(f, se)


def blackhat(f, se):
    return closing(f, se) - f


def nonflat_erode(f, bN):
    bN = np.asarray(bN, np.float64)
    r0, r1 = bN.shape[0] // 2, bN.shape[1] // 2
    P = np.pad(f.astype(np.float64), ((r0, r0), (r1, r1)), constant_values=np.inf)
    return (np.lib.stride_tricks.sliding_window_view(P, bN.shape) - bN).min((-2, -1))


from skimage import data
print(gradient(data.camera(), np.ones((3, 3), bool)).mean())
