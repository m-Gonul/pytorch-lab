import numpy as np

M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(img):
    return srgb_to_linear(img) @ M.T


def xyz_to_lab(xyz):
    q = xyz / WHITE
    f = np.where(q > 0.008856, np.cbrt(q), 7.787 * q + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def chromaticity(xyz):
    s = xyz.sum(-1, keepdims=True)
    s = np.where(s == 0, 1, s)
    return xyz[..., 0] / s[..., 0], xyz[..., 1] / s[..., 0]


def delta_e76(lab1, lab2):
    return np.sqrt(((np.asarray(lab1) - np.asarray(lab2)) ** 2).sum(-1))


print(xyz_to_lab(rgb_to_xyz(np.array([[[1.0, 1.0, 1.0]]]))))
