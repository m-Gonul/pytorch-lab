import numpy as np
import cv2


def box_sum(ii, y0, x0, h, w):
    return ii[y0 + h, x0 + w] - ii[y0, x0 + w] - ii[y0 + h, x0] + ii[y0, x0]


def haar(ii, y, x, s):
    dx = box_sum(ii, y - s, x, 2 * s, s) - box_sum(ii, y - s, x - s, 2 * s, s)
    dy = box_sum(ii, y, x - s, s, 2 * s) - box_sum(ii, y - s, x - s, s, 2 * s)
    return dx, dy


def usurf(img, pts, s):
    ii = cv2.integral(img).astype(np.float64)
    pts = np.asarray(pts)
    off = np.round((np.arange(20) - 9.5) * s).astype(np.int64)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    Y = pts[:, 0, None, None] + oy
    X = pts[:, 1, None, None] + ox
    dx, dy = haar(ii, Y, X, s)
    g = np.exp(-(oy ** 2 + ox ** 2) / (2 * (3.3 * s) ** 2))
    dx, dy = dx * g, dy * g
    sub = lambda a: a.reshape(len(pts), 4, 5, 4, 5).sum((2, 4))
    d = np.stack([sub(dx), sub(np.abs(dx)), sub(dy), sub(np.abs(dy))], -1).reshape(len(pts), 64)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


from skimage import data
print(usurf(data.camera(), np.array([[256, 256]]), 2).shape)
