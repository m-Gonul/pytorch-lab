import numpy as np
import cv2

_POS = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]


def _kernels():
    ring = np.array([5, 5, 5, -3, -3, -3, -3, -3], np.float64)
    ks = []
    for i in range(8):
        k = np.zeros((3, 3))
        for (y, x), v in zip(_POS, np.roll(ring, -i)):
            k[y, x] = v
        ks.append(k)
    return ks


def kirsch(img):
    f = img.astype(np.float64)
    R = np.stack([cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REPLICATE) for k in _kernels()])
    return R.max(0), R.argmax(0)


from skimage import data
r, i = kirsch(data.camera())
print(r.max(), np.bincount(i.ravel(), minlength=8))
