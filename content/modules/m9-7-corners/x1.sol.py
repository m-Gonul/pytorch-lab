import numpy as np
import cv2


def _abc(img, sigma):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) / 8
    b = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    return b(gx * gx), b(gx * gy), b(gy * gy)


def eigen_maps(img, sigma=1.5):
    a, b, c = _abc(img, sigma)
    m = (a + c) / 2
    d = np.sqrt(((a - c) / 2) ** 2 + b * b)
    return m + d, m - d


def harris_from_eigs(l1, l2, k):
    return l1 * l2 - k * (l1 + l2) ** 2


def count_corners(R, T):
    return int(((R > T) & (R == cv2.dilate(R, np.ones((3, 3))))).sum())


from skimage import data
l1, l2 = eigen_maps(data.checkerboard() / 255.0)
print(l1.max(), l2.max())
