import numpy as np
import cv2


def _min_eig(img, block=3, ksize=3):
    s = 1.0 / ((1 << (ksize - 1)) * block * 255.0)
    f = img.astype(np.float32)
    dx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ksize, scale=s)
    dy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ksize, scale=s)
    box = lambda a: cv2.boxFilter(a, -1, (block, block), normalize=False)
    a, b, c = box(dx * dx) * 0.5, box(dx * dy), box(dy * dy) * 0.5
    return (a + c) - np.sqrt((a - c) ** 2 + b * b)


def good_features(img, max_corners, quality, min_dist):
    e = _min_eig(img)
    e = np.where(e > e.max() * quality, e, 0).astype(np.float32)
    m = (e != 0) & (e == cv2.dilate(e, None))
    m[0, :] = m[-1, :] = False
    m[:, 0] = m[:, -1] = False
    ys, xs = np.nonzero(m)
    o = np.argsort(-e[ys, xs], kind="stable")
    acc = []
    for x, y in zip(xs[o], ys[o]):
        if all((x - a) ** 2 + (y - b) ** 2 >= min_dist * min_dist for a, b in acc):
            acc.append((x, y))
            if len(acc) == max_corners:
                break
    return np.array(acc, np.float32).reshape(-1, 2)


from skimage import data
print(good_features(data.camera(), 10, 0.01, 10))
