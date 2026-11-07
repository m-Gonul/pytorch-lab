import numpy as np
import cv2


def match_ratio(d1, d2, ratio=0.8, cross=False):
    if d1.dtype == np.uint8:
        D = np.unpackbits(d1[:, None] ^ d2[None], axis=-1).sum(-1).astype(np.float64)
    else:
        D = np.sqrt(np.maximum((d1 ** 2).sum(1)[:, None] + (d2 ** 2).sum(1)[None] - 2 * d1 @ d2.T, 0))
    o = np.argsort(D, 1)[:, :2]
    best, second = D[np.arange(len(D)), o[:, 0]], D[np.arange(len(D)), o[:, 1]]
    ok = best < ratio * second
    if cross:
        ok &= D.argmin(0)[o[:, 0]] == np.arange(len(D))
    return [(int(i), int(o[i, 0])) for i in np.nonzero(ok)[0]]


def evaluate(img, M, detector):
    H, W = img.shape
    img2 = cv2.warpAffine(img, M, (W, H))
    det = cv2.SIFT_create() if detector == "sift" else cv2.ORB_create(2000)
    k1, d1 = det.detectAndCompute(img, None)
    k2, d2 = det.detectAndCompute(img2, None)
    m = match_ratio(d1, d2)
    p1 = np.array([k1[i].pt for i, _ in m]); p2 = np.array([k2[j].pt for _, j in m])
    q = np.c_[p1, np.ones(len(p1))] @ M.T
    return len(m), int((np.linalg.norm(q - p2, axis=1) < 3).sum())


from skimage import data
print(evaluate(data.camera(), cv2.getRotationMatrix2D((256, 256), 20, 0.8), "sift"))
