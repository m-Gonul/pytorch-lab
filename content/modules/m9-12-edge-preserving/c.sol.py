import numpy as np
import cv2


def clahe_cv(img, clip=2.0, tiles=(8, 8)):
    H, W = img.shape
    tX, tY = tiles
    if W % tX == 0 and H % tY == 0:
        src = img
    else:
        src = cv2.copyMakeBorder(img, 0, tY - H % tY, 0, tX - W % tX, cv2.BORDER_REFLECT_101)
    th, tw = src.shape[0] // tY, src.shape[1] // tX
    N = th * tw
    T = src.reshape(tY, th, tX, tw).transpose(0, 2, 1, 3).reshape(tY * tX, N).astype(np.int64)
    hist = np.bincount((T + 256 * np.arange(tY * tX)[:, None]).ravel(), minlength=256 * tY * tX).reshape(tY * tX, 256)
    if clip > 0:
        cl = max(int(clip * N / 256), 1)
        ex = np.maximum(hist - cl, 0).sum(1)
        hist = np.minimum(hist, cl) + (ex // 256)[:, None]
        r = ex % 256
        step = np.maximum(256 // np.maximum(r, 1), 1)
        i = np.arange(256)[None]
        hist = hist + ((i % step[:, None] == 0) & (i // step[:, None] < r[:, None]))
    lut = np.rint(hist.cumsum(1).astype(np.float32) * np.float32(255.0 / N)).astype(np.float32).reshape(tY, tX, 256)
    x = np.arange(W, dtype=np.float32) * np.float32(1.0 / tw) - np.float32(0.5)
    y = np.arange(H, dtype=np.float32) * np.float32(1.0 / th) - np.float32(0.5)
    x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
    xa, ya = (x - x1).astype(np.float32), (y - y1).astype(np.float32)
    x2, y2 = np.minimum(x1 + 1, tX - 1), np.minimum(y1 + 1, tY - 1)
    x1, y1 = np.maximum(x1, 0), np.maximum(y1, 0)
    v = img.astype(np.int64)
    L = lambda ty, tx: lut[ty[:, None], tx[None, :], v]
    one = np.float32(1)
    res = (L(y1, x1) * (one - xa) + L(y1, x2) * xa) * (one - ya)[:, None] + (L(y2, x1) * (one - xa) + L(y2, x2) * xa) * ya[:, None]
    return np.clip(np.rint(res), 0, 255).astype(np.uint8)


from skimage import data
print(clahe_cv(data.moon()).mean())
