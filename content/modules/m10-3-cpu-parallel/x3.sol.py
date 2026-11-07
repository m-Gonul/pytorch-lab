import numpy as np


def box_filter(img, r):
    H, W = img.shape
    I = np.zeros((H + 1, W + 1), np.int64)
    I[1:, 1:] = np.cumsum(np.cumsum(np.asarray(img, np.int64), 0), 1)
    y0 = np.clip(np.arange(H) - r, 0, H)[:, None]
    y1 = np.clip(np.arange(H) + r + 1, 0, H)[:, None]
    x0 = np.clip(np.arange(W) - r, 0, W)[None, :]
    x1 = np.clip(np.arange(W) + r + 1, 0, W)[None, :]
    return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]


def box_direct(img, r):
    H, W = img.shape
    out = np.zeros((H, W), np.int64)
    for y in range(H):
        for x in range(W):
            out[y, x] = img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].astype(np.int64).sum()
    return out


print(box_filter(np.arange(16).reshape(4, 4), 1))
