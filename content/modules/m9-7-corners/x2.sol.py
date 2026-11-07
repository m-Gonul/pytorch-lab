import numpy as np

CIRCLE = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]


def fast(img, t):
    H, W = img.shape
    f = img.astype(np.int16)
    c = f[3:H - 3, 3:W - 3]
    ring = np.stack([f[3 + dy:H - 3 + dy, 3 + dx:W - 3 + dx] for dx, dy in CIRCLE])
    out = np.zeros((H, W), bool)
    for m in (ring - c > t, ring - c < -t):
        mm = np.concatenate([m, m[:8]]).astype(np.int32)
        cs = np.concatenate([np.zeros((1,) + mm.shape[1:], np.int32), mm.cumsum(0)])
        out[3:H - 3, 3:W - 3] |= ((cs[9:] - cs[:-9]) == 9).any(0)
    return out


from skimage import data
print(fast(data.camera(), 20).sum())
