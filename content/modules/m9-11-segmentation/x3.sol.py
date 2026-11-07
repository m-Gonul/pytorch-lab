import numpy as np
from collections import deque


def region_grow(img, seeds, T):
    H, W = img.shape
    f = img.astype(np.int64)
    lab = np.zeros((H, W), np.int32)
    for n, (sy, sx) in enumerate(seeds, 1):
        if lab[sy, sx]:
            continue
        v = f[sy, sx]
        lab[sy, sx] = n
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and not lab[yy, xx] and abs(f[yy, xx] - v) <= T:
                        lab[yy, xx] = n
                        q.append((yy, xx))
    return lab


from skimage import data
print(np.bincount(region_grow(data.coins(), [(100, 100), (50, 300)], 20).ravel()))
