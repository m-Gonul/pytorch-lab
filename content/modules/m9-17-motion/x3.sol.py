import numpy as np


def fruc_repeat(I0, I2):
    return I0.astype(np.float64)


def fruc_average(I0, I2):
    return (I0.astype(np.float64) + I2) / 2


def fruc_block(I0, I2, bs=8, R=8):
    A, B = I0.astype(np.float64), I2.astype(np.float64)
    H, W = A.shape
    out = (A + B) / 2
    for by in range(0, H - bs + 1, bs):
        for bx in range(0, W - bs + 1, bs):
            best = None
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    y0, x0, y2, x2 = by + dy, bx + dx, by - dy, bx - dx
                    if min(y0, x0, y2, x2) < 0 or max(y0, y2) > H - bs or max(x0, x2) > W - bs:
                        continue
                    a, b = A[y0:y0 + bs, x0:x0 + bs], B[y2:y2 + bs, x2:x2 + bs]
                    s = np.abs(a - b).sum()
                    if best is None or s < best[0]:
                        best = (s, a, b)
            out[by:by + bs, bx:bx + bs] = (best[1] + best[2]) / 2
    return out


print("FRUC")
