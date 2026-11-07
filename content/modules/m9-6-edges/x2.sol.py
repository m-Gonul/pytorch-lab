import numpy as np


def hough_lines(edges, thetas):
    H, W = edges.shape
    off = int(np.ceil(np.hypot(H, W)))
    y, x = np.nonzero(edges)
    r = np.round(np.outer(x, np.cos(thetas)) + np.outer(y, np.sin(thetas))).astype(np.int64) + off
    acc = np.zeros((2 * off + 1, len(thetas)), np.int64)
    np.add.at(acc, (r, np.broadcast_to(np.arange(len(thetas)), r.shape)), 1)
    return acc, np.linspace(-off, off, 2 * off + 1)


def top_lines(acc, thetas, dists, k, nb=(10, 10)):
    a = acc.copy()
    out = []
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(a), a.shape)
        out.append((dists[i], thetas[j]))
        a[max(i - nb[0], 0):i + nb[0] + 1, max(j - nb[1], 0):j + nb[1] + 1] = 0
    return out


thetas = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
e = np.zeros((50, 50), bool); e[25, :] = True
acc, d = hough_lines(e, thetas)
print(top_lines(acc, thetas, d, 1))
