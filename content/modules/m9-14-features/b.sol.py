import math


def gradients(img):
    H, W = len(img), len(img[0])
    gx = [[0.0] * W for _ in range(H)]
    gy = [[0.0] * W for _ in range(H)]
    for y in range(H):
        for x in range(1, W - 1):
            gx[y][x] = img[y][x + 1] - img[y][x - 1]
    for y in range(1, H - 1):
        for x in range(W):
            gy[y][x] = img[y + 1][x] - img[y - 1][x]
    return gx, gy


def cell_histogram(gx, gy, y0, x0, cell=8, nbins=9):
    h = [0.0] * nbins
    w = 180 / nbins
    for y in range(y0, y0 + cell):
        for x in range(x0, x0 + cell):
            a = math.degrees(math.atan2(gy[y][x], gx[y][x])) % 180
            h[min(int(a // w), nbins - 1)] += math.hypot(gx[y][x], gy[y][x])
    return [v / (cell * cell) for v in h]


def l2hys(v, eps=1e-5):
    n = math.sqrt(sum(a * a for a in v) + eps * eps)
    v = [min(a / n, 0.2) for a in v]
    n = math.sqrt(sum(a * a for a in v) + eps * eps)
    return [a / n for a in v]


def hog16(img):
    gx, gy = gradients(img)
    v = []
    for y0 in (0, 8):
        for x0 in (0, 8):
            v += cell_histogram(gx, gy, y0, x0)
    return l2hys(v)


img = [[(x * y) % 17 for x in range(16)] for y in range(16)]
print(len(hog16(img)))
