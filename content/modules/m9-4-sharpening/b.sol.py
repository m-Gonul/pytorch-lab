def diff1(f):
    return [f[i + 1] - f[i] for i in range(len(f) - 1)]


def diff2(f):
    return [f[i + 1] + f[i - 1] - 2 * f[i] for i in range(1, len(f) - 1)]


def _r101(i, n):
    if i < 0:
        return -i
    if i >= n:
        return 2 * n - 2 - i
    return i


def _at(img, y, x):
    H, W = len(img), len(img[0])
    return img[_r101(y, H)][_r101(x, W)]


def laplacian(img):
    H, W = len(img), len(img[0])
    return [[_at(img, y - 1, x) + _at(img, y + 1, x) + _at(img, y, x - 1) + _at(img, y, x + 1) - 4 * img[y][x]
             for x in range(W)] for y in range(H)]


def sobel(img):
    H, W = len(img), len(img[0])
    a = lambda y, x: _at(img, y, x)
    gx = [[(a(y - 1, x + 1) + 2 * a(y, x + 1) + a(y + 1, x + 1)) - (a(y - 1, x - 1) + 2 * a(y, x - 1) + a(y + 1, x - 1))
           for x in range(W)] for y in range(H)]
    gy = [[(a(y + 1, x - 1) + 2 * a(y + 1, x) + a(y + 1, x + 1)) - (a(y - 1, x - 1) + 2 * a(y - 1, x) + a(y - 1, x + 1))
           for x in range(W)] for y in range(H)]
    return gx, gy


f = [5, 5, 4, 3, 2, 1, 0, 0, 0, 6, 0, 0]
print(diff1(f), diff2(f))
