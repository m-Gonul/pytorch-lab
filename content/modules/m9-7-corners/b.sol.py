def _grad(img, y, x):
    return img[y][x + 1] - img[y][x - 1], img[y + 1][x] - img[y - 1][x]


def harris_matrix(patch):
    n = len(patch)
    a = b = c = 0
    for y in range(1, n - 1):
        for x in range(1, n - 1):
            fx, fy = _grad(patch, y, x)
            a += fx * fx
            b += fx * fy
            c += fy * fy
    return [[a, b], [b, c]]


def harris_response(img, k, r=1):
    H, W = len(img), len(img[0])
    out = [[0.0] * W for _ in range(H)]
    for y in range(r + 1, H - r - 1):
        for x in range(r + 1, W - r - 1):
            a = b = c = 0
            for v in range(y - r, y + r + 1):
                for u in range(x - r, x + r + 1):
                    fx, fy = _grad(img, v, u)
                    a += fx * fx
                    b += fx * fy
                    c += fy * fy
            out[y][x] = a * c - b * b - k * (a + c) ** 2
    return out


def local_maxima(R, thr, r):
    H, W = len(R), len(R[0])
    pts = []
    for y in range(H):
        for x in range(W):
            v = R[y][x]
            if v <= thr:
                continue
            if all(v >= R[yy][xx] for yy in range(max(0, y - r), min(H, y + r + 1)) for xx in range(max(0, x - r), min(W, x + r + 1))):
                pts.append((x, y))
    return pts


left = [[0, 255, 255, 255, 255], [0, 0, 255, 255, 255], [0, 0, 0, 255, 255], [0, 0, 0, 0, 255], [0, 0, 0, 0, 0]]
print(harris_matrix(left))
