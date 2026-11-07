def integral(img):
    H, W = len(img), len(img[0])
    ii = [[0] * (W + 1) for _ in range(H + 1)]
    for y in range(H):
        s = 0
        for x in range(W):
            s += img[y][x]
            ii[y + 1][x + 1] = ii[y][x + 1] + s
    return ii


def rect_sum(ii, rect):
    (r0, c0), (r1, c1) = rect
    return ii[r1 + 1][c1 + 1] - ii[r0][c1 + 1] - ii[r1 + 1][c0] + ii[r0][c0]


SIGNS = {"type-2-x": (-1, 1), "type-2-y": (-1, 1), "type-3-x": (-1, 1, -1), "type-3-y": (-1, 1, -1), "type-4": (-1, 1, -1, 1)}


def haar(ii, rects, kind):
    return sum(s * rect_sum(ii, r) for s, r in zip(SIGNS[kind], rects))


def features_at(img, coords, kinds):
    ii = integral(img)
    return [haar(ii, c, k) for c, k in zip(coords, kinds)]


ii = integral([[1, 2], [3, 4]])
print(ii, rect_sum(ii, [(0, 0), (1, 1)]))
