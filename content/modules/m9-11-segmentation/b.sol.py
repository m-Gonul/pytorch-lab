KERNELS = {
    "h": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
    "v": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
    "d1": [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
    "d2": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
}
LAP = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]


def _corr(img, k, y, x):
    return sum(k[i][j] * img[y + i - 1][x + j - 1] for i in range(3) for j in range(3))


def point_detect(img, T):
    H, W = len(img), len(img[0])
    return [[int(0 < y < H - 1 and 0 < x < W - 1 and abs(_corr(img, LAP, y, x)) >= T) for x in range(W)] for y in range(H)]


def line_detect(img, kind, T):
    H, W = len(img), len(img[0])
    k = KERNELS[kind]
    return [[int(0 < y < H - 1 and 0 < x < W - 1 and _corr(img, k, y, x) >= T) for x in range(W)] for y in range(H)]


def kmeans(points, init, max_iter=300):
    m = [list(map(float, c)) for c in init]
    labels = []
    for _ in range(max_iter):
        labels = []
        for p in points:
            d = [sum((a - b) ** 2 for a, b in zip(p, c)) for c in m]
            labels.append(d.index(min(d)))
        new = []
        for k in range(len(m)):
            mem = [p for p, l in zip(points, labels) if l == k]
            new.append([sum(col) / len(mem) for col in zip(*mem)] if mem else m[k])
        if new == m:
            break
        m = new
    return labels, m


print(kmeans([[1], [2], [3], [10], [11], [12]], [[1], [2]]))
