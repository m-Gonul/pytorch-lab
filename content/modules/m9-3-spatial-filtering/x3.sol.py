ops = 0


def _run1d(row, size):
    global ops
    r = size // 2
    n = len(row)
    get = lambda i: row[i] if 0 <= i < n else 0.0
    s = 0.0
    for i in range(-r, r + 1):
        s += get(i)
        ops += 1
    out = [s]
    for x in range(1, n):
        s += get(x + r) - get(x - r - 1)
        ops += 2
        out.append(s)
    return out


def box_running(img, size):
    H, W = len(img), len(img[0])
    rows = [_run1d(row, size) for row in img]
    cols = [_run1d([rows[y][x] for y in range(H)], size) for x in range(W)]
    return [[cols[x][y] / (size * size) for x in range(W)] for y in range(H)]


img = [[float(x + y) for x in range(8)] for y in range(6)]
print(box_running(img, 3)[2][2], ops)
