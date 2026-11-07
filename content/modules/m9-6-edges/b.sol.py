import math
from collections import deque


def nms(mag, gx, gy):
    H, W = len(mag), len(mag[0])
    at = lambda y, x: mag[y][x] if 0 <= y < H and 0 <= x < W else 0
    out = [[0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            a = math.degrees(math.atan2(gy[y][x], gx[y][x])) % 180
            if a < 22.5 or a >= 157.5:
                n1, n2 = at(y, x - 1), at(y, x + 1)
            elif a < 67.5:
                n1, n2 = at(y - 1, x - 1), at(y + 1, x + 1)
            elif a < 112.5:
                n1, n2 = at(y - 1, x), at(y + 1, x)
            else:
                n1, n2 = at(y - 1, x + 1), at(y + 1, x - 1)
            K = mag[y][x]
            out[y][x] = 0 if (K < n1 or K < n2) else K
    return out


def hysteresis(gn, lo, hi):
    H, W = len(gn), len(gn[0])
    out = [[0] * W for _ in range(H)]
    q = deque()
    for y in range(H):
        for x in range(W):
            if gn[y][x] >= hi:
                out[y][x] = 1
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v, u = y + dy, x + dx
                if 0 <= v < H and 0 <= u < W and not out[v][u] and lo <= gn[v][u] < hi:
                    out[v][u] = 1
                    q.append((v, u))
    return out


def zero_crossings(g, thr):
    H, W = len(g), len(g[0])
    out = [[0] * W for _ in range(H)]
    pairs = [((0, -1), (0, 1)), ((-1, 0), (1, 0)), ((-1, -1), (1, 1)), ((-1, 1), (1, -1))]
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            for (a, b), (c, d) in pairs:
                p, n = g[y + a][x + b], g[y + c][x + d]
                if p * n < 0 and abs(p - n) > thr:
                    out[y][x] = 1
                    break
    return out


print(hysteresis([[0, 40, 120, 60, 0, 70, 0]], 50, 100))
