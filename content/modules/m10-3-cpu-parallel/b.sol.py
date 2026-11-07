def block_scan(a, p):
    n = len(a)
    bounds = [(n * t // p, n * (t + 1) // p) for t in range(p)]
    out = [0] * n
    totals = []
    for lo, hi in bounds:
        s = 0
        for i in range(lo, hi):
            s += a[i]
            out[i] = s
        totals.append(s)
    off, acc = [], 0
    for s in totals:
        off.append(acc)
        acc += s
    for (lo, hi), o in zip(bounds, off):
        for i in range(lo, hi):
            out[i] += o
    return out


def integral_image(img, p):
    H, W = len(img), len(img[0])
    I = [row[:] for row in img]
    for t in range(p):
        for y in range(H * t // p, H * (t + 1) // p):
            for x in range(1, W):
                I[y][x] += I[y][x - 1]
    for t in range(p):
        for x in range(W * t // p, W * (t + 1) // p):
            for y in range(1, H):
                I[y][x] += I[y - 1][x]
    return I


def conv_bands(img, k, p):
    H, W, K = len(img), len(img[0]), len(k)
    r = K // 2
    out = [[0] * W for _ in range(H)]
    for t in range(p):
        y0, y1 = H * t // p, H * (t + 1) // p
        lo, hi = max(0, y0 - r), min(H, y1 + r)
        band = img[lo:hi]
        for y in range(y0, y1):
            for x in range(W):
                acc = 0.0
                for u in range(K):
                    yy = y + u - r
                    if 0 <= yy < H:
                        row = band[yy - lo]
                        for v in range(K):
                            xx = x + v - r
                            if 0 <= xx < W:
                                acc += k[u][v] * row[xx]
                out[y][x] = min(255, max(0, int(acc + 0.5)))
    return out


def matmul_rows(A, B, p):
    n, m, q = len(A), len(B), len(B[0])
    C = [[0.0] * q for _ in range(n)]
    for t in range(p):
        for i in range(n * t // p, n * (t + 1) // p):
            Ci = C[i]
            for kk in range(m):
                a = A[i][kk]
                Bk = B[kk]
                for j in range(q):
                    Ci[j] += a * Bk[j]
    return C


print(block_scan(list(range(1, 9)), 4))
