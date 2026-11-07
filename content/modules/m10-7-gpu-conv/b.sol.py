def conv_tile_loads(T, K):
    n = (T + K - 1) ** 2
    return n, n / (T * T)


def direct_conv(img, k):
    H, W, K = len(img), len(img[0]), len(k)
    r = K // 2
    out = [[0.0] * W for _ in range(H)]
    reads = 0
    for y in range(H):
        for x in range(W):
            acc = 0.0
            for u in range(K):
                for v in range(K):
                    yy, xx = y + u - r, x + v - r
                    if 0 <= yy < H and 0 <= xx < W:
                        acc += k[u][v] * img[yy][xx]
                        reads += 1
            out[y][x] = acc
    return out, reads


def tiled_conv(img, k, T):
    H, W, K = len(img), len(img[0]), len(k)
    r = K // 2
    out = [[0.0] * W for _ in range(H)]
    reads = 0
    for y0 in range(0, H, T):
        for x0 in range(0, W, T):
            tile = {}
            for yy in range(y0 - r, y0 + T + r):
                for xx in range(x0 - r, x0 + T + r):
                    if 0 <= yy < H and 0 <= xx < W:
                        tile[(yy, xx)] = img[yy][xx]
                        reads += 1
            for y in range(y0, min(H, y0 + T)):
                for x in range(x0, min(W, x0 + T)):
                    acc = 0.0
                    for u in range(K):
                        for v in range(K):
                            val = tile.get((y + u - r, x + v - r))
                            if val is not None:
                                acc += k[u][v] * val
                    out[y][x] = acc
    return out, reads


def tiled_matmul(A, B, T):
    n, m, q = len(A), len(B), len(B[0])
    C = [[0.0] * q for _ in range(n)]
    reads = 0
    for i0 in range(0, n, T):
        for j0 in range(0, q, T):
            for k0 in range(0, m, T):
                At = {(i, kk): A[i][kk] for i in range(i0, min(n, i0 + T)) for kk in range(k0, min(m, k0 + T))}
                Bt = {(kk, j): B[kk][j] for kk in range(k0, min(m, k0 + T)) for j in range(j0, min(q, j0 + T))}
                reads += len(At) + len(Bt)
                for i in range(i0, min(n, i0 + T)):
                    for j in range(j0, min(q, j0 + T)):
                        s = 0.0
                        for kk in range(k0, min(m, k0 + T)):
                            s += At[(i, kk)] * Bt[(kk, j)]
                        C[i][j] += s
    return C, reads


print(conv_tile_loads(16, 5))
