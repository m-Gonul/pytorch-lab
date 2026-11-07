def tconv_out_size(H, k, s, p, op=0):
    return (H - 1) * s - 2 * p + k + op


def conv_transpose1d(x, w, s=1, p=0):
    k = len(w)
    out = [0.0] * ((len(x) - 1) * s + k)
    for i, xv in enumerate(x):
        for u, wv in enumerate(w):
            out[i * s + u] += xv * wv
    return out[p:len(out) - p]


def conv_transpose2d(x, w, s=1, p=0):
    H, W, k = len(x), len(x[0]), len(w)
    Ho, Wo = (H - 1) * s + k, (W - 1) * s + k
    out = [[0.0] * Wo for _ in range(Ho)]
    for i in range(H):
        for j in range(W):
            for u in range(k):
                for v in range(k):
                    out[i * s + u][j * s + v] += x[i][j] * w[u][v]
    return [row[p:Wo - p] for row in out[p:Ho - p]]


def overlap_counts(n, k, s):
    c = [0] * ((n - 1) * s + k)
    for i in range(n):
        for u in range(k):
            c[i * s + u] += 1
    return c


def unet_sizes(H, depth=4):
    sizes = [H]
    for _ in range(depth):
        H = (H - 4) // 2
        sizes.append(H)
    H -= 4
    sizes.append(H)
    for _ in range(depth):
        H = H * 2 - 4
        sizes.append(H)
    return sizes


def dice_iou(pred, gt):
    a = sum(map(sum, pred))
    b = sum(map(sum, gt))
    inter = sum(p and g for rp, rg in zip(pred, gt) for p, g in zip(rp, rg))
    if a + b == 0:
        return 1.0, 1.0
    return 2 * inter / (a + b), inter / (a + b - inter)


print(conv_transpose1d([1, 2, 3], [1, 2, 1], 2))
