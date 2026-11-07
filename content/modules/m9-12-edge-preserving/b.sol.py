import math


def clip_histogram(hist, clip):
    n = len(hist)
    excess = sum(max(v - clip, 0) for v in hist)
    out = [min(v, clip) for v in hist]
    batch, res = excess // n, excess - (excess // n) * n
    out = [v + batch for v in out]
    if res:
        step = max(n // res, 1)
        i = 0
        while i < n and res > 0:
            out[i] += 1
            i += step
            res -= 1
    return out


def tile_lut(hist, clip_limit, total):
    n = len(hist)
    if clip_limit > 0:
        hist = clip_histogram(hist, max(int(clip_limit * total / n), 1))
    lut, s = [], 0
    for v in hist:
        s += v
        lut.append(round(s * (n - 1) / total))
    return lut


def bilateral(img, d, sigma_color, sigma_space):
    H, W = len(img), len(img[0])
    r = d // 2 if d > 0 else round(1.5 * sigma_space)
    r = max(r, 1)
    refl = lambda i, n: -i if i < 0 else (2 * n - 2 - i if i >= n else i)
    offs = [(i, j, math.exp(-(i * i + j * j) / (2 * sigma_space ** 2))) for i in range(-r, r + 1) for j in range(-r, r + 1) if math.sqrt(i * i + j * j) <= r]
    out = [[0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            c = img[y][x]
            num = den = 0.0
            for i, j, ws in offs:
                v = img[refl(y + i, H)][refl(x + j, W)]
                w = ws * math.exp(-((v - c) ** 2) / (2 * sigma_color ** 2))
                num += w * v
                den += w
            out[y][x] = min(max(round(num / den), 0), 255)
    return out


print(clip_histogram([10, 2, 0, 0, 0, 0, 0, 4], 4))
