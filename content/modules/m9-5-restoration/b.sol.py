def _window(img, y, x, k):
    H, W = len(img), len(img[0])
    r = k // 2
    return [img[min(max(yy, 0), H - 1)][min(max(xx, 0), W - 1)] for yy in range(y - r, y + r + 1) for xx in range(x - r, x + r + 1)]


def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def median_filter(img, k):
    H, W = len(img), len(img[0])
    return [[_median(_window(img, y, x, k)) for x in range(W)] for y in range(H)]


def cw_median(img, k, w):
    H, W = len(img), len(img[0])
    return [[_median(_window(img, y, x, k) + [img[y][x]] * (w - 1)) for x in range(W)] for y in range(H)]


def adaptive_median(img, smax=7):
    H, W = len(img), len(img[0])
    out = [[0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            k = 3
            while True:
                win = _window(img, y, x, k)
                zmin, zmax, zmed = min(win), max(win), _median(win)
                if zmin < zmed < zmax:
                    z = img[y][x]
                    out[y][x] = z if zmin < z < zmax else zmed
                    break
                k += 2
                if k > smax:
                    out[y][x] = zmed
                    break
    return out


img = [[10, 12, 11], [255, 13, 0], [12, 11, 14]]
print(median_filter(img, 3)[1][1], cw_median(img, 3, 3)[1][1])
