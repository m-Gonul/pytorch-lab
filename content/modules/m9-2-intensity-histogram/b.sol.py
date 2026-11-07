def histogram(img):
    h = [0] * 256
    for row in img:
        for v in row:
            h[v] += 1
    return h


def _cdf(img):
    h = histogram(img)
    c, s = [], 0
    for n in h:
        s += n
        c.append(s)
    return h, c


def equalize_class(img):
    h, c = _cdf(img)
    N = c[-1]
    lut = [round(255 * ck / N) for ck in c]
    return [[lut[v] for v in row] for row in img]


def equalize_opencv(img):
    h, c = _cdf(img)
    N = c[-1]
    cmin = next(n for n in h if n > 0)
    if N == cmin:
        return [list(row) for row in img]
    lut = [min(255, max(0, round(255 * (ck - cmin) / (N - cmin)))) for ck in c]
    return [[lut[v] for v in row] for row in img]


img = [[0, 0, 1, 1], [2, 2, 2, 3]]
print(histogram(img)[:4], equalize_class(img), equalize_opencv(img))
