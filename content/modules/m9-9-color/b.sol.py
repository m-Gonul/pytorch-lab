import math


def rgb_to_hsi(r, g, b):
    i = (r + g + b) / 3
    s = 0.0 if r + g + b == 0 else 1 - 3 * min(r, g, b) / (r + g + b)
    den = math.sqrt((r - g) ** 2 + (r - b) * (g - b))
    if den == 0:
        return 0.0, s, i
    th = math.degrees(math.acos(max(-1.0, min(1.0, 0.5 * ((r - g) + (r - b)) / den))))
    return (th if b <= g else 360 - th), s, i


def _sector(h, s, i):
    x = i * (1 - s)
    y = i * (1 + s * math.cos(math.radians(h)) / math.cos(math.radians(60 - h)))
    return x, y, 3 * i - (x + y)


def hsi_to_rgb(h, s, i):
    if h < 120:
        b, r, g = _sector(h, s, i)
    elif h < 240:
        r, g, b = _sector(h - 120, s, i)
    else:
        g, b, r = _sector(h - 240, s, i)
    return r, g, b


def rgb_to_cmyk(r, g, b):
    c, m, y = 1 - r, 1 - g, 1 - b
    k = min(c, m, y)
    if k == 1:
        return 0.0, 0.0, 0.0, 1.0
    return (c - k) / (1 - k), (m - k) / (1 - k), (y - k) / (1 - k), k


def cmyk_to_rgb(c, m, y, k):
    return 1 - (c * (1 - k) + k), 1 - (m * (1 - k) + k), 1 - (y * (1 - k) + k)


print(rgb_to_hsi(0.8, 0.4, 0.2))
