import math


def bilinear_at(img, x, y):
    H, W = len(img), len(img[0])
    x = min(max(x, 0.0), W - 1.0)
    y = min(max(y, 0.0), H - 1.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    a, b = x - x0, y - y0
    top = img[y0][x0] * (1 - a) + img[y0][x1] * a
    bot = img[y1][x0] * (1 - a) + img[y1][x1] * a
    return top * (1 - b) + bot * b


def resize_nearest(img, H2, W2):
    H, W = len(img), len(img[0])
    return [[img[int(math.floor(y * H / H2))][int(math.floor(x * W / W2))] for x in range(W2)] for y in range(H2)]


def resize_bilinear(img, H2, W2):
    H, W = len(img), len(img[0])
    sy, sx = H / H2, W / W2
    return [[bilinear_at(img, (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5) for x in range(W2)] for y in range(H2)]


I = [[10.0, 20.0], [30.0, 40.0]]
print(bilinear_at(I, 0.25, 0.5))          # 22.5
print(resize_bilinear(I, 4, 4)[0])
