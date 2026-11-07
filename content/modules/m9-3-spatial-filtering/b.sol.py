def correlate2d(img, k):
    H, W = len(img), len(img[0])
    m, n = len(k), len(k[0])
    a, b = m // 2, n // 2
    out = [[0.0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            s = 0.0
            for i in range(m):
                for j in range(n):
                    yy, xx = y + i - a, x + j - b
                    if 0 <= yy < H and 0 <= xx < W:
                        s += k[i][j] * img[yy][xx]
            out[y][x] = s
    return out


def convolve2d(img, k):
    flipped = [row[::-1] for row in k[::-1]]
    return correlate2d(img, flipped)


def box_filter(img, size):
    k = [[1.0 / (size * size)] * size for _ in range(size)]
    return correlate2d(img, k)


img = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
w = [[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]
print(correlate2d(img, w)[1][1], convolve2d(img, w)[1][1])   # 1.0 9.0
