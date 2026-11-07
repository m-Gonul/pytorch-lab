def correlate2d(img, k):
    ...


def convolve2d(img, k):
    ...


def box_filter(img, size):
    ...


img = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
w = [[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]
print(correlate2d(img, w)[1][1], convolve2d(img, w)[1][1])   # 1.0 9.0
