def integral(img):
    ...


def rect_sum(ii, rect):
    ...


def haar(ii, rects, kind):
    ...


def features_at(img, coords, kinds):
    ...


ii = integral([[1, 2], [3, 4]])
print(ii, rect_sum(ii, [(0, 0), (1, 1)]))
