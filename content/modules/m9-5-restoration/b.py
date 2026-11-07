def median_filter(img, k):
    ...


def cw_median(img, k, w):
    ...


def adaptive_median(img, smax=7):
    ...


img = [[10, 12, 11], [255, 13, 0], [12, 11, 14]]
print(median_filter(img, 3)[1][1], cw_median(img, 3, 3)[1][1])
