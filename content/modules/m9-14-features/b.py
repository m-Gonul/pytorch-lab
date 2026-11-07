import math


def gradients(img):
    ...


def cell_histogram(gx, gy, y0, x0, cell=8, nbins=9):
    ...


def l2hys(v, eps=1e-5):
    ...


def hog16(img):
    ...


img = [[(x * y) % 17 for x in range(16)] for y in range(16)]
print(len(hog16(img)))
