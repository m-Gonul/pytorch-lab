import numpy as np


def hough_lines(edges, thetas):
    ...


def top_lines(acc, thetas, dists, k, nb=(10, 10)):
    ...


thetas = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
e = np.zeros((50, 50), bool); e[25, :] = True
acc, d = hough_lines(e, thetas)
print(top_lines(acc, thetas, d, 1))
