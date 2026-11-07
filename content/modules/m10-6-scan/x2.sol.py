import numpy as np


def max_subarray(x):
    P = np.concatenate([[0], np.cumsum(x)])
    m = np.minimum.accumulate(P[:-1])
    return (P[1:] - m).max()


def kadane(x):
    best = cur = x[0]
    for v in x[1:]:
        cur = max(v, cur + v)
        best = max(best, cur)
    return best


print(max_subarray(np.array([-2, 1, -3, 4, -1, 2, 1, -5, 4])))
