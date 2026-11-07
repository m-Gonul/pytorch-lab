import numpy as np


def segmented_scan(x, flags):
    v = np.array(x, float)
    f = np.array(flags, bool)
    s = 1
    while s < len(v):
        v2 = np.where(f[s:], v[s:], v[:-s] + v[s:])
        f2 = f[:-s] | f[s:]
        v[s:], f[s:] = v2, f2
        s *= 2
    return v


print(segmented_scan(np.array([1, 2, 3, 4, 5]), np.array([1, 0, 1, 0, 0])))
