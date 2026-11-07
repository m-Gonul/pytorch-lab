import numpy as np


def ks_scan(x):
    x = np.array(x)
    s = 1
    while s < len(x):
        x[s:] = x[s:] + x[:-s]
        s *= 2
    return x


def bk_scan(x):
    a = np.array(x)
    n = len(a)
    s = 1
    while s < n:
        i = np.arange(2 * s - 1, n, 2 * s)
        a[i] += a[i - s]
        s *= 2
    s = n // 4
    while s >= 1:
        j = np.arange(2 * s - 1, n - s, 2 * s)
        a[j + s] += a[j]
        s //= 2
    return a


def compact(x, keep):
    f = np.asarray(keep).astype(np.int64)
    pos = np.cumsum(f) - f
    out = np.empty(int(f.sum()), dtype=np.asarray(x).dtype)
    out[pos[keep]] = np.asarray(x)[keep]
    return out


def radix_sort(keys, bits=16):
    k = np.array(keys)
    for b in range(bits):
        bit = (k >> b) & 1
        z = 1 - bit
        n0 = z.sum()
        dest = np.where(bit == 1, n0 + np.cumsum(bit) - bit, np.cumsum(z) - z)
        out = np.empty_like(k)
        out[dest] = k
        k = out
    return k


def linear_recurrence(a, b):
    A, B = np.array(a, float), np.array(b, float)
    s = 1
    while s < len(A):
        A2 = A[s:] * A[:-s]
        B2 = A[s:] * B[:-s] + B[s:]
        A[s:], B[s:] = A2, B2
        s *= 2
    return B


print(ks_scan(np.array([3, 1, 7, 0, 4, 1, 6, 3])))
