import math


def quantize_sym(xs, bits=8):
    qmax = 2 ** (bits - 1) - 1
    m = max(abs(x) for x in xs)
    s = m / qmax if m > 0 else 1.0
    return [max(-qmax, min(qmax, round(x / s))) for x in xs], s


def quantize_affine(xs, bits=8):
    lo, hi = min(min(xs), 0.0), max(max(xs), 0.0)
    qmax = 2 ** bits - 1
    s = (hi - lo) / qmax if hi > lo else 1.0
    z = round(-lo / s)
    return [max(0, min(qmax, round(x / s) + z)) for x in xs], s, z


def dequantize(q, s, z=0):
    return [(v - z) * s for v in q]


def magnitude_prune(ws, sparsity):
    k = int(sparsity * len(ws))
    drop = set(sorted(range(len(ws)), key=lambda i: (abs(ws[i]), i))[:k])
    return [0.0 if i in drop else w for i, w in enumerate(ws)]


print(quantize_sym([0.5, -1.2, 2.54]))           # ([25, -60, 127], 0.02)
print(quantize_affine([-1.0, 0.5, 3.0]))         # ([0, 96, 255], 0.01569, 64)
print(magnitude_prune([0.1, -2, 0.3, 1.5], 0.5))
