import math


def quantize_sym(xs, bits=8):
    ...


def quantize_affine(xs, bits=8):
    ...


def dequantize(q, s, z=0):
    ...


def magnitude_prune(ws, sparsity):
    ...


print(quantize_sym([0.5, -1.2, 2.54]))           # ([25, -60, 127], 0.02)
print(quantize_affine([-1.0, 0.5, 3.0]))         # ([0, 96, 255], 0.01569, 64)
print(magnitude_prune([0.1, -2, 0.3, 1.5], 0.5))
