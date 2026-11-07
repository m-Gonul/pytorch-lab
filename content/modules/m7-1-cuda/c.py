import numpy as np


def intensity(flops, bytes_):
    ...


def attainable(peak, bw, I):
    ...


def op_costs(n):
    ...


def matmul_blocked(A, B, T):
    ...


A100 = dict(peak=19.5e12, bw=1.555e12)
for name, (f, b) in op_costs(4096).items():
    I = intensity(f, b)
    print(name, round(I, 3), f"{attainable(A100['peak'], A100['bw'], I) / 1e12:.2f} TFLOP/s")
