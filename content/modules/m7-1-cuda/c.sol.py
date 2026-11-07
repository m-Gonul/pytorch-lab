import numpy as np


def intensity(flops, bytes_):
    return np.asarray(flops, float) / np.asarray(bytes_, float)


def attainable(peak, bw, I):
    return np.minimum(peak, bw * np.asarray(I, float))


def op_costs(n):
    return {
        "add": (n, 3 * n * 4),
        "softmax": (5 * n * n, 2 * n * n * 4),
        "matmul": (2 * n ** 3, 3 * n * n * 4),
    }


def matmul_blocked(A, B, T):
    n, m = A.shape[0], B.shape[1]
    C = np.zeros((n, m))
    for i in range(0, n, T):
        for j in range(0, m, T):
            for k in range(0, A.shape[1], T):
                C[i:i + T, j:j + T] += A[i:i + T, k:k + T] @ B[k:k + T, j:j + T]
    return C


A100 = dict(peak=19.5e12, bw=1.555e12)
for name, (f, b) in op_costs(4096).items():
    I = intensity(f, b)
    print(name, round(I, 3), f"{attainable(A100['peak'], A100['bw'], I) / 1e12:.2f} TFLOP/s")
