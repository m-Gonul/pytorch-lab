import torch


def linear_scan(a, b):
    """h_t = a_t·h_{t−1} + b_t, h_0 = 0. a, b: (T, …). Yalnız offset döngüsü."""
    A, B = a.clone(), b.clone()
    T, off = a.shape[0], 1
    while off < T:
        B_new = B.clone()
        B_new[off:] = A[off:] * B[:-off] + B[off:]
        A_new = A.clone()
        A_new[off:] = A[off:] * A[:-off]
        A, B = A_new, B_new
        off *= 2
    return B


a = torch.full((8,), 0.5)
b = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
print(linear_scan(a, b))     # [1, 0.5, 0.25, …]
