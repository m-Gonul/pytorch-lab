import numpy as np
from scipy import ndimage


def skeleton(A, B):
    parts = []
    E = A.copy()
    while E.any():
        parts.append(E & ~ndimage.binary_opening(E, B))
        E = ndimage.binary_erosion(E, B)
    S = np.zeros_like(A)
    for p in parts:
        S |= p
    return S, parts


def reconstruct(parts, B):
    out = np.zeros_like(parts[0])
    for k, p in enumerate(parts):
        out |= ndimage.binary_dilation(p, B, iterations=k) if k else p
    return out


A = np.zeros((20, 30), bool); A[4:16, 3:27] = True
S, parts = skeleton(A, np.ones((3, 3), bool))
print(S.sum(), len(parts))
