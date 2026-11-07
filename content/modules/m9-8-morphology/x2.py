import numpy as np
from scipy import ndimage


def skeleton(A, B):
    ...


def reconstruct(parts, B):
    ...


A = np.zeros((20, 30), bool); A[4:16, 3:27] = True
S, parts = skeleton(A, np.ones((3, 3), bool))
print(S.sum(), len(parts))
