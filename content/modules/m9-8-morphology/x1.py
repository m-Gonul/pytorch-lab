import numpy as np
from scipy import ndimage


def fill_holes(A):
    ...


def components(A):
    ...


A = np.zeros((7, 7), bool); A[1:6, 1:6] = True; A[3, 3] = False
print(fill_holes(A).sum())
