import numpy as np


def separate(k, tol=1e-8):
    U, S, Vt = np.linalg.svd(np.asarray(k, float))
    if S[0] == 0 or (len(S) > 1 and S[1] / S[0] >= tol):
        return None
    return np.sqrt(S[0]) * U[:, 0], np.sqrt(S[0]) * Vt[0]


import cv2
g = cv2.getGaussianKernel(5, 0) @ cv2.getGaussianKernel(5, 0).T
print(separate(g))
