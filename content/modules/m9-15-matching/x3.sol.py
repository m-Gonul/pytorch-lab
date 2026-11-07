import numpy as np
import cv2


def gaussian_pyramid(img, levels):
    G = [img.astype(np.float64)]
    for _ in range(levels - 1):
        G.append(cv2.pyrDown(G[-1]))
    return G


def laplacian_pyramid(img, levels):
    G = gaussian_pyramid(img, levels)
    L = [G[i] - cv2.pyrUp(G[i + 1], dstsize=(G[i].shape[1], G[i].shape[0])) for i in range(levels - 1)]
    return L + [G[-1]]


def collapse(L):
    x = L[-1]
    for lap in reversed(L[:-1]):
        x = cv2.pyrUp(x, dstsize=(lap.shape[1], lap.shape[0])) + lap
    return x


def blend(a, b, mask, levels=5):
    La, Lb = laplacian_pyramid(a, levels), laplacian_pyramid(b, levels)
    Gm = gaussian_pyramid(mask.astype(np.float64), levels)
    if a.ndim == 3:
        Gm = [g[..., None] for g in Gm]
    return collapse([la * g + lb * (1 - g) for la, lb, g in zip(La, Lb, Gm)])


from skimage import data
x = data.camera().astype(np.float64)
print(np.abs(collapse(laplacian_pyramid(x, 5)) - x).max())
