import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import KMeans


def superpixel_kmeans(img, n_segments, k, seed=0):
    sp = slic(img, n_segments=n_segments, compactness=10, start_label=0)
    n = sp.max() + 1
    cnt = np.bincount(sp.ravel(), minlength=n)
    means = np.stack([np.bincount(sp.ravel(), weights=img[..., c].ravel().astype(float), minlength=n) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    rng = np.random.default_rng(seed)
    km = KMeans(k, init=means[rng.choice(n, k, replace=False)], n_init=1).fit(means)
    return km.labels_[sp], sp


def pixel_kmeans(img, k, seed=0):
    Z = img.reshape(-1, 3).astype(float)
    rng = np.random.default_rng(seed)
    km = KMeans(k, init=Z[rng.choice(len(Z), k, replace=False)], n_init=1).fit(Z)
    return km.labels_.reshape(img.shape[:2])


from skimage import data
lab, sp = superpixel_kmeans(data.coffee()[::2, ::2], 300, 4)
print(lab.max(), sp.max())
