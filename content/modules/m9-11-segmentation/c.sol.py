import numpy as np


def kmeans_np(Z, init, max_iter=300):
    m = np.array(init, np.float64)
    K = len(m)
    for _ in range(max_iter):
        lab = ((Z[:, None, :] - m[None]) ** 2).sum(-1).argmin(1)
        cnt = np.bincount(lab, minlength=K)
        s = np.zeros_like(m)
        np.add.at(s, lab, Z)
        new = np.where(cnt[:, None] > 0, s / np.maximum(cnt, 1)[:, None], m)
        if np.array_equal(new, m):
            break
        m = new
    return lab, m


def quantize(img, k, seed=0):
    Z = img.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(seed)
    lab, m = kmeans_np(Z, Z[rng.choice(len(Z), k, replace=False)])
    out = np.clip(np.rint(m[lab]), 0, 255).astype(np.uint8).reshape(img.shape)
    return out, lab.reshape(img.shape[:2])


from skimage import data
out, lab = quantize(data.astronaut()[::4, ::4], 8)
print(out.shape, lab.max())
