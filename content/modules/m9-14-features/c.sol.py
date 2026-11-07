import numpy as np


def hog(img, nbins=9, cell=8, block=2, eps=1e-5):
    f = img.astype(np.float64)
    gx = np.zeros_like(f); gy = np.zeros_like(f)
    gx[:, 1:-1] = f[:, 2:] - f[:, :-2]
    gy[1:-1] = f[2:] - f[:-2]
    ncy, ncx = f.shape[0] // cell, f.shape[1] // cell
    m = np.hypot(gx, gy)[:ncy * cell, :ncx * cell]
    a = (np.rad2deg(np.arctan2(gy, gx)) % 180)[:ncy * cell, :ncx * cell]
    b = np.minimum((a // (180 / nbins)).astype(np.int64), nbins - 1)
    cy = np.arange(ncy * cell)[:, None] // cell
    cx = np.arange(ncx * cell)[None, :] // cell
    idx = ((cy * ncx + cx) * nbins + b).ravel()
    hist = np.bincount(idx, weights=m.ravel(), minlength=ncy * ncx * nbins).reshape(ncy, ncx, nbins) / (cell * cell)
    blocks = np.lib.stride_tricks.sliding_window_view(hist, (block, block), axis=(0, 1)).transpose(0, 1, 3, 4, 2)
    v = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + eps ** 2)
    v = np.minimum(v, 0.2)
    v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + eps ** 2)
    return v.ravel()


def fast_hessian(img, L=9):
    f = img.astype(np.float64)
    H, W = f.shape
    l = L // 3
    pad = L
    ii = np.zeros((H + 2 * pad + 1, W + 2 * pad + 1))
    ii[1:, 1:] = np.pad(f, pad).cumsum(0).cumsum(1)
    yy, xx = np.mgrid[:H, :W] + pad

    def box(y0, x0, h, w):
        return ii[yy + y0 + h, xx + x0 + w] - ii[yy + y0, xx + x0 + w] - ii[yy + y0 + h, xx + x0] + ii[yy + y0, xx + x0]

    r = L // 2
    wl = 2 * l - 1
    Dyy = box(-r, -(wl // 2), L, wl) - 3 * box(-r + l, -(wl // 2), l, wl)
    Dxx = box(-(wl // 2), -r, wl, L) - 3 * box(-(wl // 2), -r + l, wl, l)
    Dxy = box(-l, -l, l, l) + box(1, 1, l, l) - box(-l, 1, l, l) - box(1, -l, l, l)
    Dxx, Dyy, Dxy = Dxx / L ** 2, Dyy / L ** 2, Dxy / L ** 2
    return Dxx * Dyy - (0.9 * Dxy) ** 2


from skimage import data
print(hog(data.camera()[:128, :64]).shape)
