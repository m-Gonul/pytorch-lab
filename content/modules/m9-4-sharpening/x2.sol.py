import numpy as np
import cv2


def pipeline(img):
    f = img.astype(np.float64)
    L = cv2.filter2D(f, -1, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float64))
    S1 = f - L
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    G = np.abs(gx) + np.abs(gy)
    Gs = cv2.blur(G, (5, 5))
    P = (-L) * (Gs / Gs.max())
    S2 = f + P
    out = np.rint(255 * (np.clip(S2, 0, 255) / 255) ** 0.5).astype(np.uint8)
    return dict(L=L, S1=S1, G=G, Gs=Gs, P=P, S2=S2, out=out)


from skimage import data
r = pipeline(data.camera())
print({k: (v.dtype, round(float(v.mean()), 1)) for k, v in r.items()})
