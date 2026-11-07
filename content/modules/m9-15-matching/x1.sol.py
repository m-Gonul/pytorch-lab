import numpy as np
import cv2


def best_match(img, t, method):
    r = cv2.matchTemplate(img, t, method)
    i = np.argmin(r) if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else np.argmax(r)
    y, x = np.unravel_index(i, r.shape)
    return int(x), int(y)


def multiscale(img, t, scales):
    best = None
    for s in scales:
        ts = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if ts.shape[0] > img.shape[0] or ts.shape[1] > img.shape[1]:
            continue
        r = cv2.matchTemplate(img, ts, cv2.TM_CCOEFF_NORMED)
        y, x = np.unravel_index(np.argmax(r), r.shape)
        if best is None or r[y, x] > best[3]:
            best = (int(x), int(y), s, float(r[y, x]))
    return best


from skimage import data
im = data.camera().astype(np.float32)
print(best_match(im, im[100:140, 200:250].copy(), cv2.TM_CCOEFF_NORMED))
