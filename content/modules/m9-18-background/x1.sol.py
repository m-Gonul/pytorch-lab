import numpy as np
import cv2


def diff2(prev, cur, T):
    return np.abs(cur.astype(np.int16) - prev) > T


def diff3(prev, cur, nxt, T):
    return diff2(prev, cur, T) & diff2(cur, nxt, T)


def clean(mask, k=3):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = mask.astype(np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, se), cv2.MORPH_CLOSE, se) > 0


def boxes(mask, min_area=20):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    b = [tuple(int(v) for v in st[i, :4]) + (int(st[i, 4]),) for i in range(1, n) if st[i, 4] >= min_area]
    return [x[:4] for x in sorted(b, key=lambda x: -x[4])]


print("fark")
