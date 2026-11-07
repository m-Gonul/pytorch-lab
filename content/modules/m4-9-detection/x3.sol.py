import numpy as np


def iou_m(A, B):
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    ar = lambda X: (X[:, 2:] - X[:, :2]).prod(-1)
    return inter / (ar(A)[:, None] + ar(B)[None] - inter)


def soft_nms(boxes, scores, sigma=0.5, thresh=0.001):
    s = scores.astype(float).copy()
    alive = np.ones(len(s), bool)
    keep, out = [], []
    while alive.any():
        idx = np.flatnonzero(alive)
        i = idx[np.argmax(s[idx])]
        keep.append(i)
        out.append(s[i])
        alive[i] = False
        r = np.flatnonzero(alive)
        if len(r) == 0:
            break
        s[r] *= np.exp(-iou_m(boxes[i][None], boxes[r])[0] ** 2 / sigma)
        alive[r[s[r] < thresh]] = False
    return np.array(keep, int), np.array(out)


def match_ap(dets, gts, thr=0.5):
    recs, n = [], 0
    for (b, s), g in zip(dets, gts):
        n += len(g)
        used = np.zeros(len(g), bool)
        for k in np.argsort(-s, kind="stable"):
            iou = iou_m(b[k][None], g)[0]
            iou[used] = -1
            m = int(iou.argmax())
            ok = iou[m] >= thr
            if ok:
                used[m] = True
            recs.append((s[k], ok))
    recs.sort(key=lambda r: -r[0])
    tp = np.array([r[1] for r in recs], float)
    c = np.cumsum(tp)
    p = np.maximum.accumulate((c / np.arange(1, len(tp) + 1))[::-1])[::-1]
    return float(np.sum(np.diff(c / n, prepend=0) * p))
