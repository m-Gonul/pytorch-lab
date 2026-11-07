import numpy as np


def iou_matrix(A, B):
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda X: (X[:, 2:] - X[:, :2]).prod(-1)
    return inter / (area(A)[:, None] + area(B)[None] - inter)


def nms(boxes, scores, thr):
    order = np.argsort(-scores, kind="stable")
    M = iou_matrix(boxes[order], boxes[order])
    alive = np.ones(len(order), bool)
    for i in range(len(order)):
        if alive[i]:
            alive[i + 1:] &= M[i, i + 1:] <= thr
    return order[alive]


def fast_nms(boxes, scores, thr):
    order = np.argsort(-scores, kind="stable")
    M = np.triu(iou_matrix(boxes[order], boxes[order]), 1)
    return order[M.max(0, initial=0) <= thr]


def ap_all(scores, tp, n_gt):
    o = np.argsort(-scores, kind="stable")
    c = np.cumsum(tp[o])
    prec = c / np.arange(1, len(o) + 1)
    rec = c / n_gt
    env = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(rec, prepend=0) * env))


def _wh_iou(a, b):
    inter = np.minimum(a[:, None, 0], b[None, :, 0]) * np.minimum(a[:, None, 1], b[None, :, 1])
    return inter / (a.prod(1)[:, None] + b.prod(1)[None] - inter)


def anchor_kmeans(wh, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    A = wh[rng.choice(len(wh), k, replace=False)].astype(float)
    for _ in range(iters):
        a = _wh_iou(wh, A).argmax(1)
        oh = np.eye(k)[a]
        cnt = oh.sum(0)
        A = np.where(cnt[:, None] > 0, oh.T @ wh / np.maximum(cnt, 1)[:, None], A)
    return A, float(_wh_iou(wh, A).max(1).mean())
