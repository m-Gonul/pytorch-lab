import numpy as np


def iou_m(A, B):
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    ar = lambda X: (X[:, 2:] - X[:, :2]).prod(-1)
    return inter / (ar(A)[:, None] + ar(B)[None] - inter)


def soft_nms(boxes, scores, sigma=0.5, thresh=0.001):
    ...


def match_ap(dets, gts, thr=0.5):
    ...
