import numpy as np


def iou_matrix(A, B):
    ...


def nms(boxes, scores, thr):
    ...


def fast_nms(boxes, scores, thr):
    ...


def ap_all(scores, tp, n_gt):
    ...


def anchor_kmeans(wh, k, iters=30, seed=0):
    ...
