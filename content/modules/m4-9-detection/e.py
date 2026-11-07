import time
import torch


def box_iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def nms_loop(boxes, scores, thr):
    ...


def fast_nms(boxes, scores, thr):
    ...


def batched_nms(boxes, scores, labels, thr):
    ...
