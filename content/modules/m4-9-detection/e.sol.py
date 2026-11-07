import time
import torch


def box_iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def nms_loop(boxes, scores, thr):
    order = scores.argsort(descending=True, stable=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        order = order[1:][box_iou(boxes[i][None], boxes[order[1:]])[0] <= thr]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long)


def fast_nms(boxes, scores, thr):
    order = scores.argsort(descending=True, stable=True)
    b = boxes[order]
    M = torch.triu(box_iou(b, b), 1)
    return order[M.max(0).values <= thr]


def batched_nms(boxes, scores, labels, thr):
    off = labels.to(boxes.dtype)[:, None] * (boxes.max() + 1)
    return nms_loop(boxes + off, scores, thr)
