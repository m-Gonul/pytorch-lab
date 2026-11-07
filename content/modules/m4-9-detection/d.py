import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def shapes_dataset(n, seed, size=64):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij")
    imgs = torch.zeros(n, 1, size, size)
    targets = []
    for i in range(n):
        k = int(torch.randint(1, 4, (1,), generator=g))
        boxes, labels, tries = [], [], 0
        while len(boxes) < k and tries < 50:
            tries += 1
            s = float(torch.randint(8, 25, (1,), generator=g))
            cx = float(torch.rand(1, generator=g)) * (size - s) + s / 2
            cy = float(torch.rand(1, generator=g)) * (size - s) + s / 2
            b = [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]
            if any(not (b[2] <= o[0] or o[2] <= b[0] or b[3] <= o[1] or o[3] <= b[1]) for o in boxes):
                continue
            c = int(torch.randint(0, 3, (1,), generator=g))
            if c == 0:
                m = (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])
            elif c == 1:
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (s / 2) ** 2
            else:
                m = ((xx - cx).abs() + (yy - cy).abs()) <= s / 2
            imgs[i, 0][m] = 1.0
            boxes.append(b)
            labels.append(c)
        targets.append((torch.tensor(boxes), torch.tensor(labels)))
    imgs += 0.1 * torch.randn(imgs.shape, generator=g)
    return imgs, targets


def box_iou(a, b):
    ...


def encode(boxes, labels, S, anchors, C, img=64):
    ...


def decode(p, anchors, img=64, conf=0.5):
    ...


def nms(boxes, scores, thr):
    ...


class TinyYOLO(nn.Module):
    def __init__(self, A=2, C=3):
        super().__init__()
        ...

    def forward(self, x):
        ...


def yolo_loss(p, t, l_coord=5.0, l_noobj=0.5):
    ...


def mean_ap(dets, gts, C, thr=0.5):
    ...
