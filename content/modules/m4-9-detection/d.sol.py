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
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def encode(boxes, labels, S, anchors, C, img=64):
    A = len(anchors)
    t = torch.zeros(S, S, A, 5 + C)
    cell = img / S
    for b, c in zip(boxes, labels):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        w, h = b[2] - b[0], b[3] - b[1]
        i, j = min(int(cy // cell), S - 1), min(int(cx // cell), S - 1)
        inter = torch.min(anchors[:, 0], w) * torch.min(anchors[:, 1], h)
        a = int((inter / (anchors[:, 0] * anchors[:, 1] + w * h - inter)).argmax())
        t[i, j, a, 0] = cx / cell - j
        t[i, j, a, 1] = cy / cell - i
        t[i, j, a, 2] = torch.log(w / anchors[a, 0])
        t[i, j, a, 3] = torch.log(h / anchors[a, 1])
        t[i, j, a, 4] = 1
        t[i, j, a, 5 + int(c)] = 1
    return t


def decode(p, anchors, img=64, conf=0.5):
    S = p.shape[0]
    cell = img / S
    ii, jj = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    cx = (jj[:, :, None] + torch.sigmoid(p[..., 0])) * cell
    cy = (ii[:, :, None] + torch.sigmoid(p[..., 1])) * cell
    w = anchors[:, 0] * torch.exp(p[..., 2])
    h = anchors[:, 1] * torch.exp(p[..., 3])
    sc, lab = (torch.sigmoid(p[..., 4])[..., None] * torch.softmax(p[..., 5:], -1)).max(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    m = sc > conf
    return boxes[m], sc[m], lab[m]


def nms(boxes, scores, thr):
    order = scores.argsort(descending=True)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        order = order[1:][box_iou(boxes[i][None], boxes[order[1:]])[0] <= thr]
    return torch.tensor(keep, dtype=torch.long)


class TinyYOLO(nn.Module):
    def __init__(self, A=2, C=3):
        super().__init__()
        cb = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.LeakyReLU(0.1))
        self.f = nn.Sequential(cb(1, 16), nn.MaxPool2d(2), cb(16, 32), nn.MaxPool2d(2), cb(32, 64), nn.MaxPool2d(2), cb(64, 64), nn.Conv2d(64, A * (5 + C), 1))
        self.A, self.C = A, C

    def forward(self, x):
        y = self.f(x)
        N, _, S, _ = y.shape
        return y.view(N, self.A, 5 + self.C, S, S).permute(0, 3, 4, 1, 2)


def yolo_loss(p, t, l_coord=5.0, l_noobj=0.5):
    obj = t[..., 4] == 1
    xy = F.mse_loss(torch.sigmoid(p[..., :2][obj]), t[..., :2][obj], reduction="sum")
    wh = F.mse_loss(p[..., 2:4][obj], t[..., 2:4][obj], reduction="sum")
    bce = F.binary_cross_entropy_with_logits(p[..., 4], t[..., 4], reduction="none")
    o = bce[obj].sum() + l_noobj * bce[~obj].sum()
    c = F.cross_entropy(p[..., 5:][obj], t[..., 5:][obj].argmax(-1), reduction="sum")
    return (l_coord * (xy + wh) + o + c) / p.shape[0]


def mean_ap(dets, gts, C, thr=0.5):
    aps = []
    for cls in range(C):
        recs, n_gt = [], 0
        for (b, s, l), (gb, gl) in zip(dets, gts):
            g = gb[gl == cls]
            n_gt += len(g)
            bc, sc = b[l == cls], s[l == cls]
            used = torch.zeros(len(g), dtype=torch.bool)
            for k in sc.argsort(descending=True):
                ok = False
                if len(g):
                    iou = box_iou(bc[k][None], g)[0]
                    iou[used] = -1
                    m = int(iou.argmax())
                    if iou[m] >= thr:
                        used[m] = True
                        ok = True
                recs.append((float(sc[k]), ok))
        if n_gt == 0:
            continue
        recs.sort(key=lambda r: -r[0])
        tp = torch.tensor([r[1] for r in recs], dtype=torch.float)
        if len(tp) == 0:
            aps.append(0.0)
            continue
        c = tp.cumsum(0)
        prec = c / torch.arange(1, len(tp) + 1)
        rec = c / n_gt
        env = torch.flip(torch.cummax(torch.flip(prec, [0]), 0).values, [0])
        aps.append(float(((rec - torch.cat([torch.zeros(1), rec[:-1]])) * env).sum()))
    return sum(aps) / len(aps)
