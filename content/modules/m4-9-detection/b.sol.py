def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, thr):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) <= thr for j in keep):
            keep.append(i)
    return keep


def match(det_boxes, det_scores, gt_boxes, thr=0.5):
    used = [False] * len(gt_boxes)
    out = []
    for i in sorted(range(len(det_boxes)), key=lambda i: -det_scores[i]):
        best, bj = -1.0, -1
        for j, g in enumerate(gt_boxes):
            if not used[j]:
                v = iou(det_boxes[i], g)
                if v > best:
                    best, bj = v, j
        ok = bj >= 0 and best >= thr
        if ok:
            used[bj] = True
        out.append(ok)
    return out


def pr_curve(tp, n_gt):
    prec, rec, c = [], [], 0
    for k, t in enumerate(tp):
        c += bool(t)
        prec.append(c / (k + 1))
        rec.append(c / n_gt)
    return prec, rec


def ap11(prec, rec):
    s = 0.0
    for k in range(11):
        r = k / 10
        s += max([p for p, q in zip(prec, rec) if q >= r - 1e-12] or [0.0])
    return s / 11


def ap_all(prec, rec):
    env = prec[:]
    for k in range(len(env) - 2, -1, -1):
        env[k] = max(env[k], env[k + 1])
    s, prev = 0.0, 0.0
    for p, r in zip(env, rec):
        s += (r - prev) * p
        prev = r
    return s


print(iou((1, 1, 5, 5), (3, 2, 7, 6)))
