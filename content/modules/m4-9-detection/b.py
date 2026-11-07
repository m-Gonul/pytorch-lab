def iou(a, b):
    ...


def nms(boxes, scores, thr):
    ...


def match(det_boxes, det_scores, gt_boxes, thr=0.5):
    ...


def pr_curve(tp, n_gt):
    ...


def ap11(prec, rec):
    ...


def ap_all(prec, rec):
    ...


print(iou((1, 1, 5, 5), (3, 2, 7, 6)))
