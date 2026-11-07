import math
import numpy as np


def cascade_rates(stages):
    D = F = 1.0
    for d, f in stages:
        D *= d
        F *= f
    return D, F


def stages_needed(f_i, F_target):
    return math.ceil(math.log(F_target) / math.log(f_i) - 1e-12)


def roc(scores, y):
    scores, y = np.asarray(scores, float), np.asarray(y, float)
    o = np.argsort(-scores, kind="stable")
    s, t = scores[o], y[o]
    last = np.r_[np.diff(s) != 0, True]
    tp, fp = np.cumsum(t)[last], np.cumsum(1 - t)[last]
    tpr = np.r_[0, tp / t.sum()]
    fpr = np.r_[0, fp / (1 - t).sum()]
    return fpr, tpr, float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


print(cascade_rates([(0.99, 0.3)] * 10))
