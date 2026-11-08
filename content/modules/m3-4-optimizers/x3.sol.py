import math


def power(t, lr0, s, c=1):
    return lr0 / (1 + t / s) ** c


def exponential(t, lr0, s):
    return lr0 * 0.1 ** (t / s)


def piecewise(t, bounds, values):
    for b, v in zip(bounds, values):
        if t < b:
            return v
    return values[-1]


def one_cycle(t, T, lr0, lr1, tail=0.1):
    half = (1 - tail) * T / 2
    if t < half:
        return lr0 + (lr1 - lr0) * t / half
    if t < 2 * half:
        return lr1 - (lr1 - lr0) * (t - half) / half
    return lr0 - (lr0 - lr0 / 1000) * (t - 2 * half) / (T - 2 * half)


def plateau(losses, lr0, factor=0.5, patience=3):
    lr, best, wait, out = lr0, float("inf"), 0, []
    for L in losses:
        if L < best:
            best, wait = L, 0
        else:
            wait += 1
            if wait >= patience:
                lr *= factor
                wait = 0
        out.append(lr)
    return out
