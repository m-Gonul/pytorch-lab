import math


def frechet_1d(m1, s1, m2, s2):
    return (m1 - m2) ** 2 + s1 ** 2 + s2 ** 2 - 2 * s1 * s2


def inception_score(P):
    C = len(P[0])
    py = [sum(p[c] for p in P) / len(P) for c in range(C)]
    kl = [sum(p[c] * math.log(p[c] / py[c]) for c in range(C) if p[c] > 0) for p in P]
    return math.exp(sum(kl) / len(kl))


def adain(x, ys, yb, eps=0.0):
    m = sum(x) / len(x)
    s = math.sqrt(sum((v - m) ** 2 for v in x) / len(x) + eps)
    return [ys * (v - m) / s + yb for v in x]


def truncate(w, w_avg, psi):
    return [a + psi * (v - a) for v, a in zip(w, w_avg)]


def fade_in(old, new, alpha):
    return [(1 - alpha) * o + alpha * n for o, n in zip(old, new)]


def he_scale(fan_in, gain=2 ** 0.5):
    return gain / math.sqrt(fan_in)
