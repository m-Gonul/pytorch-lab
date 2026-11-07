import math


def mean(xs):
    s = 0.0
    for v in xs:
        s += v
    return s / len(xs)


def var(xs, ddof=0):
    m = mean(xs)
    ss = 0.0
    for v in xs:
        ss += (v - m) ** 2
    return ss / (len(xs) - ddof)


def bernoulli_ll(ys, p):
    ll = 0.0
    for y in ys:
        q = p if y == 1 else 1 - p
        if q == 0:
            return -math.inf          # veri modele göre imkânsız
        ll += math.log(q)
    return ll


def categorical_mle(rolls, K):
    counts = [0] * K
    for r in rolls:
        counts[r - 1] += 1
    return [c / len(rolls) for c in counts]


print(mean([2, 4, 6]), var([2, 4, 6]), var([2, 4, 6], ddof=1))   # 4.0 2.666… 4.0
print(bernoulli_ll([1, 0, 1, 1], 0.75))                          # -2.2493…
print(categorical_mle([1, 3, 3, 6, 3, 1], 6))
