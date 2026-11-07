import math


def mean(xs):
    ...


def var(xs, ddof=0):
    ...


def bernoulli_ll(ys, p):
    ...


def categorical_mle(rolls, K):
    ...


print(mean([2, 4, 6]), var([2, 4, 6]), var([2, 4, 6], ddof=1))   # 4.0 2.666… 4.0
print(bernoulli_ll([1, 0, 1, 1], 0.75))                          # -2.2493…
print(categorical_mle([1, 3, 3, 6, 3, 1], 6))
