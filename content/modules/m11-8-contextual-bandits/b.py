import math
import random


class BayesLinReg:
    def __init__(self, d, lam=1.0, sigma=1.0):
        ...

    def update(self, x, r):
        ...

    def mean(self, x):
        ...

    def var(self, x):
        ...


def news(T, policy, seed=0, c=1.0):
    ...


print(news(2000, "linucb"), news(2000, "random"))
