import math


class Dual:
    def __init__(self, val, dot=0.0):
        self.val, self.dot = val, dot

    def __add__(self, o):
        ...

    __radd__ = __add__

    def __sub__(self, o):
        ...

    def __mul__(self, o):
        ...

    __rmul__ = __mul__


def dsin(d):
    ...


def dexp(d):
    ...


def jvp(f, x, v):
    ...


def jacobian_fwd(f, x):
    ...


f = lambda x: [x[0] * x[1], x[0] * x[0]]
print(jvp(f, [1.0, 2.0], [1.0, 1.0]))    # ([2, 1], [3, 2])
print(jacobian_fwd(f, [1.0, 2.0]))       # [[2, 1], [2, 0]]
