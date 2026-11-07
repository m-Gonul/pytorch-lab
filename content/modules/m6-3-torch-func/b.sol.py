import math


class Dual:
    def __init__(self, val, dot=0.0):
        self.val, self.dot = val, dot

    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val + o.val, self.dot + o.dot)

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val - o.val, self.dot - o.dot)

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val * o.val, self.dot * o.val + self.val * o.dot)

    __rmul__ = __mul__


def dsin(d):
    return Dual(math.sin(d.val), math.cos(d.val) * d.dot)


def dexp(d):
    e = math.exp(d.val)
    return Dual(e, e * d.dot)


def jvp(f, x, v):
    out = f([Dual(a, b) for a, b in zip(x, v)])
    return [o.val for o in out], [o.dot for o in out]


def jacobian_fwd(f, x):
    n = len(x)
    cols = [jvp(f, x, [1.0 if i == j else 0.0 for i in range(n)])[1] for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(len(cols[0]))]


f = lambda x: [x[0] * x[1], x[0] * x[0]]
print(jvp(f, [1.0, 2.0], [1.0, 1.0]))    # ([2, 1], [3, 2])
print(jacobian_fwd(f, [1.0, 2.0]))       # [[2, 1], [2, 0]]
