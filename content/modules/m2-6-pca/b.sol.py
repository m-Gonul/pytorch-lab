import math


def covariance(X):
    n, d = len(X), len(X[0])
    m = [sum(r[j] for r in X) / n for j in range(d)]
    return [[sum((r[i] - m[i]) * (r[j] - m[j]) for r in X) / (n - 1) for j in range(d)] for i in range(d)]


def matvec(C, v):
    return [sum(c * x for c, x in zip(row, v)) for row in C]


def power_iteration(C, iters=200):
    d = len(C)
    v = [1 / math.sqrt(d)] * d
    for _ in range(iters):
        w = matvec(C, v)
        nrm = math.sqrt(sum(x * x for x in w))
        if nrm == 0:
            break
        v = [x / nrm for x in w]
    k = max(range(d), key=lambda i: abs(v[i]))
    if v[k] < 0:
        v = [-x for x in v]
    lam = sum(a * b for a, b in zip(v, matvec(C, v)))
    return lam, v


def top_k(C, k, iters=200):
    C = [row[:] for row in C]
    lams, vecs = [], []
    for _ in range(k):
        lam, v = power_iteration(C, iters)
        lams.append(lam)
        vecs.append(v)
        C = [[C[i][j] - lam * v[i] * v[j] for j in range(len(C))] for i in range(len(C))]
    return lams, vecs


print(power_iteration([[2.0, 1.0], [1.0, 2.0]]))   # (3.0, [0.7071, 0.7071])
print(top_k([[2.0, 1.0], [1.0, 2.0]], 2))
