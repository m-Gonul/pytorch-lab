import math


def f(p):
    x, y = p
    return x * x - y * y + 0.25 * y ** 4


def grad(p):
    x, y = p
    return [2 * x, -2 * y + y ** 3]


def hess(p):
    x, y = p
    return [[2.0, 0.0], [0.0, -2.0 + 3 * y * y]]


def _solve(H, g):
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    return [(H[1][1] * g[0] - H[0][1] * g[1]) / det, (-H[1][0] * g[0] + H[0][0] * g[1]) / det]


def newton(p0, n):
    p, out = list(p0), [list(p0)]
    for _ in range(n):
        s = _solve(hess(p), grad(p))
        p = [p[0] - s[0], p[1] - s[1]]
        out.append(p)
    return out


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _mv(H, v):
    return [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]]


def _step(g, H, delta):
    gn = math.sqrt(_dot(g, g))
    gHg = _dot(g, _mv(H, g))
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    if det > 0 and H[0][0] > 0:
        pB = _solve(H, g)
        pB = [-pB[0], -pB[1]]
        if math.sqrt(_dot(pB, pB)) <= delta:
            return pB
    tau = 1.0 if gHg <= 0 else min(gn ** 3 / (delta * gHg), 1.0)
    return [-tau * delta * g[0] / gn, -tau * delta * g[1] / gn]


def trust_region_saddle(p0, n, delta0=0.5):
    p, delta, out = list(p0), delta0, [list(p0)]
    for _ in range(n):
        g, H = grad(p), hess(p)
        if math.sqrt(_dot(g, g)) < 1e-12:
            out.append(p)
            continue
        s = _step(g, H, delta)
        pred = -(_dot(g, s) + 0.5 * _dot(s, _mv(H, s)))
        new = [p[0] + s[0], p[1] + s[1]]
        rho = (f(p) - f(new)) / pred if pred > 0 else -1.0
        if rho < 0.25:
            delta /= 4
        elif rho > 0.75 and abs(math.sqrt(_dot(s, s)) - delta) < 1e-9:
            delta = min(2 * delta, 5.0)
        if rho > 0.1:
            p = new
        out.append(p)
    return out


print(newton([1.0, 0.01], 20)[-1], trust_region_saddle([1.0, 0.01], 50)[-1])
