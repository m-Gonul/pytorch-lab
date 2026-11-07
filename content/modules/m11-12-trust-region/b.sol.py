import math


def rosen(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def rosen_grad(p):
    x, y = p
    return [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)]


def rosen_hess(p):
    x, y = p
    return [[2 - 400 * (y - x * x) + 800 * x * x, -400 * x], [-400 * x, 200.0]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _mv(H, v):
    return [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]]


def _norm(v):
    return math.sqrt(_dot(v, v))


def gd_backtracking(p0, n_iter):
    p, out = list(p0), [list(p0)]
    for _ in range(n_iter):
        g = rosen_grad(p)
        t, f0, gg = 1.0, rosen(p), _dot(g, g)
        while rosen([p[0] - t * g[0], p[1] - t * g[1]]) > f0 - 1e-4 * t * gg:
            t /= 2
        p = [p[0] - t * g[0], p[1] - t * g[1]]
        out.append(p)
    return out


def dogleg(g, H, delta):
    gn = _norm(g)
    gHg = _dot(g, _mv(H, g))
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    if not (det > 0 and H[0][0] > 0):
        tau = 1.0 if gHg <= 0 else min(gn ** 3 / (delta * gHg), 1.0)
        return [-tau * delta * g[0] / gn, -tau * delta * g[1] / gn]
    pB = [-(H[1][1] * g[0] - H[0][1] * g[1]) / det, -(-H[1][0] * g[0] + H[0][0] * g[1]) / det]
    if _norm(pB) <= delta:
        return pB
    pU = [-(gn * gn / gHg) * g[0], -(gn * gn / gHg) * g[1]]
    if _norm(pU) >= delta:
        return [-delta * g[0] / gn, -delta * g[1] / gn]
    d = [pB[0] - pU[0], pB[1] - pU[1]]
    a, b, c = _dot(d, d), 2 * _dot(pU, d), _dot(pU, pU) - delta * delta
    tau = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return [pU[0] + tau * d[0], pU[1] + tau * d[1]]


def trust_region(p0, n_iter, delta0=1.0, delta_max=10.0, eta=0.1):
    p, delta, out = list(p0), delta0, [list(p0)]
    for _ in range(n_iter):
        g, H = rosen_grad(p), rosen_hess(p)
        if _norm(g) < 1e-12:
            out.append(p)
            continue
        s = dogleg(g, H, delta)
        pred = -(_dot(g, s) + 0.5 * _dot(s, _mv(H, s)))
        new = [p[0] + s[0], p[1] + s[1]]
        rho = (rosen(p) - rosen(new)) / pred if pred > 0 else -1.0
        if rho < 0.25:
            delta /= 4
        elif rho > 0.75 and abs(_norm(s) - delta) < 1e-9 * max(1.0, delta):
            delta = min(2 * delta, delta_max)
        if rho > eta:
            p = new
        out.append(p)
    return out


print(trust_region([-1.2, 1.0], 50)[-1], gd_backtracking([-1.2, 1.0], 50)[-1])
